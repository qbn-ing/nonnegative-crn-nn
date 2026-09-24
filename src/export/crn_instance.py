from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from export.spec_utils import assert_frozen_rates, is_finite_number, spec_to_payload


class CRNInstanceError(ValueError):
    """Raised when a compiled model spec cannot be lowered to a concrete CRN instance."""


@dataclass(frozen=True)
class CRNInstanceOptions:
    """Options for lowering one frozen model spec and one input sample to a CRN instance."""

    input_values: Sequence[float] | None = None
    initial_z_free: float = 1.0
    require_frozen_rates: bool = True


@dataclass(frozen=True)
class CRNReaction:
    """One mass-action reaction with multiplicity-preserving reactant/product species."""

    reactants: tuple[str, ...]
    products: tuple[str, ...]
    rate_key: str
    rate_value: float
    name: str | None = None

    @property
    def order(self) -> int:
        return len(self.reactants)


@dataclass(frozen=True)
class CRNInstance:
    """Concrete mass-action CRN generated from one compiled model spec and one input sample."""

    model_name: str
    species: tuple[str, ...]
    initials: Mapping[str, float]
    parameters: Mapping[str, float]
    reactions: tuple[CRNReaction, ...]
    final_output_species: tuple[str, ...]
    metadata: Mapping[str, Any]
    inactive_output_species: str | None = None
    final_response_species: tuple[str, ...] = ()

    @property
    def species_count(self) -> int:
        return len(self.species)

    @property
    def reaction_count(self) -> int:
        return len(self.reactions)

    @property
    def parameter_count(self) -> int:
        return len(self.parameters)

    def initial_value(self, species: str, *, default: float = 0.0) -> float:
        return float(self.initials.get(species, default))

    def write_json(self, path: str | Path, *, encoding: str = "utf-8") -> Path:
        """Write a small JSON-compatible structural summary for debugging."""

        import json

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "species": list(self.species),
            "initials": dict(self.initials),
            "parameters": dict(self.parameters),
            "reactions": [
                {
                    "reactants": list(reaction.reactants),
                    "products": list(reaction.products),
                    "rate_key": reaction.rate_key,
                    "rate_value": reaction.rate_value,
                    "name": reaction.name,
                }
                for reaction in self.reactions
            ],
            "final_output_species": list(self.final_output_species),
            "inactive_output_species": self.inactive_output_species,
            "final_response_species": list(self.final_response_species),
            "metadata": spec_to_payload(self.metadata),
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding=encoding,
        )
        return output_path


def build_crn_instance(
    spec: Any,
    options: CRNInstanceOptions | None = None,
) -> CRNInstance:
    """Lower a frozen model specification to a concrete CRN instance."""

    opts = options or CRNInstanceOptions()
    payload = assert_frozen_rates(spec) if opts.require_frozen_rates else spec_to_payload(spec)

    if not isinstance(payload, Mapping):
        raise CRNInstanceError("Model spec payload must be a mapping.")

    builder = _CRNInstanceBuilder(payload, opts)
    return builder.build()


class _ParameterTable:
    def __init__(self) -> None:
        self._values: "OrderedDict[str, float]" = OrderedDict()
        self._raw_to_exported: dict[tuple[str | None, str], str] = {}

    @property
    def values(self) -> Mapping[str, float]:
        return self._values

    def register(
        self,
        rate: Mapping[str, Any],
        *,
        context: str,
        namespace: str | None = None,
    ) -> tuple[str, float]:
        if not isinstance(rate, Mapping):
            raise CRNInstanceError(
                f"{context}: expected a rate mapping, got {type(rate).__name__}."
            )

        raw_key = rate.get("key", rate.get("rate_key", rate.get("name")))
        if raw_key is None:
            raise CRNInstanceError(f"{context}: rate mapping has no key/rate_key/name field.")

        value = rate.get("value")
        if not is_finite_number(value):
            raise CRNInstanceError(f"{context}: rate {raw_key!r} has no finite numeric value.")

        raw_key_str = str(raw_key)
        namespace_key = safe_crn_identifier(namespace) if namespace is not None else None
        lookup_key = (namespace_key, raw_key_str)
        exported_key = self._raw_to_exported.get(lookup_key)
        new_value = float(value)

        if exported_key is None:
            if namespace_key is None:
                base_key = raw_key_str
            else:
                base_key = f"{namespace_key}_{raw_key_str}"

            exported_key = _unique_identifier(
                safe_crn_identifier(base_key),
                used=set(self._values),
            )
            self._raw_to_exported[lookup_key] = exported_key
            self._values[exported_key] = new_value
            return exported_key, new_value

        old_value = self._values[exported_key]
        if old_value != new_value:
            if namespace_key is None:
                shown_key = raw_key_str
            else:
                shown_key = f"{namespace_key}:{raw_key_str}"

            raise CRNInstanceError(
                f"{context}: repeated rate key {shown_key!r} has inconsistent "
                f"values {old_value!r} and {new_value!r}."
            )

        return exported_key, old_value


class _CRNInstanceBuilder:
    def __init__(self, payload: Mapping[str, Any], options: CRNInstanceOptions) -> None:
        self.payload = payload
        self.options = options
        self.parameters = _ParameterTable()
        self.initials: "OrderedDict[str, float]" = OrderedDict()
        self.species: set[str] = set()
        self.reactions: list[CRNReaction] = []
        self.final_output_species: list[str] = []
        self.inactive_output_species: str | None = None
        self.final_response_species: list[str] = []

    def build(self) -> CRNInstance:
        input_dim = _required_int(self.payload, "input_dim", context="model spec")
        input_species = [safe_crn_identifier(f"X_{i}") for i in range(input_dim)]
        self._add_input_initials(input_species)

        basis_payload = self.payload.get("basis")
        if basis_payload is None:
            basis_specs: Sequence[Any] = ()
            model_inputs = input_species
        else:
            if not isinstance(basis_payload, Sequence) or isinstance(
                basis_payload,
                (str, bytes, bytearray),
            ):
                raise CRNInstanceError("model spec.basis must be a sequence.")
            basis_specs = basis_payload
            if len(basis_specs) == 0:
                model_inputs = input_species
            else:
                model_inputs = self._basis_species_names(
                    basis_specs,
                    input_species,
                )
                self._add_basis_initials_and_reactions(
                    basis_specs,
                    input_species,
                    model_inputs,
                )

        rational_payload = _required_sequence(
            self.payload,
            "rational_layers",
            context="model spec",
        )
        output_spec = _required_mapping(
            self.payload,
            "output",
            context="model spec",
        )
        self._add_rational_network_reactions(
            rational_payload,
            output_spec,
            model_inputs,
        )

        metadata = self._metadata()
        return CRNInstance(
            model_name=metadata["model_name"],
            species=tuple(sorted(self.species)),
            initials=dict(self.initials),
            parameters=dict(self.parameters.values),
            reactions=tuple(self.reactions),
            final_output_species=tuple(self.final_output_species),
            metadata=metadata,
            inactive_output_species=self.inactive_output_species,
            final_response_species=tuple(self.final_response_species),
        )

    def _basis_species_names(
        self,
        basis_specs: Sequence[Any],
        input_species: Sequence[str],
    ) -> list[str]:
        names: list[str] = []
        used: set[str] = set()

        for index, basis in enumerate(basis_specs):
            basis_map = _as_mapping(basis, context=f"basis[{index}]")
            metadata = _metadata_mapping(basis_map)
            compile_mode = metadata.get("compile_mode")

            if compile_mode == "input_alias":
                input_index = _metadata_input_index(
                    metadata,
                    input_species,
                    context=f"basis[{index}].metadata",
                )
                names.append(input_species[input_index])
                continue

            raw_name = basis_map.get("output_species") or f"Phi_{index}"
            name = _unique_identifier(safe_crn_identifier(str(raw_name)), used=used)
            used.add(name)
            names.append(name)

        return names

    def _add_input_initials(self, input_species: Sequence[str]) -> None:
        values = self.options.input_values
        if values is None:
            values = [0.0 for _ in input_species]
        else:
            values = self._project_input_values_if_needed(values, expected=len(input_species))

        if len(values) != len(input_species):
            raise CRNInstanceError(
                f"Expected {len(input_species)} input concentrations, got {len(values)}."
            )

        for species, value in zip(input_species, values):
            self._set_initial(species, _finite_float(value, context=f"initial {species}"))

    def _project_input_values_if_needed(
        self,
        values: Sequence[float],
        *,
        expected: int,
    ) -> Sequence[float]:
        if len(values) == expected:
            return values

        model = _as_mapping(self.payload, context="model")
        active = model.get("active_input_indices")
        if active is None:
            structural_pruning = model.get("structural_pruning")
            if isinstance(structural_pruning, Mapping):
                active = structural_pruning.get("active_input_indices")

        if active is None:
            return values

        active_indices = tuple(int(index) for index in active)
        if len(active_indices) != expected:
            return values

        if not active_indices:
            return values

        max_index = max(active_indices)
        if len(values) <= max_index:
            return values

        return [values[index] for index in active_indices]

    def _add_basis_initials_and_reactions(
        self,
        basis_specs: Sequence[Any],
        input_species: Sequence[str],
        basis_species: Sequence[str],
    ) -> None:
        for index, (basis, output_species) in enumerate(zip(basis_specs, basis_species)):
            basis_map = _as_mapping(basis, context=f"basis[{index}]")
            metadata = _metadata_mapping(basis_map)
            compile_mode = metadata.get("compile_mode")

            if compile_mode == "input_alias":
                # This basis feature is the original input species. The network
                # can read X_i directly; no copy species or reactions are needed.
                continue

            if compile_mode == "simplex_complement_motif":
                self._add_simplex_complement_motif(
                    index=index,
                    output_species=output_species,
                    metadata=metadata,
                    input_species=input_species,
                )
                continue

            self._set_initial(output_species, 0.0)

            for term_index, term in enumerate(basis_map.get("production_terms", [])):
                term_map = _as_mapping(
                    term,
                    context=f"basis[{index}].production_terms[{term_index}]",
                )
                rate_key, rate_value = self.parameters.register(
                    _required_mapping(
                        term_map,
                        "rate",
                        context=f"basis[{index}].production_terms[{term_index}]",
                    ),
                    context=f"basis[{index}].production_terms[{term_index}].rate",
                )
                reactants = self._factor_species(
                    term_map.get("factors", []),
                    input_species,
                    basis_species,
                )
                self._add_reaction(
                    reactants,
                    [*reactants, output_species],
                    rate_key,
                    rate_value,
                    name=f"basis_{index}_production_{term_index}",
                )

            decay = basis_map.get("decay")
            if decay is not None:
                rate_key, rate_value = self.parameters.register(
                    _as_mapping(decay, context=f"basis[{index}].decay"),
                    context=f"basis[{index}].decay",
                )
                self._add_reaction(
                    [output_species],
                    [],
                    rate_key,
                    rate_value,
                    name=f"basis_{index}_decay",
                )

    def _add_simplex_complement_motif(
        self,
        *,
        index: int,
        output_species: str,
        metadata: Mapping[str, Any],
        input_species: Sequence[str],
    ) -> None:
        input_index = _metadata_input_index(
            metadata,
            input_species,
            context=f"basis[{index}].metadata",
        )

        total = _finite_float(
            metadata.get("total", 1.0),
            context=f"basis[{index}].metadata.total",
        )
        rho = _finite_float(
            metadata.get("rho", 1.0),
            context=f"basis[{index}].metadata.rho",
        )
        initial = _finite_float(
            metadata.get("initial", 1e-6),
            context=f"basis[{index}].metadata.initial",
        )
        if total <= 0:
            raise CRNInstanceError(
                f"basis[{index}].metadata.total must be positive. Got {total}."
            )
        if rho <= 0:
            raise CRNInstanceError(
                f"basis[{index}].metadata.rho must be positive. Got {rho}."
            )
        if initial < 0:
            raise CRNInstanceError(
                f"basis[{index}].metadata.initial must be non-negative. Got {initial}."
            )

        reservoir_raw = metadata.get("reservoir_species") or f"R_simplex_{input_index}"
        reservoir_species = safe_crn_identifier(str(reservoir_raw))
        input_name = input_species[input_index]

        self._set_initial(output_species, initial)
        self._set_initial(reservoir_species, total)

        rate_key, rate_value = self.parameters.register(
            {
                "key": f"simplex_xbar{input_index}_rho",
                "value": rho,
                "trainable": False,
                "shared": False,
            },
            context=f"basis[{index}].metadata.rho",
        )

        # dXbar/dt = rho * Xbar * (R - X - Xbar)
        self._add_reaction(
            [reservoir_species, output_species],
            [reservoir_species, output_species, output_species],
            rate_key,
            rate_value,
            name=f"basis_{index}_simplex_autocatalysis",
        )
        self._add_reaction(
            [input_name, output_species],
            [input_name],
            rate_key,
            rate_value,
            name=f"basis_{index}_simplex_input_inhibition",
        )
        self._add_reaction(
            [output_species, output_species],
            [output_species],
            rate_key,
            rate_value,
            name=f"basis_{index}_simplex_self_limiting",
        )


    def _add_rational_network_reactions(
        self,
        layer_specs: Sequence[Any],
        output: Mapping[str, Any],
        first_layer_inputs: Sequence[str],
    ) -> None:
        if len(layer_specs) == 0:
            raise CRNInstanceError(
                "model spec.rational_layers requires at least one layer."
            )

        current_inputs = list(first_layer_inputs)
        layer_count = len(layer_specs)

        for layer_index, layer in enumerate(layer_specs):
            layer_map = _as_mapping(
                layer,
                context=f"rational_layers[{layer_index}]",
            )
            input_dim = _required_int(
                layer_map,
                "input_dim",
                context=f"rational_layers[{layer_index}]",
            )
            width = _required_int(
                layer_map,
                "width",
                context=f"rational_layers[{layer_index}]",
            )
            if input_dim != len(current_inputs):
                raise CRNInstanceError(
                    f"rational_layers[{layer_index}].input_dim={input_dim} "
                    f"does not match {len(current_inputs)} upstream species."
                )

            evidence = _required_mapping(
                layer_map,
                "evidence",
                context=f"rational_layers[{layer_index}]",
            )
            response = _required_mapping(
                layer_map,
                "response",
                context=f"rational_layers[{layer_index}]",
            )
            include_bias = layer_map.get(
                "include_bias",
                layer_map.get("include_constant_channel", False),
            )
            if not isinstance(include_bias, bool):
                raise CRNInstanceError(
                    f"rational_layers[{layer_index}]."
                    "include_bias must be a bool."
                )
            layer_inputs = list(current_inputs)
            evidence_biases = evidence.get("biases", [])
            if evidence_biases and not include_bias:
                raise CRNInstanceError(
                    f"rational_layers[{layer_index}] exports evidence biases "
                    "but include_bias is false."
                )
            bias_catalyst_species: str | None = None
            if evidence_biases:
                catalyst_initial = _finite_float(
                    layer_map.get(
                        "bias_catalyst_initial",
                        layer_map.get("constant_value", 1.0),
                    ),
                    context=(
                        f"rational_layers[{layer_index}]."
                        "bias_catalyst_initial"
                    ),
                )
                if catalyst_initial <= 0.0:
                    raise CRNInstanceError(
                        f"rational_layers[{layer_index}]."
                        "bias_catalyst_initial "
                        "must be positive."
                    )
                if catalyst_initial != 1.0:
                    raise CRNInstanceError(
                        f"rational_layers[{layer_index}]."
                        "bias_catalyst_initial must equal 1.0."
                    )
                bias_catalyst_species = f"L{layer_index}_C_bias"
                self._set_initial(
                    bias_catalyst_species,
                    catalyst_initial,
                )
            evidence_input_dim = int(
                evidence.get("input_dim", evidence.get("n_basis", -1))
            )
            if evidence_input_dim != len(layer_inputs):
                raise CRNInstanceError(
                    f"rational_layers[{layer_index}].evidence.input_dim="
                    f"{evidence_input_dim} does not match "
                    f"{len(layer_inputs)} variable inputs."
                )
            evidence_output_dim = int(
                evidence.get("output_dim", evidence.get("n_classes", -1))
            )
            if evidence_output_dim != width:
                raise CRNInstanceError(
                    f"rational_layers[{layer_index}].evidence.output_dim="
                    f"{evidence_output_dim} does not match layer width "
                    f"{width}."
                )
            if int(response.get("width", -1)) != width:
                raise CRNInstanceError(
                    f"rational_layers[{layer_index}].response.width "
                    "must match layer width."
                )

            name_map = self._collect_rational_name_map(
                evidence,
                response,
                layer_index,
                layer_count,
            )
            namespace = f"L{layer_index}"

            self._add_rational_layer_initials(
                evidence,
                response,
                name_map,
                layer_index=layer_index,
                namespace=namespace,
            )
            self._add_evidence_reactions(
                evidence,
                name_map,
                layer_inputs,
                layer_index,
                namespace=namespace,
            )
            if bias_catalyst_species is not None:
                self._add_evidence_bias_reactions(
                    evidence,
                    name_map,
                    bias_catalyst_species,
                    layer_index,
                    output_dim=width,
                    namespace=namespace,
                )
            self._add_response_reactions(
                response,
                name_map,
                layer_index,
                namespace=namespace,
            )

            current_inputs = self._response_species(response, name_map)
            if len(current_inputs) != width:
                raise CRNInstanceError(
                    f"rational_layers[{layer_index}] exports "
                    f"{len(current_inputs)} response species, expected {width}."
                )

        self.final_response_species = list(current_inputs)
        self._add_output_pool(output, current_inputs)

    def _collect_rational_name_map(
        self,
        evidence: Mapping[str, Any],
        response: Mapping[str, Any],
        layer_index: int,
        layer_count: int,
    ) -> dict[str, str]:
        names: set[str] = set()
        for group in (evidence.get("species", []), response.get("species", [])):
            for species in group:
                species_map = _as_mapping(
                    species,
                    context=f"rational_layers[{layer_index}].species[]",
                )
                if species_map.get("name") is not None:
                    names.add(str(species_map["name"]))

        used: set[str] = set()
        mapping: dict[str, str] = {}
        for raw_name in sorted(names):
            base = safe_crn_identifier(raw_name)
            if layer_count > 1:
                base = f"L{layer_index}_{base}"
            mapping[raw_name] = _unique_identifier(base, used=used)
            used.add(mapping[raw_name])
        return mapping

    def _add_rational_layer_initials(
        self,
        evidence: Mapping[str, Any],
        response: Mapping[str, Any],
        name_map: Mapping[str, str],
        *,
        layer_index: int,
        namespace: str,
    ) -> None:
        for species in evidence.get("species", []):
            species_map = _as_mapping(
                species,
                context=f"rational_layers[{layer_index}].evidence.species[]",
            )
            raw_name = str(species_map.get("name"))
            self._set_initial(name_map[raw_name], 0.0)

            decay = species_map.get("decay")
            if decay is not None:
                rate_key, rate_value = self.parameters.register(
                    _as_mapping(
                        decay,
                        context=(
                            f"rational_layers[{layer_index}]."
                            f"evidence.species[{raw_name}].decay"
                        ),
                    ),
                    context=(
                        f"rational_layers[{layer_index}]."
                        f"evidence.species[{raw_name}].decay"
                    ),
                    namespace=namespace,
                )
                self._add_reaction(
                    [name_map[raw_name]],
                    [],
                    rate_key,
                    rate_value,
                    name=f"layer_{layer_index}_{raw_name}_decay",
                )

        for species in response.get("species", []):
            species_map = _as_mapping(
                species,
                context=f"rational_layers[{layer_index}].response.species[]",
            )
            raw_name = str(species_map.get("name"))
            self._set_initial(name_map[raw_name], 0.0)

    def _add_response_reactions(
        self,
        response: Mapping[str, Any],
        name_map: Mapping[str, str],
        layer_index: int,
        *,
        namespace: str,
    ) -> None:
        for reaction_index, reaction in enumerate(
            response.get("reactions", [])
        ):
            reaction_map = _as_mapping(
                reaction,
                context=(
                    f"rational_layers[{layer_index}]."
                    f"response.reactions[{reaction_index}]"
                ),
            )
            rate_key, rate_value = self.parameters.register(
                _required_mapping(
                    reaction_map,
                    "rate",
                    context=(
                        f"rational_layers[{layer_index}]."
                        f"response.reactions[{reaction_index}]"
                    ),
                ),
                context=(
                    f"rational_layers[{layer_index}]."
                    f"response.reactions[{reaction_index}].rate"
                ),
                namespace=namespace,
            )
            reactants = [
                self._map_layer_species(name, name_map)
                for name in reaction_map.get("reactants", [])
            ]
            products = [
                self._map_layer_species(name, name_map)
                for name in reaction_map.get("products", [])
            ]
            self._add_reaction(
                reactants,
                products,
                rate_key,
                rate_value,
                name=(
                    f"layer_{layer_index}_response_{reaction_index}"
                ),
            )

    def _response_species(
        self,
        response: Mapping[str, Any],
        name_map: Mapping[str, str],
    ) -> list[str]:
        indexed: list[tuple[int, str]] = []
        for species in response.get("species", []):
            species_map = _as_mapping(
                species,
                context="response.species[]",
            )
            index = _required_int(
                species_map,
                "output_index",
                context="response.species[]",
            )
            indexed.append((index, name_map[str(species_map.get("name"))]))
        indexed.sort(key=lambda item: item[0])
        return [name for _, name in indexed]

    def _add_output_pool(
        self,
        output: Mapping[str, Any],
        response_species: Sequence[str],
    ) -> None:
        n_classes = _required_int(
            output,
            "n_classes",
            context="model spec.output",
        )
        if n_classes != len(response_species):
            raise CRNInstanceError(
                f"model spec.output.n_classes={n_classes} does not match "
                f"{len(response_species)} final response species."
            )

        output_names: dict[str, str] = {}
        indexed_active: list[tuple[int, str]] = []
        total = _finite_float(
            output.get("total", self.options.initial_z_free),
            context="model spec.output.total",
        )

        for species in output.get("species", []):
            species_map = _as_mapping(
                species,
                context="model spec.output.species[]",
            )
            raw_name = str(species_map.get("name"))
            exported_name = safe_crn_identifier(raw_name)
            output_names[raw_name] = exported_name
            role = species_map.get("role")
            self._set_initial(
                exported_name,
                total if role == "inactive" else 0.0,
            )
            if role == "inactive":
                if self.inactive_output_species is not None:
                    raise CRNInstanceError(
                        "model spec.output must define exactly one inactive "
                        "output-pool species."
                    )
                self.inactive_output_species = exported_name
            if role == "active":
                class_index = _required_int(
                    species_map,
                    "class_index",
                    context="model spec.output.species[]",
                )
                indexed_active.append((class_index, exported_name))

        for reaction_index, reaction in enumerate(
            output.get("reactions", [])
        ):
            reaction_map = _as_mapping(
                reaction,
                context=f"model spec.output.reactions[{reaction_index}]",
            )
            class_index = _required_int(
                reaction_map,
                "class_index",
                context=f"model spec.output.reactions[{reaction_index}]",
            )
            if class_index < 0 or class_index >= n_classes:
                raise CRNInstanceError(
                    "model spec.output reaction class_index is out of range."
                )

            rate_key, rate_value = self.parameters.register(
                _required_mapping(
                    reaction_map,
                    "rate",
                    context=(
                        f"model spec.output.reactions[{reaction_index}]"
                    ),
                ),
                context=(
                    f"model spec.output.reactions[{reaction_index}].rate"
                ),
                namespace="O",
            )

            def map_name(name: Any) -> str:
                raw_name = str(name)
                if raw_name in output_names:
                    return output_names[raw_name]
                return response_species[class_index]

            self._add_reaction(
                [map_name(name) for name in reaction_map.get("reactants", [])],
                [map_name(name) for name in reaction_map.get("products", [])],
                rate_key,
                rate_value,
                name=f"output_{reaction_index}",
            )

        indexed_active.sort(key=lambda item: item[0])
        self.final_output_species = [
            species for _, species in indexed_active
        ]
        if self.inactive_output_species is None:
            raise CRNInstanceError(
                "model spec.output must define one inactive output-pool species."
            )

    def _add_evidence_reactions(
        self,
        evidence: Mapping[str, Any],
        name_map: Mapping[str, str],
        layer_inputs: Sequence[str],
        layer_index: int,
        *,
        namespace: str | None,
    ) -> None:
        for edge_index, edge in enumerate(evidence.get("edges", [])):
            edge_map = _as_mapping(
                edge,
                context=f"layer[{layer_index}].evidence.edges[{edge_index}]",
            )

            source_index = int(
                edge_map.get(
                    "source_input_index",
                    edge_map.get("source_basis_index", -1),
                )
            )
            if source_index < 0 or source_index >= len(layer_inputs):
                raise CRNInstanceError(
                    f"layer[{layer_index}].evidence.edges[{edge_index}]: "
                    f"source_input_index {source_index} is out of range for "
                    f"{len(layer_inputs)} input species."
                )

            source_species = layer_inputs[source_index]

            raw_target = edge_map.get("target_species")
            if raw_target is None:
                output_index = int(
                    edge_map.get(
                        "target_output_index",
                        edge_map.get("target_class_index", -1),
                    )
                )
                kind = str(edge_map.get("kind", "positive"))
                raw_target = (
                    f"E_{output_index}"
                    if kind == "positive"
                    else f"I_{output_index}"
                )

            target_species = name_map.get(str(raw_target), safe_crn_identifier(str(raw_target)))

            rate_key, rate_value = self.parameters.register(
                _required_mapping(
                    edge_map,
                    "rate",
                    context=f"layer[{layer_index}].evidence.edges[{edge_index}]",
                ),
                context=f"layer[{layer_index}].evidence.edges[{edge_index}].rate",
                namespace=namespace,
            )

            self._add_reaction(
                [source_species],
                [source_species, target_species],
                rate_key,
                rate_value,
                name=f"layer_{layer_index}_evidence_edge_{edge_index}",
            )

    def _add_evidence_bias_reactions(
        self,
        evidence: Mapping[str, Any],
        name_map: Mapping[str, str],
        catalyst_species: str,
        layer_index: int,
        *,
        output_dim: int,
        namespace: str | None,
    ) -> None:
        """Lower affine offsets using one conserved unit catalyst per layer."""

        for bias_index, bias in enumerate(evidence.get("biases", [])):
            bias_map = _as_mapping(
                bias,
                context=f"layer[{layer_index}].evidence.biases[{bias_index}]",
            )
            output_index = int(
                bias_map.get(
                    "target_output_index",
                    bias_map.get("target_class_index", -1),
                )
            )
            if output_index < 0:
                raise CRNInstanceError(
                    f"layer[{layer_index}].evidence.biases[{bias_index}] "
                    "requires target_output_index."
                )
            if output_index >= output_dim:
                raise CRNInstanceError(
                    f"layer[{layer_index}].evidence.biases[{bias_index}] "
                    f"target_output_index={output_index} is out of range for "
                    f"output_dim={output_dim}."
                )
            raw_target = bias_map.get("target_species")
            if raw_target is None:
                kind = str(bias_map.get("kind", "positive"))
                if kind not in {"positive", "negative"}:
                    raise CRNInstanceError(
                        f"layer[{layer_index}].evidence.biases[{bias_index}] "
                        "kind must be 'positive' or 'negative'."
                    )
                raw_target = (
                    f"E_{output_index}"
                    if kind == "positive"
                    else f"I_{output_index}"
                )
            target_species = name_map.get(
                str(raw_target),
                safe_crn_identifier(str(raw_target)),
            )
            rate_key, rate_value = self.parameters.register(
                _required_mapping(
                    bias_map,
                    "rate",
                    context=(
                        f"layer[{layer_index}].evidence."
                        f"biases[{bias_index}]"
                    ),
                ),
                context=(
                    f"layer[{layer_index}].evidence."
                    f"biases[{bias_index}].rate"
                ),
                namespace=namespace,
            )
            self._add_reaction(
                [catalyst_species],
                [catalyst_species, target_species],
                rate_key,
                rate_value,
                name=f"layer_{layer_index}_evidence_bias_{bias_index}",
            )

    def _factor_species(
        self,
        factors: Any,
        input_species: Sequence[str],
        basis_species: Sequence[str],
    ) -> list[str]:
        result: list[str] = []

        for factor_index, factor in enumerate(factors or []):
            factor_map = _as_mapping(factor, context=f"basis factor[{factor_index}]")
            role = str(factor_map.get("role", "input"))
            index = _required_int(factor_map, "index", context=f"basis factor[{factor_index}]")
            stoich = int(factor_map.get("stoich", 1))

            if stoich < 0:
                raise CRNInstanceError(f"basis factor[{factor_index}]: stoich must be non-negative.")

            if role == "input":
                if index < 0 or index >= len(input_species):
                    raise CRNInstanceError(
                        f"basis factor[{factor_index}]: input index {index} is out of range."
                    )
                species = input_species[index]
            elif role == "basis":
                if index < 0 or index >= len(basis_species):
                    raise CRNInstanceError(
                        f"basis factor[{factor_index}]: basis index {index} is out of range."
                    )
                species = basis_species[index]
            else:
                raw_species = factor_map.get("species", f"{role}_{index}")
                species = safe_crn_identifier(str(raw_species))

            result.extend([species] * stoich)

        return result

    def _map_layer_species(self, name: Any, name_map: Mapping[str, str]) -> str:
        raw_name = str(name)
        return name_map.get(raw_name, safe_crn_identifier(raw_name))

    def _set_initial(self, species: str, value: float) -> None:
        species = safe_crn_identifier(species)

        if species in self.initials and self.initials[species] != value:
            raise CRNInstanceError(
                f"Species {species!r} receives inconsistent initial values "
                f"{self.initials[species]!r} and {value!r}."
            )

        self.initials[species] = value
        self.species.add(species)

    def _add_reaction(
        self,
        reactants: Sequence[str],
        products: Sequence[str],
        rate_key: str,
        rate_value: float,
        *,
        name: str | None = None,
    ) -> None:
        clean_reactants = tuple(safe_crn_identifier(species) for species in reactants)
        clean_products = tuple(safe_crn_identifier(species) for species in products)

        self.reactions.append(
            CRNReaction(
                reactants=clean_reactants,
                products=clean_products,
                rate_key=safe_crn_identifier(rate_key),
                rate_value=float(rate_value),
                name=name,
            )
        )

        self.species.update(clean_reactants)
        self.species.update(clean_products)

    def _metadata(self) -> dict[str, Any]:
        return {
            "model_name": str(self.payload.get("name", "model")),
            "species_count": len(self.species),
            "reaction_count": len(self.reactions),
            "parameter_count": len(self.parameters.values),
            "input": spec_to_payload(self.options.input_values),
            "final_output_species": list(self.final_output_species),
            "inactive_output_species": self.inactive_output_species,
            "final_response_species": list(self.final_response_species),
        }



def _metadata_mapping(basis_map: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = basis_map.get("metadata")
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise CRNInstanceError(
            f"basis metadata must be a mapping when provided. Got {type(metadata).__name__}."
        )
    return metadata


def _metadata_input_index(
    metadata: Mapping[str, Any],
    input_species: Sequence[str],
    *,
    context: str,
) -> int:
    raw = metadata.get("input_index")
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise CRNInstanceError(f"{context}.input_index must be an integer.")
    if raw < 0 or raw >= len(input_species):
        raise CRNInstanceError(
            f"{context}.input_index={raw} is outside valid range [0, {len(input_species)})."
        )
    return int(raw)

def safe_crn_identifier(name: str) -> str:
    """Return a conservative CRN identifier accepted by Visual DSD and PySB-compatible exporters."""

    candidate = re.sub(r"[^A-Za-z0-9_]", "_", str(name).strip())
    candidate = re.sub(r"_+", "_", candidate).strip("_")

    if not candidate:
        candidate = "S"

    if candidate[0].isdigit():
        candidate = f"S_{candidate}"

    return candidate


def _required_mapping(mapping: Mapping[str, Any], key: str, *, context: str) -> Mapping[str, Any]:
    if key not in mapping:
        raise CRNInstanceError(f"{context}: missing required field {key!r}.")

    return _as_mapping(mapping[key], context=f"{context}.{key}")


def _required_sequence(mapping: Mapping[str, Any], key: str, *, context: str) -> Sequence[Any]:
    if key not in mapping:
        raise CRNInstanceError(f"{context}: missing required field {key!r}.")

    value = mapping[key]

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise CRNInstanceError(f"{context}.{key}: expected a sequence.")

    return value


def _required_int(mapping: Mapping[str, Any], key: str, *, context: str) -> int:
    if key not in mapping:
        raise CRNInstanceError(f"{context}: missing required field {key!r}.")

    value = mapping[key]

    if isinstance(value, bool) or not isinstance(value, int):
        raise CRNInstanceError(f"{context}.{key}: expected an integer.")

    return value


def _as_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CRNInstanceError(f"{context}: expected a mapping, got {type(value).__name__}.")

    return value


def _finite_float(value: Any, *, context: str) -> float:
    if not is_finite_number(value):
        raise CRNInstanceError(f"{context}: expected a finite number, got {value!r}.")

    return float(value)


def _unique_identifier(base: str, *, used: set[str]) -> str:
    if base not in used:
        return base

    index = 1
    while f"{base}_{index}" in used:
        index += 1

    return f"{base}_{index}"


__all__ = [
    "CRNInstanceError",
    "CRNInstanceOptions",
    "CRNInstance",
    "CRNReaction",
    "build_crn_instance",
    "safe_crn_identifier",
]
