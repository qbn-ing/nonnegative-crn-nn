from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from models.basis import RateRef, rate
from utils.validation import (
    check_non_empty_str,
    check_nonnegative_int,
    check_positive_int,
)


EvidenceKind = Literal["positive", "negative"]
_VALID_EVIDENCE_KINDS = {"positive", "negative"}


@dataclass(frozen=True, init=False)
class EvidenceSpeciesSpec:
    """One activation/inhibition aggregate species.

    ``output_index`` is the canonical name. ``class_index`` remains a
    read-only compatibility alias because intermediate E/I layers are not
    class layers.
    """

    kind: EvidenceKind
    output_index: int
    name: str
    decay: RateRef

    def __init__(
        self,
        *,
        kind: EvidenceKind,
        output_index: int | None = None,
        name: str,
        decay: RateRef,
        class_index: int | None = None,
    ) -> None:
        resolved = _resolve_alias(
            canonical_name="output_index",
            canonical_value=output_index,
            legacy_name="class_index",
            legacy_value=class_index,
        )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "output_index", resolved)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "decay", decay)
        self.__post_init__()

    def __post_init__(self) -> None:
        _check_evidence_kind(self.kind)
        check_nonnegative_int("output_index", self.output_index)
        check_non_empty_str("name", self.name)
        if not isinstance(self.decay, RateRef):
            raise TypeError("decay must be a RateRef.")
        if self.decay.value is not None and self.decay.value <= 0:
            raise ValueError("decay.value must be positive when provided.")

    @property
    def class_index(self) -> int:
        return self.output_index

    @property
    def is_positive(self) -> bool:
        return self.kind == "positive"

    @property
    def is_negative(self) -> bool:
        return self.kind == "negative"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "output_index": self.output_index,
            "class_index": self.output_index,
            "name": self.name,
            "decay": self.decay.to_dict(),
        }


@dataclass(frozen=True, init=False)
class EvidenceEdgeSpec:
    """One variable-input contribution to an E/I aggregate.

    The canonical indices describe an arbitrary layer input and output. Legacy
    basis/class names are retained as read-only aliases for old callers.
    """

    source_input_index: int
    target_output_index: int
    kind: EvidenceKind
    rate: RateRef
    source_input_species: str | None = None
    target_species: str | None = None

    def __init__(
        self,
        *,
        source_input_index: int | None = None,
        target_output_index: int | None = None,
        kind: EvidenceKind,
        rate: RateRef,
        source_input_species: str | None = None,
        target_species: str | None = None,
        source_basis_index: int | None = None,
        target_class_index: int | None = None,
        source_basis_species: str | None = None,
    ) -> None:
        resolved_source = _resolve_alias(
            canonical_name="source_input_index",
            canonical_value=source_input_index,
            legacy_name="source_basis_index",
            legacy_value=source_basis_index,
        )
        resolved_target = _resolve_alias(
            canonical_name="target_output_index",
            canonical_value=target_output_index,
            legacy_name="target_class_index",
            legacy_value=target_class_index,
        )
        resolved_species = _resolve_optional_alias(
            canonical_name="source_input_species",
            canonical_value=source_input_species,
            legacy_name="source_basis_species",
            legacy_value=source_basis_species,
        )
        object.__setattr__(self, "source_input_index", resolved_source)
        object.__setattr__(self, "target_output_index", resolved_target)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "source_input_species", resolved_species)
        object.__setattr__(self, "target_species", target_species)
        self.__post_init__()

    def __post_init__(self) -> None:
        check_nonnegative_int("source_input_index", self.source_input_index)
        check_nonnegative_int("target_output_index", self.target_output_index)
        _check_evidence_kind(self.kind)
        if not isinstance(self.rate, RateRef):
            raise TypeError("rate must be a RateRef.")
        if self.rate.value is not None and self.rate.value < 0:
            raise ValueError("rate.value must be non-negative when provided.")
        _check_optional_non_empty_str(
            "source_input_species",
            self.source_input_species,
        )
        _check_optional_non_empty_str("target_species", self.target_species)

    @property
    def source_basis_index(self) -> int:
        return self.source_input_index

    @property
    def target_class_index(self) -> int:
        return self.target_output_index

    @property
    def source_basis_species(self) -> str | None:
        return self.source_input_species

    @property
    def is_positive(self) -> bool:
        return self.kind == "positive"

    @property
    def is_negative(self) -> bool:
        return self.kind == "negative"

    @property
    def default_source_input_species(self) -> str:
        return f"Phi_{self.source_input_index}"

    @property
    def default_source_basis_species(self) -> str:
        return self.default_source_input_species

    @property
    def default_target_species(self) -> str:
        return default_evidence_species_name(
            kind=self.kind,
            output_index=self.target_output_index,
        )

    @property
    def resolved_source_input_species(self) -> str:
        return self.source_input_species or self.default_source_input_species

    @property
    def resolved_source_basis_species(self) -> str:
        return self.resolved_source_input_species

    @property
    def resolved_target_species(self) -> str:
        return self.target_species or self.default_target_species

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_input_index": self.source_input_index,
            "target_output_index": self.target_output_index,
            "source_basis_index": self.source_input_index,
            "target_class_index": self.target_output_index,
            "kind": self.kind,
            "rate": self.rate.to_dict(),
            "source_input_species": self.source_input_species,
            "source_basis_species": self.source_input_species,
            "target_species": self.target_species,
            "resolved_source_input_species": self.resolved_source_input_species,
            "resolved_source_basis_species": self.resolved_source_input_species,
            "resolved_target_species": self.resolved_target_species,
        }


@dataclass(frozen=True, init=False)
class EvidenceBiasSpec:
    """One trainable nonnegative affine offset for an E/I output."""

    target_output_index: int
    kind: EvidenceKind
    rate: RateRef
    target_species: str | None = None

    def __init__(
        self,
        *,
        target_output_index: int | None = None,
        kind: EvidenceKind,
        rate: RateRef,
        target_species: str | None = None,
        target_class_index: int | None = None,
    ) -> None:
        resolved_target = _resolve_alias(
            canonical_name="target_output_index",
            canonical_value=target_output_index,
            legacy_name="target_class_index",
            legacy_value=target_class_index,
        )
        object.__setattr__(self, "target_output_index", resolved_target)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "target_species", target_species)
        self.__post_init__()

    def __post_init__(self) -> None:
        check_nonnegative_int("target_output_index", self.target_output_index)
        _check_evidence_kind(self.kind)
        if not isinstance(self.rate, RateRef):
            raise TypeError("rate must be a RateRef.")
        if self.rate.value is not None and self.rate.value < 0:
            raise ValueError("rate.value must be non-negative when provided.")
        _check_optional_non_empty_str("target_species", self.target_species)

    @property
    def target_class_index(self) -> int:
        return self.target_output_index

    @property
    def is_positive(self) -> bool:
        return self.kind == "positive"

    @property
    def is_negative(self) -> bool:
        return self.kind == "negative"

    @property
    def default_target_species(self) -> str:
        return default_evidence_species_name(
            kind=self.kind,
            output_index=self.target_output_index,
        )

    @property
    def resolved_target_species(self) -> str:
        return self.target_species or self.default_target_species

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_output_index": self.target_output_index,
            "target_class_index": self.target_output_index,
            "kind": self.kind,
            "rate": self.rate.to_dict(),
            "target_species": self.target_species,
            "resolved_target_species": self.resolved_target_species,
        }


@dataclass(frozen=True, init=False)
class EvidenceLayerSpec:
    """Input/output specification of an arbitrary E/I aggregation layer."""

    input_dim: int
    output_dim: int
    species: tuple[EvidenceSpeciesSpec, ...]
    edges: tuple[EvidenceEdgeSpec, ...]
    biases: tuple[EvidenceBiasSpec, ...]
    name: str = "evidence"

    def __init__(
        self,
        *,
        input_dim: int | None = None,
        output_dim: int | None = None,
        species: tuple[EvidenceSpeciesSpec, ...],
        edges: tuple[EvidenceEdgeSpec, ...],
        biases: tuple[EvidenceBiasSpec, ...] = (),
        name: str = "evidence",
        n_basis: int | None = None,
        n_classes: int | None = None,
    ) -> None:
        resolved_input = _resolve_alias(
            canonical_name="input_dim",
            canonical_value=input_dim,
            legacy_name="n_basis",
            legacy_value=n_basis,
        )
        resolved_output = _resolve_alias(
            canonical_name="output_dim",
            canonical_value=output_dim,
            legacy_name="n_classes",
            legacy_value=n_classes,
        )
        object.__setattr__(self, "input_dim", resolved_input)
        object.__setattr__(self, "output_dim", resolved_output)
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "biases", biases)
        object.__setattr__(self, "name", name)
        self.__post_init__()

    def __post_init__(self) -> None:
        check_positive_int("input_dim", self.input_dim)
        check_positive_int("output_dim", self.output_dim)
        check_non_empty_str("name", self.name)
        for name, values, expected_type in (
            ("species", self.species, EvidenceSpeciesSpec),
            ("edges", self.edges, EvidenceEdgeSpec),
            ("biases", self.biases, EvidenceBiasSpec),
        ):
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be a tuple.")
            if not all(isinstance(item, expected_type) for item in values):
                raise TypeError(
                    f"{name} must contain {expected_type.__name__} objects."
                )
        self._validate_species()
        self._validate_edges()
        self._validate_biases()

    @property
    def n_basis(self) -> int:
        return self.input_dim

    @property
    def n_classes(self) -> int:
        return self.output_dim

    @property
    def positive_species(self) -> tuple[EvidenceSpeciesSpec, ...]:
        return tuple(item for item in self.species if item.is_positive)

    @property
    def negative_species(self) -> tuple[EvidenceSpeciesSpec, ...]:
        return tuple(item for item in self.species if item.is_negative)

    @property
    def positive_edges(self) -> tuple[EvidenceEdgeSpec, ...]:
        return tuple(item for item in self.edges if item.is_positive)

    @property
    def negative_edges(self) -> tuple[EvidenceEdgeSpec, ...]:
        return tuple(item for item in self.edges if item.is_negative)

    @property
    def positive_biases(self) -> tuple[EvidenceBiasSpec, ...]:
        return tuple(item for item in self.biases if item.is_positive)

    @property
    def negative_biases(self) -> tuple[EvidenceBiasSpec, ...]:
        return tuple(item for item in self.biases if item.is_negative)

    @property
    def n_species(self) -> int:
        return len(self.species)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_biases(self) -> int:
        return len(self.biases)

    @property
    def n_positive_edges(self) -> int:
        return len(self.positive_edges)

    @property
    def n_negative_edges(self) -> int:
        return len(self.negative_edges)

    def species_name(
        self,
        *,
        kind: EvidenceKind,
        output_index: int | None = None,
        class_index: int | None = None,
    ) -> str:
        index = _resolve_alias(
            canonical_name="output_index",
            canonical_value=output_index,
            legacy_name="class_index",
            legacy_value=class_index,
        )
        _check_evidence_kind(kind)
        check_nonnegative_int("output_index", index)
        for item in self.species:
            if item.kind == kind and item.output_index == index:
                return item.name
        raise KeyError(
            f"No evidence species found for kind={kind!r}, output_index={index}."
        )

    def edges_for(
        self,
        *,
        kind: EvidenceKind | None = None,
        output_index: int | None = None,
        input_index: int | None = None,
        class_index: int | None = None,
        basis_index: int | None = None,
    ) -> tuple[EvidenceEdgeSpec, ...]:
        if kind is not None:
            _check_evidence_kind(kind)
        resolved_output = _resolve_optional_index_alias(
            "output_index",
            output_index,
            "class_index",
            class_index,
        )
        resolved_input = _resolve_optional_index_alias(
            "input_index",
            input_index,
            "basis_index",
            basis_index,
        )
        return tuple(
            edge
            for edge in self.edges
            if (kind is None or edge.kind == kind)
            and (
                resolved_output is None
                or edge.target_output_index == resolved_output
            )
            and (
                resolved_input is None
                or edge.source_input_index == resolved_input
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "n_basis": self.input_dim,
            "n_classes": self.output_dim,
            "n_species": self.n_species,
            "n_edges": self.n_edges,
            "n_biases": self.n_biases,
            "n_positive_edges": self.n_positive_edges,
            "n_negative_edges": self.n_negative_edges,
            "species": [item.to_dict() for item in self.species],
            "edges": [item.to_dict() for item in self.edges],
            "biases": [item.to_dict() for item in self.biases],
        }

    def _validate_species(self) -> None:
        expected = {
            (kind, output_index)
            for kind in _VALID_EVIDENCE_KINDS
            for output_index in range(self.output_dim)
        }
        actual = {(item.kind, item.output_index) for item in self.species}
        if actual != expected:
            raise ValueError(
                "species must contain one positive and one negative species "
                f"for every output. Missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}."
            )
        names = [item.name for item in self.species]
        if len(set(names)) != len(names):
            raise ValueError("Evidence species names must be unique.")

    def _validate_edges(self) -> None:
        seen: set[tuple[EvidenceKind, int, int]] = set()
        species_lookup = {
            (item.kind, item.output_index): item.name for item in self.species
        }
        for edge in self.edges:
            if edge.source_input_index >= self.input_dim:
                raise ValueError(
                    f"source_input_index={edge.source_input_index} is out of "
                    f"range for input_dim={self.input_dim}."
                )
            if edge.target_output_index >= self.output_dim:
                raise ValueError(
                    f"target_output_index={edge.target_output_index} is out of "
                    f"range for output_dim={self.output_dim}."
                )
            key = (
                edge.kind,
                edge.target_output_index,
                edge.source_input_index,
            )
            if key in seen:
                raise ValueError(f"Duplicate evidence edge detected: {key}.")
            seen.add(key)
            if edge.target_species is not None:
                expected_target = species_lookup[
                    (edge.kind, edge.target_output_index)
                ]
                if edge.target_species != expected_target:
                    raise ValueError(
                        "Evidence edge target_species is inconsistent with "
                        "the evidence species table."
                    )

    def _validate_biases(self) -> None:
        seen: set[tuple[EvidenceKind, int]] = set()
        species_lookup = {
            (item.kind, item.output_index): item.name for item in self.species
        }
        for bias in self.biases:
            if bias.target_output_index >= self.output_dim:
                raise ValueError(
                    f"bias target_output_index={bias.target_output_index} is "
                    f"out of range for output_dim={self.output_dim}."
                )
            key = (bias.kind, bias.target_output_index)
            if key in seen:
                raise ValueError(f"Duplicate evidence bias detected: {key}.")
            seen.add(key)
            if bias.target_species is not None:
                expected_target = species_lookup[key]
                if bias.target_species != expected_target:
                    raise ValueError(
                        "Evidence bias target_species is inconsistent with "
                        "the evidence species table."
                    )


def evidence_species(
    *,
    kind: EvidenceKind,
    output_index: int | None = None,
    name: str | None = None,
    decay: RateRef | None = None,
    class_index: int | None = None,
) -> EvidenceSpeciesSpec:
    index = _resolve_alias(
        canonical_name="output_index",
        canonical_value=output_index,
        legacy_name="class_index",
        legacy_value=class_index,
    )
    if name is None:
        name = default_evidence_species_name(kind=kind, output_index=index)
    if decay is None:
        decay = rate(
            f"{name}_decay",
            value=1.0,
            trainable=False,
            shared=False,
        )
    return EvidenceSpeciesSpec(
        kind=kind,
        output_index=index,
        name=name,
        decay=decay,
    )


def evidence_edge(
    *,
    source_input_index: int | None = None,
    target_output_index: int | None = None,
    kind: EvidenceKind,
    rate_ref: RateRef,
    source_input_species: str | None = None,
    target_species: str | None = None,
    source_basis_index: int | None = None,
    target_class_index: int | None = None,
    source_basis_species: str | None = None,
) -> EvidenceEdgeSpec:
    return EvidenceEdgeSpec(
        source_input_index=source_input_index,
        target_output_index=target_output_index,
        kind=kind,
        rate=rate_ref,
        source_input_species=source_input_species,
        target_species=target_species,
        source_basis_index=source_basis_index,
        target_class_index=target_class_index,
        source_basis_species=source_basis_species,
    )


def evidence_bias(
    *,
    target_output_index: int | None = None,
    kind: EvidenceKind,
    rate_ref: RateRef,
    target_species: str | None = None,
    target_class_index: int | None = None,
) -> EvidenceBiasSpec:
    return EvidenceBiasSpec(
        target_output_index=target_output_index,
        target_class_index=target_class_index,
        kind=kind,
        rate=rate_ref,
        target_species=target_species,
    )


def evidence_layer_spec(
    *,
    input_dim: int | None = None,
    output_dim: int | None = None,
    edges: Sequence[EvidenceEdgeSpec],
    biases: Sequence[EvidenceBiasSpec] = (),
    species: Sequence[EvidenceSpeciesSpec] | None = None,
    name: str = "evidence",
    n_basis: int | None = None,
    n_classes: int | None = None,
) -> EvidenceLayerSpec:
    resolved_input = _resolve_alias(
        canonical_name="input_dim",
        canonical_value=input_dim,
        legacy_name="n_basis",
        legacy_value=n_basis,
    )
    resolved_output = _resolve_alias(
        canonical_name="output_dim",
        canonical_value=output_dim,
        legacy_name="n_classes",
        legacy_value=n_classes,
    )
    if species is None:
        species = default_evidence_species(output_dim=resolved_output)
    return EvidenceLayerSpec(
        input_dim=resolved_input,
        output_dim=resolved_output,
        species=tuple(species),
        edges=tuple(edges),
        biases=tuple(biases),
        name=name,
    )


def default_evidence_species(
    *,
    output_dim: int | None = None,
    n_classes: int | None = None,
) -> tuple[EvidenceSpeciesSpec, ...]:
    resolved = _resolve_alias(
        canonical_name="output_dim",
        canonical_value=output_dim,
        legacy_name="n_classes",
        legacy_value=n_classes,
    )
    check_positive_int("output_dim", resolved)
    return tuple(
        evidence_species(kind=kind, output_index=output_index)
        for kind in ("positive", "negative")
        for output_index in range(resolved)
    )


def default_evidence_species_name(
    *,
    kind: EvidenceKind,
    output_index: int | None = None,
    class_index: int | None = None,
) -> str:
    index = _resolve_alias(
        canonical_name="output_index",
        canonical_value=output_index,
        legacy_name="class_index",
        legacy_value=class_index,
    )
    _check_evidence_kind(kind)
    check_nonnegative_int("output_index", index)
    return f"E_{index}" if kind == "positive" else f"I_{index}"


def _resolve_alias(
    *,
    canonical_name: str,
    canonical_value: int | None,
    legacy_name: str,
    legacy_value: int | None,
) -> int:
    if canonical_value is None and legacy_value is None:
        raise TypeError(
            f"{canonical_name} is required "
            f"({legacy_name} is accepted for compatibility)."
        )
    if (
        canonical_value is not None
        and legacy_value is not None
        and canonical_value != legacy_value
    ):
        raise ValueError(
            f"{canonical_name} and {legacy_name} must agree when both are set."
        )
    value = canonical_value if canonical_value is not None else legacy_value
    assert value is not None
    return value


def _resolve_optional_alias(
    *,
    canonical_name: str,
    canonical_value: str | None,
    legacy_name: str,
    legacy_value: str | None,
) -> str | None:
    if (
        canonical_value is not None
        and legacy_value is not None
        and canonical_value != legacy_value
    ):
        raise ValueError(
            f"{canonical_name} and {legacy_name} must agree when both are set."
        )
    return canonical_value if canonical_value is not None else legacy_value


def _resolve_optional_index_alias(
    canonical_name: str,
    canonical_value: int | None,
    legacy_name: str,
    legacy_value: int | None,
) -> int | None:
    if canonical_value is None and legacy_value is None:
        return None
    resolved = _resolve_alias(
        canonical_name=canonical_name,
        canonical_value=canonical_value,
        legacy_name=legacy_name,
        legacy_value=legacy_value,
    )
    check_nonnegative_int(canonical_name, resolved)
    return resolved


def _check_optional_non_empty_str(name: str, value: str | None) -> None:
    if value is not None:
        check_non_empty_str(name, value)


def _check_evidence_kind(kind: str) -> None:
    if kind not in _VALID_EVIDENCE_KINDS:
        raise ValueError(
            f"Unsupported evidence kind: {kind!r}. "
            f"Expected one of {sorted(_VALID_EVIDENCE_KINDS)}."
        )


__all__ = [
    "EvidenceBiasSpec",
    "EvidenceEdgeSpec",
    "EvidenceKind",
    "EvidenceLayerSpec",
    "EvidenceSpeciesSpec",
    "default_evidence_species",
    "default_evidence_species_name",
    "evidence_bias",
    "evidence_edge",
    "evidence_layer_spec",
    "evidence_species",
]
