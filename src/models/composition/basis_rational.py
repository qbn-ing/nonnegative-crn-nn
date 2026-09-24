from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from models.basis.base import BaseBasisLayer
from models.basis.specs import BasisFeatureSpec, ProductionTerm, SpeciesFactor
from models.composition.rational_network import (
    RationalNetwork,
    RationalNetworkSpec,
    RationalNetworkState,
)
from models.evidence.rational_specs import RationalLayerSpec
from models.evidence.specs import EvidenceEdgeSpec, EvidenceLayerSpec
from models.gates import FeatureGate, InputGate
from utils.validation import (
    check_non_empty_str,
    check_nonnegative_float,
)


@dataclass(frozen=True)
class BasisRationalState:
    """Runtime state of basis features followed by a rational network."""

    x: torch.Tensor
    x_for_basis: torch.Tensor
    raw_phi: torch.Tensor
    phi: torch.Tensor
    network: RationalNetworkState

    def __post_init__(self) -> None:
        for name, value in (
            ("x", self.x),
            ("x_for_basis", self.x_for_basis),
            ("raw_phi", self.raw_phi),
            ("phi", self.phi),
        ):
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                raise TypeError(f"{name} must be a 2D torch.Tensor.")
        if not isinstance(self.network, RationalNetworkState):
            raise TypeError("network must be a RationalNetworkState.")
        if self.raw_phi.shape != self.phi.shape:
            raise ValueError("raw_phi and phi must have the same shape.")
        if self.network.input_h.shape != self.phi.shape:
            raise ValueError("The rational network input must match phi.")

    @property
    def depth(self) -> int:
        return self.network.depth

    @property
    def widths(self) -> tuple[int, ...]:
        return self.network.widths

    @property
    def Z(self) -> torch.Tensor:
        return self.network.Z

    @property
    def Z_free(self) -> torch.Tensor:
        return self.network.Z_free

    @property
    def scores(self) -> torch.Tensor:
        return self.network.scores

    @property
    def predicted_classes(self) -> torch.Tensor:
        return self.network.predicted_classes

    @property
    def n_classes(self) -> int:
        return self.network.n_classes

    @property
    def n_outputs(self) -> int:
        return self.n_classes

    def normalized_scores(self, eps: float = 1e-12) -> torch.Tensor:
        return self.network.normalized_scores(eps=eps)

    def detach(self) -> BasisRationalState:
        return BasisRationalState(
            x=self.x.detach(),
            x_for_basis=self.x_for_basis.detach(),
            raw_phi=self.raw_phi.detach(),
            phi=self.phi.detach(),
            network=self.network.detach(),
        )

    def to_detached_dict(self) -> dict[str, Any]:
        return {
            "x": self.x.detach().cpu(),
            "x_for_basis": self.x_for_basis.detach().cpu(),
            "raw_phi": self.raw_phi.detach().cpu(),
            "phi": self.phi.detach().cpu(),
            "network": self.network.to_detached_dict(),
        }


class BasisRationalModel(nn.Module):
    """Compose a fixed CRN-supported basis with a rational classifier."""

    def __init__(
        self,
        *,
        basis: BaseBasisLayer,
        network: RationalNetwork,
        name: str = "basis_rational_model",
        check_finite_input: bool = True,
        input_gate: InputGate | None = None,
        feature_gate: FeatureGate | None = None,
    ) -> None:
        super().__init__()
        check_non_empty_str("name", name)
        if not isinstance(basis, BaseBasisLayer):
            raise TypeError("basis must be a BaseBasisLayer.")
        if not isinstance(network, RationalNetwork):
            raise TypeError("network must be a RationalNetwork.")
        if not isinstance(check_finite_input, bool):
            raise TypeError("check_finite_input must be a bool.")
        if basis.n_basis != network.input_dim:
            raise ValueError("basis.n_basis must equal network.input_dim.")

        self.basis = basis
        self.network = network
        self.input_gate = input_gate
        self.feature_gate = feature_gate
        self.name = name
        self.check_finite_input = check_finite_input
        self._validate_gates()

    @property
    def input_dim(self) -> int:
        value = getattr(self.basis, "n_inputs", None)
        if value is None:
            raise ValueError("The basis layer does not expose n_inputs.")
        return int(value)

    @property
    def n_basis(self) -> int:
        return self.basis.n_basis

    @property
    def depth(self) -> int:
        return self.network.depth

    @property
    def widths(self) -> tuple[int, ...]:
        return self.network.widths

    @property
    def hidden_widths(self) -> tuple[int, ...]:
        return self.network.hidden_widths

    @property
    def n_classes(self) -> int:
        return self.network.n_classes

    @property
    def n_outputs(self) -> int:
        return self.n_classes

    @property
    def has_input_gate(self) -> bool:
        return self.input_gate is not None

    @property
    def has_feature_gate(self) -> bool:
        return self.feature_gate is not None

    @property
    def n_gate_parameters(self) -> int:
        total = 0
        if self.input_gate is not None:
            total += self.input_gate.n_inputs
        if self.feature_gate is not None:
            total += self.feature_gate.n_features
        return total

    def forward(self, x: torch.Tensor) -> BasisRationalState:
        self.validate_input(x)
        x_for_basis = self._apply_input_gate(x)
        raw_phi = self.basis(x_for_basis)
        phi = self._apply_feature_gate(raw_phi)
        return BasisRationalState(
            x=x,
            x_for_basis=x_for_basis,
            raw_phi=raw_phi,
            phi=phi,
            network=self.network(phi),
        )

    def gated_input(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_input(x)
        return self._apply_input_gate(x)

    def raw_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.basis(self.gated_input(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_feature_gate(self.raw_features(x))

    def output(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).Z

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).scores

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).predicted_classes

    def validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor.")
        if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"x must have shape (batch_size, {self.input_dim}). "
                f"Got {tuple(x.shape)}."
            )
        if self.check_finite_input and not torch.all(torch.isfinite(x)).item():
            raise ValueError("x must contain only finite values.")

    def rate_l1_loss(
        self,
        *,
        basis_weight: float = 1.0,
        layer_weights: Sequence[float] | None = None,
        reduction: str = "sum",
    ) -> torch.Tensor:
        check_nonnegative_float("basis_weight", basis_weight)
        loss = self._zero_scalar()
        basis_loss = getattr(self.basis, "rate_l1_loss", None)
        if basis_weight > 0.0 and callable(basis_loss):
            value = basis_loss()
            if not isinstance(value, torch.Tensor):
                raise TypeError("basis.rate_l1_loss() must return a torch.Tensor.")
            loss = loss + basis_weight * value
        return loss + self.network.rate_l1_loss(
            layer_weights=layer_weights,
            reduction=reduction,
        )

    def gate_l1_loss(self, reduction: str = "sum") -> torch.Tensor:
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'.")
        values: list[torch.Tensor] = []
        for gate in (self.input_gate, self.feature_gate):
            if gate is None:
                continue
            current = gate.gate()
            values.append(
                current.sum() if reduction == "sum" else current.mean()
            )
        return torch.stack(values).sum() if values else self._zero_scalar()

    def active_input_indices_for_export(self) -> tuple[int, ...]:
        return self._active_indices(self.input_gate, self.input_dim)

    def active_basis_indices_for_export(self) -> tuple[int, ...]:
        return self._active_indices(self.feature_gate, self.n_basis)

    def project_input_values_for_export(
        self,
        input_values: Sequence[float],
    ) -> tuple[float, ...]:
        values = tuple(float(value) for value in input_values)
        active = self.active_input_indices_for_export()
        if len(values) == len(active):
            return values
        if len(values) != self.input_dim:
            raise ValueError(
                "input_values must have the full or exported input dimension."
            )
        return tuple(values[index] for index in active)

    def model_spec(self, *, export: bool = False) -> RationalNetworkSpec:
        if export and hasattr(self.basis, "export_basis_spec"):
            basis = tuple(self.basis.export_basis_spec())  # type: ignore[attr-defined]
        else:
            basis = tuple(self.basis.compile_spec())

        layers = tuple(
            layer.export_rational_spec() if export else layer.compile_spec()
            for layer in self.network.layers
        )
        output = (
            self.network.output_layer.export_output_spec()
            if export
            else self.network.output_layer.compile_spec()
        )

        input_dim = self.input_dim
        active_inputs: tuple[int, ...] | None = None
        active_basis: tuple[int, ...] | None = None
        metadata: Mapping[str, Any] | None = None

        if export:
            (
                input_dim,
                basis,
                layers,
                active_inputs,
                active_basis,
                metadata,
            ) = self._project_export_structure(basis, layers)

        return RationalNetworkSpec(
            input_dim=input_dim,
            basis=basis,
            rational_layers=layers,
            output=output,
            name=self.name,
            active_input_indices=active_inputs,
            active_basis_indices=active_basis,
            structural_pruning=metadata,
        )

    def compile_spec(self) -> RationalNetworkSpec:
        return self.model_spec(export=False)

    def export_model_spec(self) -> RationalNetworkSpec:
        return self.model_spec(export=True)

    def structure_info(self) -> dict[str, Any]:
        spec = self.compile_spec()
        result = {
            "type": self.__class__.__name__,
            "name": self.name,
            "input_dim": self.input_dim,
            "n_basis": self.n_basis,
            "depth": self.depth,
            "widths": self.widths,
            "hidden_widths": self.hidden_widths,
            "n_outputs": self.n_outputs,
            "n_classes": self.n_classes,
            "n_species": spec.n_species,
            "n_evidence_species": spec.n_evidence_species,
            "n_response_species": spec.n_response_species,
            "n_output_species": spec.n_output_species,
            "n_reactions": spec.n_reactions,
            "n_rational_reactions": spec.n_rational_reactions,
            "n_output_reactions": spec.n_output_reactions,
            "basis": self.basis.structure_info(),
            "network": self.network.structure_info(),
        }
        if self.has_input_gate or self.has_feature_gate:
            result["gates"] = {
                "input": self._gate_info(self.input_gate),
                "feature": self._gate_info(self.feature_gate),
            }
        return result

    def _project_export_structure(
        self,
        basis: tuple[Any, ...],
        layers: tuple[RationalLayerSpec, ...],
    ) -> tuple[
        int,
        tuple[Any, ...],
        tuple[RationalLayerSpec, ...],
        tuple[int, ...] | None,
        tuple[int, ...] | None,
        Mapping[str, Any] | None,
    ]:
        active_input_candidates = set(self.active_input_indices_for_export())
        active_basis_candidates = set(self.active_basis_indices_for_export())
        live_basis = self._live_first_layer_basis(layers[0].evidence)

        kept_basis_indices = [
            index
            for index, feature in enumerate(basis)
            if index in active_basis_candidates
            and index in live_basis
            and self._basis_dependencies(feature).issubset(active_input_candidates)
        ]
        if not kept_basis_indices:
            kept_basis_indices = [0]

        used_inputs: set[int] = set()
        for index in kept_basis_indices:
            used_inputs.update(self._basis_dependencies(basis[index]))
        active_inputs = tuple(sorted(used_inputs or active_input_candidates or {0}))

        old_input_to_new = {
            old_index: new_index
            for new_index, old_index in enumerate(active_inputs)
        }
        remapped_basis = tuple(
            self._remap_basis_feature(
                basis[old_index],
                old_input_to_new=old_input_to_new,
            )
            for old_index in kept_basis_indices
        )

        changed = (
            len(active_inputs) != self.input_dim
            or len(remapped_basis) != len(basis)
        )
        if not changed:
            return self.input_dim, basis, layers, None, None, None

        old_basis_to_new = {
            old_index: new_index
            for new_index, old_index in enumerate(kept_basis_indices)
        }
        first = layers[0]
        first_evidence = self._remap_evidence(
            first.evidence,
            old_basis_to_new=old_basis_to_new,
            new_n_basis=len(remapped_basis),
        )
        remapped_layers = (
            replace(
                first,
                input_dim=len(remapped_basis),
                evidence=first_evidence,
            ),
        ) + layers[1:]

        metadata = {
            "enabled": True,
            "original_input_dim": self.input_dim,
            "export_input_dim": len(active_inputs),
            "original_n_basis": len(basis),
            "export_n_basis": len(remapped_basis),
            "active_input_indices": active_inputs,
            "active_basis_indices": tuple(kept_basis_indices),
        }
        return (
            len(active_inputs),
            remapped_basis,
            remapped_layers,
            active_inputs,
            tuple(kept_basis_indices),
            metadata,
        )

    def _validate_gates(self) -> None:
        if self.input_gate is not None:
            if not isinstance(self.input_gate, InputGate):
                raise TypeError("input_gate must be an InputGate or None.")
            if self.input_gate.n_inputs != self.input_dim:
                raise ValueError("input_gate.n_inputs must equal input_dim.")
        if self.feature_gate is not None:
            if not isinstance(self.feature_gate, FeatureGate):
                raise TypeError("feature_gate must be a FeatureGate or None.")
            if self.feature_gate.n_features != self.n_basis:
                raise ValueError("feature_gate.n_features must equal n_basis.")

    def _apply_input_gate(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.input_gate is None else self.input_gate(x)

    def _apply_feature_gate(self, phi: torch.Tensor) -> torch.Tensor:
        return phi if self.feature_gate is None else self.feature_gate(phi)

    def _zero_scalar(self) -> torch.Tensor:
        for parameter in self.parameters():
            return parameter.sum() * 0.0
        return torch.zeros(())

    @staticmethod
    def _active_indices(gate: nn.Module | None, size: int) -> tuple[int, ...]:
        if gate is None or not hasattr(gate, "active_mask"):
            return tuple(range(size))
        mask = gate.active_mask(0.0).detach().cpu().reshape(-1).bool()
        if mask.numel() != size:
            raise ValueError("Gate active mask has an unexpected size.")
        indices = tuple(
            index for index, is_active in enumerate(mask.tolist()) if is_active
        )
        return indices or (0,)

    @staticmethod
    def _live_first_layer_basis(evidence: EvidenceLayerSpec) -> set[int]:
        return {
            int(edge.source_input_index)
            for edge in evidence.edges
        }

    @staticmethod
    def _basis_dependencies(feature: Any) -> set[int]:
        dependencies: set[int] = set()
        for term in getattr(feature, "production_terms", ()):
            for factor in getattr(term, "factors", ()):
                if getattr(factor, "role", None) == "input":
                    dependencies.add(int(factor.index))
        return dependencies

    @staticmethod
    def _remap_basis_feature(
        feature: Any,
        *,
        old_input_to_new: Mapping[int, int],
    ) -> Any:
        if not isinstance(feature, BasisFeatureSpec):
            return feature
        terms: list[ProductionTerm] = []
        for term in feature.production_terms:
            factors: list[SpeciesFactor] = []
            for factor in term.factors:
                if factor.role == "input":
                    factors.append(
                        replace(
                            factor,
                            index=int(old_input_to_new[int(factor.index)]),
                        )
                    )
                else:
                    factors.append(factor)
            terms.append(replace(term, factors=tuple(factors)))

        metadata = None if feature.metadata is None else dict(feature.metadata)
        if metadata is not None and "input_index" in metadata:
            old_index = int(metadata["input_index"])
            new_index = int(old_input_to_new[old_index])
            metadata["input_index"] = new_index
            if metadata.get("basis_family") == "simplex":
                metadata["input_species"] = f"X_{new_index}"
                if metadata.get("simplex_kind") == "complement":
                    metadata["reservoir_species"] = (
                        f"R_simplex_{new_index}"
                    )
        return replace(feature, production_terms=tuple(terms), metadata=metadata)

    @staticmethod
    def _remap_evidence(
        evidence: EvidenceLayerSpec,
        *,
        old_basis_to_new: Mapping[int, int],
        new_n_basis: int,
    ) -> EvidenceLayerSpec:
        edges: list[EvidenceEdgeSpec] = []
        for edge in evidence.edges:
            new_index = old_basis_to_new.get(int(edge.source_input_index))
            if new_index is not None:
                edges.append(
                    replace(edge, source_input_index=int(new_index))
                )
        return replace(
            evidence,
            input_dim=new_n_basis,
            edges=tuple(edges),
        )

    @staticmethod
    def _gate_info(gate: InputGate | FeatureGate | None) -> dict[str, Any]:
        if gate is None:
            return {"enabled": False}
        values = gate.gate().detach().cpu()
        return {
            "enabled": True,
            "size": int(values.numel()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "mean": float(values.mean().item()),
        }

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, n_basis={self.n_basis}, "
            f"depth={self.depth}, widths={self.widths}, name={self.name!r}"
        )


__all__ = ["BasisRationalModel", "BasisRationalState"]
