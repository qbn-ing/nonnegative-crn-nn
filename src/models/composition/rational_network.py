from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from models.evidence.dense import DenseRationalLayer, DenseRationalState
from models.evidence.rational_specs import RationalLayerSpec
from models.readout.competitive import (
    CompetitiveOutputLayer,
    CompetitiveOutputSpec,
    CompetitiveOutputState,
)
from utils.validation import (
    check_non_empty_str,
    check_nonnegative_float,
    check_positive_int,
)


@dataclass(frozen=True)
class RationalNetworkState:
    """Runtime state of D rational layers followed by one O layer."""

    input_h: torch.Tensor
    layer_states: tuple[DenseRationalState, ...]
    output: CompetitiveOutputState

    def __post_init__(self) -> None:
        if not isinstance(self.input_h, torch.Tensor) or self.input_h.ndim != 2:
            raise TypeError("input_h must be a 2D torch.Tensor.")
        if not isinstance(self.layer_states, tuple) or not self.layer_states:
            raise ValueError("layer_states must be a non-empty tuple.")
        if not all(isinstance(state, DenseRationalState) for state in self.layer_states):
            raise TypeError("layer_states must contain DenseRationalState objects.")
        if not isinstance(self.output, CompetitiveOutputState):
            raise TypeError("output must be a CompetitiveOutputState.")

        batch_size = int(self.input_h.shape[0])
        current_width = int(self.input_h.shape[1])
        for index, state in enumerate(self.layer_states):
            if state.batch_size != batch_size:
                raise ValueError(f"Layer {index} batch size is inconsistent.")
            if index > 0 and self.layer_states[index - 1].width != current_width:
                raise ValueError(f"Layer {index} input width is inconsistent.")
            current_width = state.width
        if self.output.r is not self.layer_states[-1].r:
            if self.output.r.shape != self.layer_states[-1].r.shape:
                raise ValueError("The O-layer input must match the final rational response.")

    @property
    def depth(self) -> int:
        return len(self.layer_states)

    @property
    def widths(self) -> tuple[int, ...]:
        return tuple(state.width for state in self.layer_states)

    @property
    def hidden_states(self) -> tuple[DenseRationalState, ...]:
        return self.layer_states[:-1]

    @property
    def final_rational_state(self) -> DenseRationalState:
        return self.layer_states[-1]

    @property
    def Z(self) -> torch.Tensor:
        return self.output.Z

    @property
    def Z_free(self) -> torch.Tensor:
        return self.output.Z_free

    @property
    def scores(self) -> torch.Tensor:
        return self.output.scores

    @property
    def predicted_classes(self) -> torch.Tensor:
        return self.output.predicted_classes

    @property
    def n_classes(self) -> int:
        return self.output.n_classes

    @property
    def n_outputs(self) -> int:
        return self.n_classes

    def normalized_scores(self, eps: float = 1e-12) -> torch.Tensor:
        return self.output.normalized_scores(eps=eps)

    def detach(self) -> RationalNetworkState:
        return RationalNetworkState(
            input_h=self.input_h.detach(),
            layer_states=tuple(state.detach() for state in self.layer_states),
            output=self.output.detach(),
        )

    def to_detached_dict(self) -> dict[str, Any]:
        return {
            "input_h": self.input_h.detach().cpu(),
            "layer_states": [
                {
                    "E": state.E.detach().cpu(),
                    "I": state.I.detach().cpu(),
                    "r": state.r.detach().cpu(),
                }
                for state in self.layer_states
            ],
            "output": self.output.to_detached_dict(),
            "depth": self.depth,
            "widths": self.widths,
        }


@dataclass(frozen=True)
class RationalNetworkSpec:
    """Export specification for rational layers followed by one O layer."""

    input_dim: int
    rational_layers: tuple[RationalLayerSpec, ...]
    output: CompetitiveOutputSpec
    basis: tuple[Any, ...] = ()
    name: str = "rational_network"
    active_input_indices: tuple[int, ...] | None = None
    active_basis_indices: tuple[int, ...] | None = None
    structural_pruning: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        check_positive_int("input_dim", self.input_dim)
        check_non_empty_str("name", self.name)
        if not isinstance(self.rational_layers, tuple) or not self.rational_layers:
            raise ValueError("rational_layers must be a non-empty tuple.")
        if not all(
            isinstance(layer, RationalLayerSpec)
            for layer in self.rational_layers
        ):
            raise TypeError("rational_layers must contain RationalLayerSpec objects.")
        if not isinstance(self.output, CompetitiveOutputSpec):
            raise TypeError("output must be a CompetitiveOutputSpec.")
        if not isinstance(self.basis, tuple):
            raise TypeError("basis must be a tuple.")

        first_input_dim = len(self.basis) if self.basis else self.input_dim
        if self.rational_layers[0].input_dim != first_input_dim:
            raise ValueError("The first rational layer input dimension is inconsistent.")
        for previous, current in zip(
            self.rational_layers,
            self.rational_layers[1:],
        ):
            if previous.width != current.input_dim:
                raise ValueError("Rational layer dimensions are inconsistent.")
        if self.rational_layers[-1].width != self.output.n_classes:
            raise ValueError(
                "The final rational layer width must equal output.n_classes."
            )

    @property
    def depth(self) -> int:
        return len(self.rational_layers)

    @property
    def widths(self) -> tuple[int, ...]:
        return tuple(layer.width for layer in self.rational_layers)

    @property
    def hidden_widths(self) -> tuple[int, ...]:
        return self.widths[:-1]

    @property
    def n_classes(self) -> int:
        return self.output.n_classes

    @property
    def n_basis_species(self) -> int:
        return len(self.basis)

    @property
    def n_evidence_species(self) -> int:
        return sum(layer.evidence.n_species for layer in self.rational_layers)

    @property
    def n_constant_species(self) -> int:
        return sum(
            layer.n_constant_species for layer in self.rational_layers
        )

    @property
    def n_response_species(self) -> int:
        return sum(layer.response.n_species for layer in self.rational_layers)

    @property
    def n_output_species(self) -> int:
        return self.output.n_species

    @property
    def n_species(self) -> int:
        return (
            self.n_basis_species
            + self.n_constant_species
            + self.n_evidence_species
            + self.n_response_species
            + self.n_output_species
        )

    @property
    def n_basis_reactions(self) -> int:
        total = 0
        for feature in self.basis:
            metadata = getattr(feature, "metadata", None)
            if isinstance(metadata, Mapping) and metadata.get("compile_mode") == "input_alias":
                continue
            total += int(getattr(feature, "n_production_terms", 0)) + 1
        return total

    @property
    def n_rational_reactions(self) -> int:
        return sum(layer.n_reactions for layer in self.rational_layers)

    @property
    def n_output_reactions(self) -> int:
        return self.output.n_reactions

    @property
    def n_reactions(self) -> int:
        return (
            self.n_basis_reactions
            + self.n_rational_reactions
            + self.n_output_reactions
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "input_dim": self.input_dim,
            "depth": self.depth,
            "widths": self.widths,
            "basis": [
                feature.to_dict() if hasattr(feature, "to_dict") else feature
                for feature in self.basis
            ],
            "rational_layers": [
                layer.to_dict() for layer in self.rational_layers
            ],
            "output": self.output.to_dict(),
            "active_input_indices": self.active_input_indices,
            "active_basis_indices": self.active_basis_indices,
            "structural_pruning": (
                None
                if self.structural_pruning is None
                else dict(self.structural_pruning)
            ),
            "n_species": self.n_species,
            "n_constant_species": self.n_constant_species,
            "n_reactions": self.n_reactions,
        }


class RationalNetwork(nn.Module):
    """Compose D dense rational layers with exactly one terminal O layer."""

    def __init__(
        self,
        layers: Iterable[DenseRationalLayer],
        output: CompetitiveOutputLayer,
        *,
        name: str = "rational_network",
        check_finite_input: bool = True,
        check_nonnegative_input: bool = True,
    ) -> None:
        super().__init__()
        check_non_empty_str("name", name)
        if not isinstance(check_finite_input, bool):
            raise TypeError("check_finite_input must be a bool.")
        if not isinstance(check_nonnegative_input, bool):
            raise TypeError("check_nonnegative_input must be a bool.")

        layer_tuple = tuple(layers)
        if not layer_tuple:
            raise ValueError("RationalNetwork requires at least one layer.")
        if not all(isinstance(layer, DenseRationalLayer) for layer in layer_tuple):
            raise TypeError("layers must contain DenseRationalLayer objects.")
        if not isinstance(output, CompetitiveOutputLayer):
            raise TypeError("output must be a CompetitiveOutputLayer.")
        for previous, current in zip(layer_tuple, layer_tuple[1:]):
            if previous.out_features != current.in_features:
                raise ValueError("Rational layer dimensions are inconsistent.")
        if layer_tuple[-1].out_features != output.n_classes:
            raise ValueError(
                "The final rational layer width must equal output.n_classes."
            )

        self.layers = nn.ModuleList(layer_tuple)
        self.output_layer = output
        self.name = name
        self.check_finite_input = check_finite_input
        self.check_nonnegative_input = check_nonnegative_input
        self.initialization_audit: Any | None = None

    @property
    def input_dim(self) -> int:
        return self.layers[0].in_features

    @property
    def depth(self) -> int:
        return len(self.layers)

    @property
    def widths(self) -> tuple[int, ...]:
        return tuple(layer.out_features for layer in self.layers)

    @property
    def hidden_widths(self) -> tuple[int, ...]:
        return self.widths[:-1]

    @property
    def n_classes(self) -> int:
        return self.output_layer.n_classes

    @property
    def n_outputs(self) -> int:
        return self.n_classes

    def forward(self, h: torch.Tensor) -> RationalNetworkState:
        self.validate_input(h)
        current = h
        states: list[DenseRationalState] = []
        for layer in self.layers:
            state = layer(current)
            states.append(state)
            current = state.r
        output = self.output_layer(current)
        return RationalNetworkState(
            input_h=h,
            layer_states=tuple(states),
            output=output,
        )

    def validate_input(self, h: torch.Tensor) -> None:
        if not isinstance(h, torch.Tensor):
            raise TypeError("h must be a torch.Tensor.")
        if h.ndim != 2 or h.shape[0] <= 0 or h.shape[1] != self.input_dim:
            raise ValueError(
                f"h must have shape (batch_size, {self.input_dim}). "
                f"Got {tuple(h.shape)}."
            )
        if self.check_finite_input and not torch.all(torch.isfinite(h)).item():
            raise ValueError("h must contain only finite values.")
        if self.check_nonnegative_input and torch.any(h < 0).item():
            raise ValueError("h must be nonnegative.")

    def iter_rational_layers(self) -> tuple[DenseRationalLayer, ...]:
        return tuple(self.layers)

    def rate_l1_loss(
        self,
        *,
        layer_weights: Sequence[float] | None = None,
        reduction: str = "sum",
    ) -> torch.Tensor:
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'.")
        if layer_weights is None:
            weights = (1.0,) * self.depth
        else:
            weights = tuple(float(value) for value in layer_weights)
            if len(weights) != self.depth:
                raise ValueError("layer_weights length must equal network depth.")
            for index, value in enumerate(weights):
                check_nonnegative_float(f"layer_weights[{index}]", value)

        values = [
            weight * layer.rate_l1_loss(reduction=reduction)
            for weight, layer in zip(weights, self.layers)
        ]
        return torch.stack(values).sum()

    def network_spec(self, *, export: bool = False) -> RationalNetworkSpec:
        layer_specs = tuple(
            layer.export_rational_spec() if export else layer.compile_spec()
            for layer in self.layers
        )
        output_spec = (
            self.output_layer.export_output_spec()
            if export
            else self.output_layer.compile_spec()
        )
        return RationalNetworkSpec(
            input_dim=self.input_dim,
            rational_layers=layer_specs,
            output=output_spec,
            name=self.name,
        )

    def compile_spec(self) -> RationalNetworkSpec:
        return self.network_spec(export=False)

    def export_model_spec(self) -> RationalNetworkSpec:
        return self.network_spec(export=True)

    def structure_info(self) -> dict[str, Any]:
        spec = self.compile_spec()
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "input_dim": self.input_dim,
            "depth": self.depth,
            "widths": self.widths,
            "hidden_widths": self.hidden_widths,
            "n_outputs": self.n_outputs,
            "n_classes": self.n_classes,
            "n_species": spec.n_species,
            "n_constant_species": spec.n_constant_species,
            "n_evidence_species": spec.n_evidence_species,
            "n_response_species": spec.n_response_species,
            "n_output_species": spec.n_output_species,
            "n_reactions": spec.n_reactions,
            "n_rational_reactions": spec.n_rational_reactions,
            "n_output_reactions": spec.n_output_reactions,
            "initialization_audit": (
                None
                if self.initialization_audit is None
                else self.initialization_audit.to_dict()
            ),
            "layers": [layer.structure_info() for layer in self.layers],
            "output": self.output_layer.structure_info(),
        }

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, depth={self.depth}, "
            f"widths={self.widths}, name={self.name!r}"
        )


__all__ = [
    "RationalNetwork",
    "RationalNetworkSpec",
    "RationalNetworkState",
]
