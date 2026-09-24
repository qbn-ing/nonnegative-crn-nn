from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .specs import (
    EvidenceBiasSpec,
    EvidenceEdgeSpec,
    EvidenceKind,
    EvidenceLayerSpec,
    EvidenceSpeciesSpec,
)


@dataclass(frozen=True)
class EvidenceState:
    """Output container of an evidence layer.

    Attributes:
        E:
            Positive evidence tensor.

            Shape:
                (batch_size, n_classes)

        I:
            Negative / inhibitory evidence tensor.

            Shape:
                (batch_size, n_classes)

    Notes:
        This class only checks tensor type and shape consistency.
        It does not check non-negativity by default, because that may add
        unnecessary overhead in training loops. Non-negativity should mainly
        be guaranteed by layer construction, e.g. non-negative rates and
        non-negative inputs.
    """

    E: torch.Tensor
    I: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.E, torch.Tensor):
            raise TypeError(
                "EvidenceState.E must be a torch.Tensor. "
                f"Got {type(self.E).__name__}."
            )

        if not isinstance(self.I, torch.Tensor):
            raise TypeError(
                "EvidenceState.I must be a torch.Tensor. "
                f"Got {type(self.I).__name__}."
            )

        if self.E.ndim != 2:
            raise ValueError(
                "EvidenceState.E must be a 2D tensor with shape "
                "(batch_size, n_classes). "
                f"Got shape {tuple(self.E.shape)}."
            )

        if self.I.ndim != 2:
            raise ValueError(
                "EvidenceState.I must be a 2D tensor with shape "
                "(batch_size, n_classes). "
                f"Got shape {tuple(self.I.shape)}."
            )

        if self.E.shape != self.I.shape:
            raise ValueError(
                "EvidenceState.E and EvidenceState.I must have the same shape. "
                f"Got E.shape={tuple(self.E.shape)}, "
                f"I.shape={tuple(self.I.shape)}."
            )
        
    @property
    def batch_size(self) -> int:
        return int(self.E.shape[0])

    @property
    def n_classes(self) -> int:
        return int(self.E.shape[1])

    @property
    def shape(self) -> torch.Size:
        return self.E.shape

    @property
    def device(self) -> torch.device:
        if self.E.device != self.I.device:
            raise ValueError(
                "EvidenceState.E and EvidenceState.I are on different devices. "
                f"Got E.device={self.E.device}, I.device={self.I.device}."
            )

        return self.E.device

    @property
    def dtype(self) -> torch.dtype:
        if self.E.dtype != self.I.dtype:
            raise ValueError(
                "EvidenceState.E and EvidenceState.I have different dtypes. "
                f"Got E.dtype={self.E.dtype}, I.dtype={self.I.dtype}."
            )

        return self.E.dtype
    
    def to_tuple(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.E, self.I

    def detach(self) -> EvidenceState:
        return EvidenceState(
            E=self.E.detach(),
            I=self.I.detach(),
        )

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "E": self.E,
            "I": self.I,
        }
    

class BaseEvidenceLayer(nn.Module, ABC):
    """Abstract base class for all evidence layers.

    Expected numerical map:

        phi -> EvidenceState(E, I)

    where:

        phi.shape == (batch_size, input_dim)
        E.shape   == (batch_size, output_dim)
        I.shape   == (batch_size, output_dim)

    Chemical interpretation:

        Positive edge:
            H_j -> H_j + E_k

        Negative edge:
            H_j -> H_j + I_k

    This class is responsible for interface consistency. Concrete rate
    parameterization, gate logic, sparsity, low-rank structure, or feedback
    should be implemented by subclasses.
    """

    @abstractmethod
    def forward(self, phi: torch.Tensor) -> EvidenceState:
        """Compute positive and negative evidence from basis features."""
        raise NotImplementedError

    @abstractmethod
    def evidence_spec(self) -> EvidenceLayerSpec:
        """Return compile-time metadata of this evidence layer."""
        raise NotImplementedError
    
    def compile_spec(self) -> EvidenceLayerSpec:
        """Alias for evidence_spec.

        The name compile_spec emphasizes that the returned metadata is intended
        for later CRN / PySB / DSD export.
        """
        return self.evidence_spec()

    def evidence_specs(self) -> EvidenceLayerSpec:
        """Backward-compatible alias.

        Although the method name is plural, the returned object describes the
        whole evidence layer.
        """
        return self.evidence_spec()

    @property
    def input_dim(self) -> int:
        return self.evidence_spec().input_dim

    @property
    def output_dim(self) -> int:
        return self.evidence_spec().output_dim

    @property
    def n_basis(self) -> int:
        return self.input_dim

    @property
    def n_classes(self) -> int:
        return self.output_dim

    @property
    def n_species(self) -> int:
        return self.evidence_spec().n_species
    
    @property
    def n_edges(self) -> int:
        return self.evidence_spec().n_edges

    @property
    def n_positive_edges(self) -> int:
        return self.evidence_spec().n_positive_edges

    @property
    def n_negative_edges(self) -> int:
        return self.evidence_spec().n_negative_edges

    @property
    def species(self) -> tuple[EvidenceSpeciesSpec, ...]:
        return self.evidence_spec().species

    @property
    def edges(self) -> tuple[EvidenceEdgeSpec, ...]:
        return self.evidence_spec().edges

    @property
    def biases(self) -> tuple[EvidenceBiasSpec, ...]:
        return self.evidence_spec().biases

    @property
    def positive_species(self) -> tuple[EvidenceSpeciesSpec, ...]:
        return self.evidence_spec().positive_species

    @property
    def negative_species(self) -> tuple[EvidenceSpeciesSpec, ...]:
        return self.evidence_spec().negative_species

    @property
    def positive_edges(self) -> tuple[EvidenceEdgeSpec, ...]:
        return self.evidence_spec().positive_edges

    @property
    def negative_edges(self) -> tuple[EvidenceEdgeSpec, ...]:
        return self.evidence_spec().negative_edges
    
    def species_names(
        self,
        *,
        kind: EvidenceKind | None = None,
    ) -> tuple[str, ...]:
        """Return evidence species names.

        Args:
            kind:
                None, "positive", or "negative".
        """
        if kind is None:
            return tuple(item.name for item in self.species)

        if kind == "positive":
            return tuple(item.name for item in self.positive_species)

        if kind == "negative":
            return tuple(item.name for item in self.negative_species)

        raise ValueError(
            f"Unsupported evidence kind: {kind!r}. "
            "Expected 'positive', 'negative', or None."
        )
    
    def validate_input(self, phi: torch.Tensor) -> None:
        """Validate basis feature tensor before evidence computation."""
        if not isinstance(phi, torch.Tensor):
            raise TypeError(
                "Evidence layer input phi must be a torch.Tensor. "
                f"Got {type(phi).__name__}."
            )

        if phi.ndim != 2:
            raise ValueError(
                "Evidence layer input phi must be a 2D tensor with shape "
                "(batch_size, n_basis). "
                f"Got shape {tuple(phi.shape)}."
            )

        if phi.shape[1] != self.input_dim:
            raise ValueError(
                "Evidence layer input dimension does not match input_dim. "
                f"Got phi.shape[1]={phi.shape[1]}, expected {self.input_dim}."
            )

    def validate_state(self, state: EvidenceState) -> None:
        """Validate output state shape against this layer metadata."""
        if not isinstance(state, EvidenceState):
            raise TypeError(
                "Evidence layer output must be an EvidenceState. "
                f"Got {type(state).__name__}."
            )

        if state.n_classes != self.output_dim:
            raise ValueError(
                "EvidenceState output dimension does not match output_dim. "
                f"Got state.n_classes={state.n_classes}, "
                f"expected {self.output_dim}."
            )
        
    def structure_info(self) -> dict[str, Any]:
        """Return lightweight structural information for logging/debugging."""
        spec = self.evidence_spec()

        return {
            "type": self.__class__.__name__,
            "name": spec.name,
            "input_dim": spec.input_dim,
            "output_dim": spec.output_dim,
            "n_basis": spec.input_dim,
            "n_classes": spec.output_dim,
            "n_species": spec.n_species,
            "n_edges": spec.n_edges,
            "n_positive_edges": spec.n_positive_edges,
            "n_negative_edges": spec.n_negative_edges,
            "n_biases": spec.n_biases,
            "positive_species": self.species_names(kind="positive"),
            "negative_species": self.species_names(kind="negative"),
        }
