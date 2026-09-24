from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from models.basis.specs import RateRef, rate
from utils.validation import (
    check_non_empty_str,
    check_positive_float,
    check_positive_int,
)


@dataclass(frozen=True)
class CompetitiveOutputState:
    """Runtime state of the conserved competitive output pool."""

    r: torch.Tensor
    Z: torch.Tensor
    Z_free: torch.Tensor
    total: float

    def __post_init__(self) -> None:
        for name, value in (("r", self.r), ("Z", self.Z), ("Z_free", self.Z_free)):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if not torch.all(torch.isfinite(value)).item():
                raise ValueError(f"{name} must contain only finite values.")
            if torch.any(value < 0).item():
                raise ValueError(f"{name} must be nonnegative.")

        if self.r.ndim != 2 or self.r.shape != self.Z.shape:
            raise ValueError("r and Z must have the same 2D shape.")
        if self.r.shape[0] <= 0 or self.r.shape[1] <= 0:
            raise ValueError("r and Z must have positive batch and class dimensions.")
        if self.Z_free.shape != (self.r.shape[0], 1):
            raise ValueError("Z_free must have shape (batch_size, 1).")
        check_positive_float("total", self.total)

    @property
    def batch_size(self) -> int:
        return int(self.Z.shape[0])

    @property
    def n_classes(self) -> int:
        return int(self.Z.shape[1])

    @property
    def scores(self) -> torch.Tensor:
        return self.Z

    @property
    def predicted_classes(self) -> torch.Tensor:
        return torch.argmax(self.Z, dim=1)

    @property
    def active_output_fraction(self) -> torch.Tensor:
        """Fraction of the conserved output pool occupying active classes."""

        return self.Z.sum(dim=1, keepdim=True) / float(self.total)

    def normalized_scores(self, eps: float = 1e-12) -> torch.Tensor:
        """Normalize active-class concentrations while excluding ``Z_free``."""

        check_positive_float("eps", eps)
        return self.Z / self.Z.sum(dim=1, keepdim=True).clamp_min(eps)

    def detach(self) -> CompetitiveOutputState:
        return CompetitiveOutputState(
            r=self.r.detach(),
            Z=self.Z.detach(),
            Z_free=self.Z_free.detach(),
            total=self.total,
        )

    def to_detached_dict(self) -> dict[str, Any]:
        return {
            "r": self.r.detach().cpu(),
            "Z": self.Z.detach().cpu(),
            "Z_free": self.Z_free.detach().cpu(),
            "active_output_fraction": (
                self.active_output_fraction.detach().cpu()
            ),
            "total": self.total,
            "batch_size": self.batch_size,
            "n_classes": self.n_classes,
        }


@dataclass(frozen=True)
class CompetitiveOutputSpec:
    """CRN specification of the terminal O layer."""

    n_classes: int
    total: float = 1.0
    name: str = "competitive_output"
    inactive_name: str = "Z_free"
    active_prefix: str = "Z"
    response_prefix: str = "R"
    activation_rate: RateRef = rate(
        "k_output_activation",
        value=1.0,
        trainable=False,
        shared=True,
    )
    decay_rate: RateRef = rate(
        "k_output_decay",
        value=1.0,
        trainable=False,
        shared=True,
    )

    def __post_init__(self) -> None:
        check_positive_int("n_classes", self.n_classes)
        check_positive_float("total", self.total)
        for field_name in ("name", "inactive_name", "active_prefix", "response_prefix"):
            check_non_empty_str(field_name, getattr(self, field_name))
        if not isinstance(self.activation_rate, RateRef):
            raise TypeError("activation_rate must be a RateRef.")
        if not isinstance(self.decay_rate, RateRef):
            raise TypeError("decay_rate must be a RateRef.")

    @property
    def n_species(self) -> int:
        return self.n_classes + 1

    @property
    def n_reactions(self) -> int:
        return 2 * self.n_classes

    def active_species_name(self, class_index: int) -> str:
        if class_index < 0 or class_index >= self.n_classes:
            raise IndexError("class_index is out of range.")
        return f"{self.active_prefix}_{class_index}"

    def response_species_name(self, class_index: int) -> str:
        if class_index < 0 or class_index >= self.n_classes:
            raise IndexError("class_index is out of range.")
        return f"{self.response_prefix}_{class_index}"

    def to_dict(self) -> dict[str, Any]:
        species = [
            {
                "name": self.inactive_name,
                "role": "inactive",
                "class_index": None,
            }
        ]
        reactions: list[dict[str, Any]] = []

        for class_index in range(self.n_classes):
            z_name = self.active_species_name(class_index)
            r_name = self.response_species_name(class_index)
            species.append(
                {
                    "name": z_name,
                    "role": "active",
                    "class_index": class_index,
                }
            )
            reactions.append(
                {
                    "kind": "activation",
                    "class_index": class_index,
                    "reactants": [self.inactive_name, r_name],
                    "products": [z_name, r_name],
                    "catalysts": [r_name],
                    "order": 2,
                    "rate": self.activation_rate.to_dict(),
                }
            )

        for class_index in range(self.n_classes):
            z_name = self.active_species_name(class_index)
            reactions.append(
                {
                    "kind": "decay",
                    "class_index": class_index,
                    "reactants": [z_name],
                    "products": [self.inactive_name],
                    "catalysts": [],
                    "order": 1,
                    "rate": self.decay_rate.to_dict(),
                }
            )

        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "n_classes": self.n_classes,
            "total": self.total,
            "n_species": self.n_species,
            "n_reactions": self.n_reactions,
            "species": species,
            "reactions": reactions,
        }


class CompetitiveOutputLayer(nn.Module):
    """Map class responses to the single terminal conserved output pool."""

    def __init__(
        self,
        *,
        n_classes: int,
        total: float = 1.0,
        name: str = "competitive_output",
        inactive_name: str = "Z_free",
        active_prefix: str = "Z",
        response_prefix: str = "R",
        check_nonnegative_r: bool = True,
    ) -> None:
        super().__init__()

        check_positive_int("n_classes", n_classes)
        check_positive_float("total", total)
        for field_name, value in (
            ("name", name),
            ("inactive_name", inactive_name),
            ("active_prefix", active_prefix),
            ("response_prefix", response_prefix),
        ):
            check_non_empty_str(field_name, value)
        if not isinstance(check_nonnegative_r, bool):
            raise TypeError("check_nonnegative_r must be a bool.")

        self.n_classes = n_classes
        self.total = float(total)
        self.name = name
        self.inactive_name = inactive_name
        self.active_prefix = active_prefix
        self.response_prefix = response_prefix
        self.check_nonnegative_r = check_nonnegative_r

    def forward(self, r: torch.Tensor) -> CompetitiveOutputState:
        self.validate_input(r)
        normalizer = 1.0 + r.sum(dim=1, keepdim=True)
        return CompetitiveOutputState(
            r=r,
            Z=self.total * r / normalizer,
            Z_free=self.total / normalizer,
            total=self.total,
        )

    def validate_input(self, r: torch.Tensor) -> None:
        if not isinstance(r, torch.Tensor):
            raise TypeError("r must be a torch.Tensor.")
        if r.ndim != 2 or r.shape[0] <= 0 or r.shape[1] != self.n_classes:
            raise ValueError(
                "r must have shape (batch_size, n_classes) with "
                f"n_classes={self.n_classes}. Got {tuple(r.shape)}."
            )
        if not torch.all(torch.isfinite(r)).item():
            raise ValueError("r must contain only finite values.")
        if self.check_nonnegative_r and torch.any(r < 0).item():
            raise ValueError("r must be nonnegative.")

    def output_spec(self) -> CompetitiveOutputSpec:
        return CompetitiveOutputSpec(
            n_classes=self.n_classes,
            total=self.total,
            name=self.name,
            inactive_name=self.inactive_name,
            active_prefix=self.active_prefix,
            response_prefix=self.response_prefix,
        )

    def compile_spec(self) -> CompetitiveOutputSpec:
        return self.output_spec()

    def export_output_spec(self) -> CompetitiveOutputSpec:
        return self.output_spec()

    def structure_info(self) -> dict[str, Any]:
        spec = self.output_spec()
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "n_classes": self.n_classes,
            "total": self.total,
            "n_species": spec.n_species,
            "n_reactions": spec.n_reactions,
        }

    def extra_repr(self) -> str:
        return (
            f"n_classes={self.n_classes}, total={self.total}, "
            f"name={self.name!r}"
        )


__all__ = [
    "CompetitiveOutputLayer",
    "CompetitiveOutputSpec",
    "CompetitiveOutputState",
]
