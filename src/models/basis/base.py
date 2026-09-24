from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from .specs import BasisFeatureSpec

class BaseBasisLayer(nn.Module, ABC):
    @abstractmethod
    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def feature_specs(self) -> list[BasisFeatureSpec]:
        raise NotImplementedError
    
    @property
    def n_basis(self) -> int:
        return len(self.feature_specs())

    @property
    def names(self) -> list[str]:
        return [spec.name for spec in self.feature_specs()]

    @property
    def output_species(self) -> list[str]:
        return [spec.output_species for spec in self.feature_specs()]

    @property
    def dependencies(self) -> list[tuple[int, ...]]:
        return [spec.dependencies for spec in self.feature_specs()]

    def compile_spec(self) -> list[BasisFeatureSpec]: # 返回元信息用于编译
        return self.feature_specs()
    
    def structure_info(self) -> dict[str, Any]:
        specs = self.feature_specs()

        return {
            "type": self.__class__.__name__,
            "n_basis": len(specs),
            "names": [spec.name for spec in specs],
            "output_species": [spec.output_species for spec in specs],
            "dependencies": [spec.dependencies for spec in specs],
            "max_order": max((spec.max_order for spec in specs), default=0),
            "features": [spec.to_dict() for spec in specs],
            "has_trainable_parameters": any(
                param.requires_grad for param in self.parameters()
            ),
        }
    
    def validate_metadata(self) -> None:
        specs = self.feature_specs()

        if len(specs) == 0:
            raise ValueError("A basis layer must contain at least one feature.")

        for spec in specs:
            if not isinstance(spec, BasisFeatureSpec):
                raise TypeError(
                    "feature_specs() must return BasisFeatureSpec objects. "
                    f"Got {type(spec).__name__}."
                )

        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError("Basis feature names must be unique.")

        species = [spec.output_species for spec in specs]
        if len(set(species)) != len(species):
            raise ValueError("Basis output species names must be unique.")
