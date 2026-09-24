from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from models.builders import (
    DenseRationalConfig,
    build_rational_network,
)
from models.basis.polynomial import PolynomialBasisLayer
from models.basis.simplex import SimplexBasisLayer
from models.composition.basis_rational import BasisRationalModel
from models.initialization import (
    InitializationAudit,
    NonnegativeXavierUniformInitialization,
)

from .data import expected_basis_dim
from .protocol import RunSpec


@dataclass(frozen=True)
class ModelBuild:
    model: BasisRationalModel
    initialization_audit: InitializationAudit
    representation: str
    raw_input_dim: int
    basis_dim: int
    n_classes: int

    def to_dict(self) -> dict[str, Any]:
        layer = self.model.network.layers[0]
        return {
            "schema_version": 1,
            "raw_input_dim": self.raw_input_dim,
            "basis_dim": self.basis_dim,
            "n_classes": self.n_classes,
            "depth": self.model.depth,
            "widths": self.model.widths,
            "representation": self.representation,
            "include_constant_basis": False,
            "include_trainable_affine_bias": layer.include_bias,
            "weight_parameterization": layer.evidence.weight_parameterization,
            "initialization": self.initialization_audit.to_dict(),
        }


class VectorizedQuadraticBasisLayer(PolynomialBasisLayer):
    """Full degree-two basis with vectorized runtime evaluation."""

    def __init__(self, n_inputs: int) -> None:
        super().__init__(
            n_inputs=n_inputs,
            include_constant=False,
            include_identity=True,
            powers=(2,),
            include_pairwise=True,
            check_nonnegative=True,
        )
        indices = torch.triu_indices(n_inputs, n_inputs, offset=1)
        self.register_buffer("_pair_i", indices[0], persistent=True)
        self.register_buffer("_pair_j", indices[1], persistent=True)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        if x_in.ndim != 2 or x_in.shape[1] != self.n_inputs:
            raise ValueError(
                f"expected (batch, {self.n_inputs}), got {tuple(x_in.shape)}."
            )
        if self.check_nonnegative and torch.any(x_in < 0).item():
            raise ValueError("quadratic basis inputs must be nonnegative.")
        pairwise = x_in[:, self._pair_i] * x_in[:, self._pair_j]
        return torch.cat((x_in, x_in.square(), pairwise), dim=1)


def build_model(
    spec: RunSpec,
    *,
    n_inputs: int,
    n_classes: int,
) -> ModelBuild:
    """Build one terminal class-evidence E/I-to-r layer followed by O."""

    if spec.representation == "raw":
        basis = SimplexBasisLayer(
            n_inputs=n_inputs,
            include_complement=False,
            include_identity=True,
            check_bounds=False,
            species_prefix="X",
        )
    elif spec.representation == "quadratic":
        basis = VectorizedQuadraticBasisLayer(n_inputs=n_inputs)
    else:
        raise ValueError("unsupported representation.")
    rational_config = DenseRationalConfig(
        evidence_init_rate=0.0,
        evidence_eps=0.0,
        weight_parameterization="exact_nonnegative",
        eta_init=1.0,
        gamma_init=1.0,
        chi_init=1.0,
        response_eps=1e-8,
        denom_eps=1e-12,
        check_nonnegative_input=True,
        include_bias=True,
    )
    initialization = NonnegativeXavierUniformInitialization(
        gain=spec.xavier_gain
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec.model_seed + 23)
        network = build_rational_network(
            input_dim=basis.n_basis,
            widths=(n_classes,),
            config=rational_config,
            initialization=initialization,
            name=f"single_{spec.dataset}_network",
            layer_name_prefix="rational",
        )
        model = BasisRationalModel(
            basis=basis,
            network=network,
            name=(
                f"single_{spec.dataset}_{spec.representation}_d1_classifier"
            ),
        )
    audit = model.network.initialization_audit
    if not isinstance(audit, InitializationAudit):
        raise RuntimeError("network initialization audit is missing.")
    expected = expected_basis_dim(n_inputs, spec.representation)
    if model.depth != 1 or model.widths != (n_classes,):
        raise RuntimeError("single-layer topology is inconsistent.")
    if model.n_basis != expected:
        raise RuntimeError(
            f"basis dimension mismatch: expected {expected}, got {model.n_basis}."
        )
    layer = model.network.layers[0]
    if not layer.include_bias:
        raise RuntimeError("single-layer evidence must retain affine bias.")
    return ModelBuild(
        model=model,
        initialization_audit=audit,
        representation=spec.representation,
        raw_input_dim=n_inputs,
        basis_dim=model.n_basis,
        n_classes=n_classes,
    )


__all__ = [
    "ModelBuild",
    "VectorizedQuadraticBasisLayer",
    "build_model",
]
