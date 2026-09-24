from __future__ import annotations

from typing import Any, Sequence

import torch

from .base import BaseBasisLayer
from .specs import (
    BasisFeatureSpec,
    ProductionTerm,
    RateRef,
    SpeciesFactor,
    feature,
    ma_term,
    rate,
    x,
    x_pow,
)
from utils.validation import (
    check_positive_int
)

class PolynomialBasisLayer(BaseBasisLayer):
    def __init__(
        self,
        n_inputs: int,
        *,
        include_constant: bool = True,
        include_identity: bool = True,
        powers: Sequence[int] = (2,),
        include_pairwise: bool = True,
        specs: Sequence[BasisFeatureSpec] | None = None,
        check_nonnegative: bool = True,
    ) -> None:
        super().__init__()

        check_positive_int("n_inputs", n_inputs)

        self.n_inputs = n_inputs
        self.check_nonnegative = check_nonnegative

        if specs is None:
            self._features = _build_polynomial_feature_specs(
                n_inputs=n_inputs,
                include_constant=include_constant,
                include_identity=include_identity,
                powers=powers,
                include_pairwise=include_pairwise,
            )
        else:
            self._features = list(specs)

        self.validate_metadata()
        _check_feature_input_range(self._features, n_inputs=n_inputs)

    def feature_specs(self) -> list[BasisFeatureSpec]:
        return list(self._features)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        if x_in.ndim != 2:
            raise ValueError(
                "PolynomialBasisLayer expects x_in with shape "
                f"(batch, n_inputs). Got shape {tuple(x_in.shape)}."
            )

        if x_in.shape[1] != self.n_inputs:
            raise ValueError(
                f"Input dimension mismatch: expected {self.n_inputs}, "
                f"got {x_in.shape[1]}."
            )

        if self.check_nonnegative and torch.any(x_in < 0).item():
            raise ValueError(
                "PolynomialBasisLayer received negative input values. "
                "CRN concentrations should be non-negative. "
                "Set check_nonnegative=False if this is intentional."
            )

        cols = [_eval_feature_spec(x_in, spec) for spec in self._features]
        return torch.stack(cols, dim=1)
    
    def structure_info(self) -> dict[str, Any]:
        info = super().structure_info()
        info.update(
            {
                "n_inputs": self.n_inputs,
                "check_nonnegative": self.check_nonnegative,
                "basis_family": "polynomial",
            }
        )
        return info

    def extra_repr(self) -> str:
        return (
            f"n_inputs={self.n_inputs}, "
            f"n_basis={self.n_basis}, "
            f"check_nonnegative={self.check_nonnegative}"
        )



def _check_feature_input_range(
    specs: Sequence[BasisFeatureSpec],
    *,
    n_inputs: int,
) -> None:
    for spec in specs:
        for prod in spec.production_terms:
            for factor in prod.factors:
                if factor.role == "input" and factor.index >= n_inputs:
                    raise ValueError(
                        f"Feature {spec.name!r} uses input index {factor.index}, "
                        f"but n_inputs={n_inputs}."
                    )
                
def _build_polynomial_feature_specs(
    *,
    n_inputs: int,
    include_constant: bool,
    include_identity: bool,
    powers: Sequence[int],
    include_pairwise: bool,
) -> list[BasisFeatureSpec]:
    specs: list[BasisFeatureSpec] = []

    def add_feature(
        *,
        name: str,
        production_terms: Sequence[ProductionTerm],
        group: str = "polynomial",
    ) -> None:
        idx = len(specs)
        specs.append(
            feature(
                name=name,
                output_species=f"Phi_{idx}",
                production_terms=production_terms,
                decay=rate(
                    f"phi_{idx}_decay",
                    value=1.0,
                    trainable=False,
                    shared=False,
                ),
                group=group,
            )
        )

    # 添加常数特征
    if include_constant:
        add_feature(
            name="1",
            production_terms=(
                ma_term(
                    rate_ref=rate(
                        "const_1_prod",
                        value=1.0,
                        trainable=False,
                        shared=True,
                    )
                ),
            ),
        )

    # 添加一次项
    if include_identity:
        for i in range(n_inputs):
            add_feature(
                name=f"x{i}",
                production_terms=(
                    ma_term(
                        x(i),
                        rate_ref=rate(
                            f"x{i}_prod",
                            value=1.0,
                            trainable=False,
                            shared=False,
                        ),
                    ),
                ),
            )

    clean_powers = _normalize_powers(powers)
    
    # 添加高次项
    for power in clean_powers:
        for i in range(n_inputs):
            add_feature(
                name=f"x{i}^{power}",
                production_terms=(
                    ma_term(
                        x_pow(i, power),
                        rate_ref=rate(
                            f"x{i}_pow{power}_prod",
                            value=1.0,
                            trainable=False,
                            shared=False,
                        ),
                    ),
                ),
            )

    # 添加交叉项
    if include_pairwise:
        for i in range(n_inputs):
            for j in range(i + 1, n_inputs):
                add_feature(
                    name=f"x{i}*x{j}",
                    production_terms=(
                        ma_term(
                            x(i),
                            x(j),
                            rate_ref=rate(
                                f"x{i}_x{j}_prod",
                                value=1.0,
                                trainable=False,
                                shared=False,
                            ),
                        ),
                    ),
                )

    return specs


def _normalize_powers(powers: Sequence[int]) -> tuple[int, ...]:
    out: list[int] = []

    for power in powers:
        check_positive_int("power", power)

        if power < 2:
            raise ValueError(f"power must be at least 2. Got {power}.")

        if power not in out:
            out.append(power)

    return tuple(out)

# 计算产生的特征值
def _eval_feature_spec(
    x_in: torch.Tensor,
    spec: BasisFeatureSpec,
) -> torch.Tensor:
    decay = _rate_value_tensor(spec.decay, x_in) # 基函数物质降解

    if torch.any(decay <= 0).item():
        raise ValueError(
            f"Decay rate must be positive for feature {spec.name!r}."
        )

    total = torch.zeros(
        x_in.shape[0],
        dtype=x_in.dtype,
        device=x_in.device,
    )

    for prod in spec.production_terms: # 基函数物质生成
        total = total + _eval_production_term(x_in, prod)

    return total / decay


def _eval_production_term(
    x_in: torch.Tensor,
    prod: ProductionTerm,
) -> torch.Tensor:
    value = _rate_value_tensor(prod.rate, x_in)

    out = torch.ones(
        x_in.shape[0],
        dtype=x_in.dtype,
        device=x_in.device,
    )

    for factor in prod.factors:
        out = out * _eval_factor(x_in, factor)

    return value * out


def _eval_factor(
    x_in: torch.Tensor,
    factor: SpeciesFactor,
) -> torch.Tensor:
    if factor.role != "input":
        raise NotImplementedError(
            "PolynomialBasisLayer.forward only supports input factors. "
            f"Got factor role={factor.role!r}. "
            "Custom layers using basis or buffer factors should override forward()."
        )

    value = x_in[:, factor.index]

    if factor.stoich == 1:
        return value

    return value.pow(factor.stoich)


def _rate_value_tensor(
    rate_ref: RateRef,
    x_in: torch.Tensor,
) -> torch.Tensor:
    if rate_ref.value is None:
        raise ValueError(
            f"RateRef {rate_ref.key!r} has no numeric value. "
            "PolynomialBasisLayer.forward requires fixed numeric rates. "
            "Custom trainable basis layers should override forward()."
        )

    return torch.as_tensor(
        rate_ref.value,
        dtype=x_in.dtype,
        device=x_in.device,
    )