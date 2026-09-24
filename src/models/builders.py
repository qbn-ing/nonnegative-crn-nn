from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from models.basis.base import BaseBasisLayer
from models.basis.polynomial import PolynomialBasisLayer
from models.basis.simplex import SimplexBasisLayer
from models.composition.basis_rational import BasisRationalModel
from models.composition.rational_network import RationalNetwork
from models.evidence.dense import (
    DenseRationalLayer,
    WeightParameterization,
)
from models.gates import FeatureGate, InputGate
from models.initialization import (
    NetworkInitialization,
    initialize_rational_network_,
)
from models.readout.competitive import CompetitiveOutputLayer
from utils.validation import (
    check_non_empty_str,
    check_nonnegative_float,
    check_positive_float,
    check_positive_int,
)


@dataclass(frozen=True)
class GateConfig:
    enabled: bool = False
    init: float | Sequence[float] | torch.Tensor = 1.0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        _check_bool("enabled", self.enabled)
        check_nonnegative_float("eps", self.eps)
        _check_rate_init("init", self.init, allow_zero=False)


@dataclass(frozen=True)
class PolynomialBasisConfig:
    n_inputs: int
    include_constant: bool = True
    include_identity: bool = True
    powers: Sequence[int] = (2,)
    include_pairwise: bool = True
    check_nonnegative: bool = True

    def __post_init__(self) -> None:
        check_positive_int("n_inputs", self.n_inputs)
        for name in (
            "include_constant",
            "include_identity",
            "include_pairwise",
            "check_nonnegative",
        ):
            _check_bool(name, getattr(self, name))
        powers = _normalize_positive_int_sequence(
            "powers",
            self.powers,
            allow_empty=True,
        )
        if any(power < 2 for power in powers):
            raise ValueError("Every polynomial power must be at least 2.")
        object.__setattr__(self, "powers", powers)


@dataclass(frozen=True)
class SimplexBasisConfig:
    n_inputs: int
    total: float = 1.0
    input_indices: Sequence[int] | None = None
    include_complement: bool = True
    include_identity: bool = True
    complement_first: bool = True
    check_bounds: bool = True
    complement_rho: float = 1.0
    complement_initial: float = 1e-6
    species_prefix: str = "Phi"

    def __post_init__(self) -> None:
        check_positive_int("n_inputs", self.n_inputs)
        check_positive_float("total", self.total)
        check_positive_float("complement_rho", self.complement_rho)
        check_nonnegative_float("complement_initial", self.complement_initial)
        for name in (
            "include_complement",
            "include_identity",
            "complement_first",
            "check_bounds",
        ):
            _check_bool(name, getattr(self, name))
        check_non_empty_str("species_prefix", self.species_prefix)
        if not self.include_complement and not self.include_identity:
            raise ValueError("At least one simplex feature family must be enabled.")

        if self.input_indices is not None:
            indices = tuple(self.input_indices)
            if not indices:
                raise ValueError("input_indices must not be empty.")
            if len(set(indices)) != len(indices):
                raise ValueError("input_indices must not contain duplicates.")
            for index, value in enumerate(indices):
                if not isinstance(value, int) or isinstance(value, bool):
                    raise TypeError(f"input_indices[{index}] must be an int.")
                if value < 0 or value >= self.n_inputs:
                    raise ValueError(f"input_indices[{index}] is out of range.")
            object.__setattr__(self, "input_indices", indices)


BasisConfig = PolynomialBasisConfig | SimplexBasisConfig


@dataclass(frozen=True)
class DenseRationalConfig:
    """Configuration of one complete dense E/I-to-r layer."""

    evidence_init_rate: float = 0.1
    evidence_eps: float = 1e-8
    weight_parameterization: WeightParameterization = "positive"
    eta_init: float | Sequence[float] | torch.Tensor = 1.0
    gamma_init: float | Sequence[float] | torch.Tensor = 1.0
    chi_init: float | Sequence[float] | torch.Tensor = 1.0
    response_eps: float = 1e-8
    denom_eps: float = 1e-12
    check_nonnegative_input: bool = True
    include_bias: bool | None = None
    bias_init_rate: float | None = None
    include_constant_channel: bool = False
    constant_value: float = 1.0

    def __post_init__(self) -> None:
        if self.weight_parameterization not in {"positive", "exact_nonnegative"}:
            raise ValueError(
                "weight_parameterization must be 'positive' or "
                "'exact_nonnegative'."
            )
        check_nonnegative_float("evidence_init_rate", self.evidence_init_rate)
        check_nonnegative_float("evidence_eps", self.evidence_eps)
        check_nonnegative_float("response_eps", self.response_eps)
        check_nonnegative_float("denom_eps", self.denom_eps)
        if (
            self.weight_parameterization == "positive"
            and self.evidence_init_rate <= self.evidence_eps
        ):
            raise ValueError(
                "Positive evidence parameters require "
                "evidence_init_rate > evidence_eps."
            )
        for name in ("eta_init", "gamma_init", "chi_init"):
            _check_rate_init(name, getattr(self, name), allow_zero=False)
        _check_bool("check_nonnegative_input", self.check_nonnegative_input)
        if self.include_bias is not None:
            _check_bool("include_bias", self.include_bias)
        _check_bool("include_constant_channel", self.include_constant_channel)
        resolved_bias = (
            self.include_constant_channel
            if self.include_bias is None
            else self.include_bias
        )
        if self.include_bias is not None and (
            self.include_constant_channel
            and not self.include_bias
        ):
            raise ValueError(
                "include_bias and include_constant_channel disagree."
            )
        object.__setattr__(self, "include_bias", resolved_bias)
        object.__setattr__(
            self,
            "include_constant_channel",
            resolved_bias,
        )
        if self.bias_init_rate is not None:
            check_nonnegative_float(
                "bias_init_rate",
                self.bias_init_rate,
            )
        check_positive_float("constant_value", self.constant_value)
        if self.constant_value != 1.0:
            raise ValueError(
                "constant_value is a compatibility field and must equal 1.0."
            )


def build_polynomial_basis(
    config: PolynomialBasisConfig,
) -> PolynomialBasisLayer:
    if not isinstance(config, PolynomialBasisConfig):
        raise TypeError("config must be a PolynomialBasisConfig.")
    return PolynomialBasisLayer(
        n_inputs=config.n_inputs,
        include_constant=config.include_constant,
        include_identity=config.include_identity,
        powers=config.powers,
        include_pairwise=config.include_pairwise,
        check_nonnegative=config.check_nonnegative,
    )


def build_simplex_basis(config: SimplexBasisConfig) -> SimplexBasisLayer:
    if not isinstance(config, SimplexBasisConfig):
        raise TypeError("config must be a SimplexBasisConfig.")
    return SimplexBasisLayer(
        n_inputs=config.n_inputs,
        total=config.total,
        input_indices=config.input_indices,
        include_complement=config.include_complement,
        include_identity=config.include_identity,
        complement_first=config.complement_first,
        check_bounds=config.check_bounds,
        complement_rho=config.complement_rho,
        complement_initial=config.complement_initial,
        species_prefix=config.species_prefix,
    )


def build_basis_layer(config: BasisConfig) -> BaseBasisLayer:
    if isinstance(config, PolynomialBasisConfig):
        return build_polynomial_basis(config)
    if isinstance(config, SimplexBasisConfig):
        return build_simplex_basis(config)
    raise TypeError(
        "config must be a PolynomialBasisConfig or SimplexBasisConfig."
    )


def build_input_gate(
    *,
    n_inputs: int,
    config: GateConfig | None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> InputGate | None:
    check_positive_int("n_inputs", n_inputs)
    if config is None or not config.enabled:
        return None
    if not isinstance(config, GateConfig):
        raise TypeError("config must be a GateConfig or None.")
    return InputGate(
        n_inputs=n_inputs,
        init=config.init,
        eps=config.eps,
        dtype=dtype,
        device=device,
    )


def build_feature_gate(
    *,
    n_features: int,
    config: GateConfig | None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> FeatureGate | None:
    check_positive_int("n_features", n_features)
    if config is None or not config.enabled:
        return None
    if not isinstance(config, GateConfig):
        raise TypeError("config must be a GateConfig or None.")
    return FeatureGate(
        n_features=n_features,
        init=config.init,
        eps=config.eps,
        dtype=dtype,
        device=device,
    )


def build_dense_rational_layer(
    *,
    input_dim: int,
    width: int,
    config: DenseRationalConfig | None = None,
    name: str = "rational",
    response_prefix: str = "R",
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> DenseRationalLayer:
    check_positive_int("input_dim", input_dim)
    check_positive_int("width", width)
    check_non_empty_str("name", name)
    check_non_empty_str("response_prefix", response_prefix)
    cfg = DenseRationalConfig() if config is None else config
    if not isinstance(cfg, DenseRationalConfig):
        raise TypeError("config must be a DenseRationalConfig or None.")
    return DenseRationalLayer(
        in_features=input_dim,
        out_features=width,
        init_rate=cfg.evidence_init_rate,
        bias_init_rate=cfg.bias_init_rate,
        weight_parameterization=cfg.weight_parameterization,
        eta_init=cfg.eta_init,
        gamma_init=cfg.gamma_init,
        chi_init=cfg.chi_init,
        evidence_eps=cfg.evidence_eps,
        response_eps=cfg.response_eps,
        denom_eps=cfg.denom_eps,
        name=name,
        response_prefix=response_prefix,
        check_nonnegative_input=cfg.check_nonnegative_input,
        include_bias=cfg.include_bias,
        dtype=dtype,
        device=device,
    )


def build_rational_network(
    *,
    input_dim: int,
    widths: Sequence[int],
    config: DenseRationalConfig | None = None,
    layer_configs: Sequence[DenseRationalConfig] | None = None,
    initialization: NetworkInitialization | None = None,
    output_total: float = 1.0,
    name: str = "rational_network",
    layer_name_prefix: str = "rational",
    check_finite_input: bool = True,
    check_nonnegative_input: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> RationalNetwork:
    check_positive_int("input_dim", input_dim)
    clean_widths = _normalize_positive_int_sequence(
        "widths",
        widths,
        allow_empty=False,
    )
    check_positive_float("output_total", output_total)
    check_non_empty_str("name", name)
    check_non_empty_str("layer_name_prefix", layer_name_prefix)
    _check_bool("check_finite_input", check_finite_input)
    _check_bool("check_nonnegative_input", check_nonnegative_input)

    if config is not None and layer_configs is not None:
        raise ValueError("Pass config or layer_configs, not both.")
    if layer_configs is None:
        configs = (
            DenseRationalConfig() if config is None else config,
        ) * len(clean_widths)
    else:
        configs = tuple(layer_configs)
        if len(configs) != len(clean_widths):
            raise ValueError("layer_configs length must equal len(widths).")
        if not all(isinstance(item, DenseRationalConfig) for item in configs):
            raise TypeError(
                "layer_configs must contain DenseRationalConfig objects."
            )

    layers: list[DenseRationalLayer] = []
    current_dim = input_dim
    for index, (width, layer_config) in enumerate(zip(clean_widths, configs)):
        layers.append(
            build_dense_rational_layer(
                input_dim=current_dim,
                width=width,
                config=layer_config,
                name=f"{layer_name_prefix}{index}",
                response_prefix=f"R{index}",
                dtype=dtype,
                device=device,
            )
        )
        current_dim = width

    output = CompetitiveOutputLayer(
        n_classes=clean_widths[-1],
        total=output_total,
        name=f"{name}_output",
        response_prefix=f"R{len(clean_widths) - 1}",
    )
    network = RationalNetwork(
        layers=layers,
        output=output,
        name=name,
        check_finite_input=check_finite_input,
        check_nonnegative_input=check_nonnegative_input,
    )
    if initialization is not None:
        initialize_rational_network_(network, initialization)
    return network


def build_basis_rational_classifier(
    *,
    n_inputs: int,
    n_classes: int,
    hidden_widths: Sequence[int] = (),
    basis_config: BasisConfig | None = None,
    rational_config: DenseRationalConfig | None = None,
    layer_configs: Sequence[DenseRationalConfig] | None = None,
    initialization: NetworkInitialization | None = None,
    output_total: float = 1.0,
    name: str = "basis_rational_classifier",
    network_name: str | None = None,
    layer_name_prefix: str | None = None,
    model_check_finite_input: bool = True,
    network_check_finite_input: bool = True,
    network_check_nonnegative_input: bool = True,
    input_gate_config: GateConfig | None = None,
    feature_gate_config: GateConfig | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> BasisRationalModel:
    check_positive_int("n_inputs", n_inputs)
    check_positive_int("n_classes", n_classes)
    check_non_empty_str("name", name)
    hidden = _normalize_positive_int_sequence(
        "hidden_widths",
        hidden_widths,
        allow_empty=True,
    )

    if basis_config is None:
        basis_config = PolynomialBasisConfig(n_inputs=n_inputs)
    if not isinstance(
        basis_config,
        (PolynomialBasisConfig, SimplexBasisConfig),
    ):
        raise TypeError("basis_config has an unsupported type.")
    if basis_config.n_inputs != n_inputs:
        raise ValueError("basis_config.n_inputs must equal n_inputs.")

    basis = build_basis_layer(basis_config)
    network = build_rational_network(
        input_dim=basis.n_basis,
        widths=hidden + (n_classes,),
        config=rational_config,
        layer_configs=layer_configs,
        initialization=initialization,
        output_total=output_total,
        name=network_name or f"{name}_network",
        layer_name_prefix=layer_name_prefix or f"{name}_rational",
        check_finite_input=network_check_finite_input,
        check_nonnegative_input=network_check_nonnegative_input,
        dtype=dtype,
        device=device,
    )
    return BasisRationalModel(
        basis=basis,
        network=network,
        name=name,
        check_finite_input=model_check_finite_input,
        input_gate=build_input_gate(
            n_inputs=basis.n_inputs,
            config=input_gate_config,
            dtype=dtype,
            device=device,
        ),
        feature_gate=build_feature_gate(
            n_features=basis.n_basis,
            config=feature_gate_config,
            dtype=dtype,
            device=device,
        ),
    )


def _normalize_positive_int_sequence(
    name: str,
    values: Sequence[int],
    *,
    allow_empty: bool,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of positive ints.")
    items = tuple(values)
    if not items and not allow_empty:
        raise ValueError(f"{name} must not be empty.")
    for index, value in enumerate(items):
        check_positive_int(f"{name}[{index}]", value)
    return tuple(int(value) for value in items)


def _check_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")


def _check_rate_init(
    name: str,
    value: float | Sequence[float] | torch.Tensor,
    *,
    allow_zero: bool,
) -> None:
    if isinstance(value, torch.Tensor):
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if allow_zero:
            check_nonnegative_float(name, float(value))
        else:
            check_positive_float(name, float(value))
        return
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a scalar, sequence, or tensor.")
    items = tuple(value)
    if not items:
        raise ValueError(f"{name} must not be empty.")
    for index, item in enumerate(items):
        if allow_zero:
            check_nonnegative_float(f"{name}[{index}]", float(item))
        else:
            check_positive_float(f"{name}[{index}]", float(item))


__all__ = [
    "BasisConfig",
    "DenseRationalConfig",
    "GateConfig",
    "PolynomialBasisConfig",
    "SimplexBasisConfig",
    "build_basis_layer",
    "build_basis_rational_classifier",
    "build_dense_rational_layer",
    "build_feature_gate",
    "build_input_gate",
    "build_polynomial_basis",
    "build_rational_network",
    "build_simplex_basis",
]
