from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from models.composition.rational_network import RationalNetwork
from models.evidence.dense import DenseRationalLayer
from models.positive import ExactNonnegativeParam, PositiveParam
from utils.validation import check_positive_float


@dataclass(frozen=True)
class ConstantDenseInitialization:
    """Initialize every dense evidence edge to one positive constant."""

    value: float = 0.1
    eta: float = 1.0
    gamma: float = 1.0
    chi: float = 1.0

    def __post_init__(self) -> None:
        for name in ("value", "eta", "gamma", "chi"):
            check_positive_float(name, getattr(self, name))


@dataclass(frozen=True)
class NonnegativeXavierUniformInitialization:
    """Nonnegative Xavier-uniform initialization of effective evidence rates.

    A standard Xavier-uniform draw has support ``[-a, a]``.  Taking its
    magnitude gives ``U(0, a)`` and preserves the Xavier second moment while
    satisfying the nonnegative parameter domain.
    """

    gain: float = 1.0
    eta: float = 1.0
    gamma: float = 1.0
    chi: float = 1.0

    def __post_init__(self) -> None:
        for name in ("gain", "eta", "gamma", "chi"):
            check_positive_float(name, getattr(self, name))


DenseInitialization = (
    ConstantDenseInitialization
    | NonnegativeXavierUniformInitialization
)


@dataclass(frozen=True)
class IdentityInitialization:
    """Topology-aware identity initialization for hidden square transitions.

    Non-square layers and the final rational layer directly followed by the O
    layer use ``dense``.  The latter rule is topology based, so a square
    class-evidence layer is never mistaken for a hidden identity transition.
    """

    dense: DenseInitialization = field(
        default_factory=NonnegativeXavierUniformInitialization
    )
    response_rate: float = 1.0
    chi: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(
            self.dense,
            (
                ConstantDenseInitialization,
                NonnegativeXavierUniformInitialization,
            ),
        ):
            raise TypeError(
                "dense must be a ConstantDenseInitialization or "
                "NonnegativeXavierUniformInitialization."
            )
        check_positive_float("response_rate", self.response_rate)
        check_positive_float("chi", self.chi)


NetworkInitialization = DenseInitialization | IdentityInitialization
AppliedInitialization = Literal[
    "constant_dense",
    "nonnegative_xavier_uniform",
    "identity",
]


@dataclass(frozen=True)
class LayerInitializationAudit:
    index: int
    in_features: int
    out_features: int
    followed_by_output: bool
    applied_initialization: AppliedInitialization
    weight_parameterization: str
    include_bias: bool
    exact_zero_count: int
    positive_diagonal_count: int
    eta_gamma_max_abs_diff: float
    algebraic_identity_error: float | None

    @property
    def include_constant_channel(self) -> bool:
        return self.include_bias

    @property
    def constant_value(self) -> float:
        return 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "followed_by_output": self.followed_by_output,
            "applied_initialization": self.applied_initialization,
            "weight_parameterization": self.weight_parameterization,
            "include_bias": self.include_bias,
            "include_constant_channel": self.include_bias,
            "constant_value": 1.0,
            "exact_zero_count": self.exact_zero_count,
            "positive_diagonal_count": self.positive_diagonal_count,
            "eta_gamma_max_abs_diff": self.eta_gamma_max_abs_diff,
            "algebraic_identity_error": self.algebraic_identity_error,
        }


@dataclass(frozen=True)
class InitializationAudit:
    layers: tuple[LayerInitializationAudit, ...]

    @property
    def identity_layer_indices(self) -> tuple[int, ...]:
        return tuple(
            layer.index
            for layer in self.layers
            if layer.applied_initialization == "identity"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "layers": [layer.to_dict() for layer in self.layers],
            "identity_layer_indices": self.identity_layer_indices,
        }


@torch.no_grad()
def initialize_dense_layer_(
    layer: DenseRationalLayer,
    initialization: DenseInitialization,
    *,
    index: int = 0,
    followed_by_output: bool = False,
) -> LayerInitializationAudit:
    """Initialize one rational layer with a valid dense strategy."""

    _check_layer(layer)
    _check_layer_index(index)
    if not isinstance(followed_by_output, bool):
        raise TypeError("followed_by_output must be a bool.")
    if not isinstance(
        initialization,
        (
            ConstantDenseInitialization,
            NonnegativeXavierUniformInitialization,
        ),
    ):
        raise TypeError(
            "initialization must be a ConstantDenseInitialization or "
            "NonnegativeXavierUniformInitialization."
        )
    _validate_dense_compatibility(layer, initialization)

    if isinstance(initialization, ConstantDenseInitialization):
        positive = torch.full_like(
            layer.evidence.raw_positive_weights(),
            initialization.value,
        )
        negative = torch.full_like(
            layer.evidence.raw_negative_weights(),
            initialization.value,
        )
        positive_bias = torch.full_like(
            layer.evidence.raw_positive_bias(),
            initialization.value,
        )
        negative_bias = torch.full_like(
            layer.evidence.raw_negative_bias(),
            initialization.value,
        )
        applied: AppliedInitialization = "constant_dense"
    elif isinstance(initialization, NonnegativeXavierUniformInitialization):
        bound = initialization.gain * math.sqrt(
            6.0 / (layer.evidence_input_dim + layer.out_features)
        )
        positive = _sample_nonnegative_xavier(
            layer.evidence.w_pos,
            bound=bound,
        )
        negative = _sample_nonnegative_xavier(
            layer.evidence.w_neg,
            bound=bound,
        )
        positive_bias = (
            _sample_nonnegative_xavier(
                layer.evidence.b_pos,
                bound=bound,
            )
            if layer.evidence.b_pos is not None
            else layer.evidence.raw_positive_bias()
        )
        negative_bias = (
            _sample_nonnegative_xavier(
                layer.evidence.b_neg,
                bound=bound,
            )
            if layer.evidence.b_neg is not None
            else layer.evidence.raw_negative_bias()
        )
        applied = "nonnegative_xavier_uniform"
    else:
        raise AssertionError("Unhandled dense initialization.")

    layer.evidence.w_pos.set_value_(positive)
    layer.evidence.w_neg.set_value_(negative)
    if layer.evidence.b_pos is not None:
        layer.evidence.b_pos.set_value_(positive_bias)
    if layer.evidence.b_neg is not None:
        layer.evidence.b_neg.set_value_(negative_bias)
    layer.eta.set_value_(initialization.eta)
    layer.gamma.set_value_(initialization.gamma)
    layer.chi.set_value_(initialization.chi)

    return _audit_layer(
        layer,
        index=index,
        followed_by_output=followed_by_output,
        applied_initialization=applied,
    )


@torch.no_grad()
def initialize_identity_layer_(
    layer: DenseRationalLayer,
    initialization: IdentityInitialization,
    *,
    index: int = 0,
    followed_by_output: bool = False,
) -> LayerInitializationAudit:
    """Initialize one square exact-nonnegative layer as the identity map."""

    _check_layer(layer)
    _check_layer_index(index)
    if not isinstance(initialization, IdentityInitialization):
        raise TypeError("initialization must be an IdentityInitialization.")
    if not isinstance(followed_by_output, bool):
        raise TypeError("followed_by_output must be a bool.")
    if followed_by_output:
        raise ValueError(
            "The rational layer directly followed by O must use dense "
            "initialization."
        )
    _validate_identity_compatibility(layer, initialization)

    identity = torch.zeros(
        (layer.out_features, layer.in_features),
        dtype=layer.evidence.w_pos.value.dtype,
        device=layer.evidence.w_pos.value.device,
    )
    identity.copy_(
        torch.eye(
            layer.out_features,
            dtype=identity.dtype,
            device=identity.device,
        )
    )
    layer.evidence.w_pos.set_value_(identity)
    layer.evidence.w_neg.set_value_(torch.zeros_like(identity))
    if layer.evidence.b_pos is not None:
        layer.evidence.b_pos.set_value_(
            torch.zeros_like(layer.evidence.raw_positive_bias())
        )
    if layer.evidence.b_neg is not None:
        layer.evidence.b_neg.set_value_(
            torch.zeros_like(layer.evidence.raw_negative_bias())
        )
    layer.eta.set_value_(initialization.response_rate)
    layer.gamma.set_value_(initialization.response_rate)
    layer.chi.set_value_(initialization.chi)

    return _audit_layer(
        layer,
        index=index,
        followed_by_output=False,
        applied_initialization="identity",
    )


@torch.no_grad()
def initialize_rational_network_(
    network: RationalNetwork,
    initialization: NetworkInitialization,
) -> InitializationAudit:
    """Initialize a constructed network using topology-derived layer choices."""

    if not isinstance(network, RationalNetwork):
        raise TypeError("network must be a RationalNetwork.")
    if not isinstance(
        initialization,
        (
            ConstantDenseInitialization,
            NonnegativeXavierUniformInitialization,
            IdentityInitialization,
        ),
    ):
        raise TypeError("initialization has an unsupported type.")

    terminal_index = network.depth - 1
    for index, layer in enumerate(network.layers):
        followed_by_output = index == terminal_index
        use_identity = (
            isinstance(initialization, IdentityInitialization)
            and not followed_by_output
            and layer.in_features == layer.out_features
        )
        if use_identity:
            _validate_identity_compatibility(layer, initialization)
        else:
            dense = (
                initialization.dense
                if isinstance(initialization, IdentityInitialization)
                else initialization
            )
            _validate_dense_compatibility(layer, dense)

    audits: list[LayerInitializationAudit] = []

    for index, layer in enumerate(network.layers):
        followed_by_output = index == terminal_index
        use_identity = (
            isinstance(initialization, IdentityInitialization)
            and not followed_by_output
            and layer.in_features == layer.out_features
        )
        if use_identity:
            audit = initialize_identity_layer_(
                layer,
                initialization,
                index=index,
                followed_by_output=False,
            )
        else:
            dense = (
                initialization.dense
                if isinstance(initialization, IdentityInitialization)
                else initialization
            )
            audit = initialize_dense_layer_(
                layer,
                dense,
                index=index,
                followed_by_output=followed_by_output,
            )
        audits.append(audit)

    result = InitializationAudit(layers=tuple(audits))
    network.initialization_audit = result
    return result


def _sample_nonnegative_xavier(
    parameter: PositiveParam | ExactNonnegativeParam | None,
    *,
    bound: float,
) -> torch.Tensor:
    if isinstance(parameter, PositiveParam):
        prototype = parameter.raw
        eps = parameter.eps
        minimum = torch.nextafter(
            prototype.new_tensor(eps),
            prototype.new_tensor(float("inf")),
        )
        if bound <= float(minimum.item()):
            raise ValueError(
                "Xavier bound must be greater than the PositiveParam eps."
            )
    elif isinstance(parameter, ExactNonnegativeParam):
        prototype = parameter.value
        minimum = prototype.new_tensor(0.0)
    elif parameter is None:
        raise TypeError("parameter must not be None.")
    else:
        raise TypeError("Unsupported nonnegative parameter module.")

    sample = torch.empty_like(prototype).uniform_(0.0, bound)
    if isinstance(parameter, PositiveParam):
        sample.clamp_(min=minimum)
    return sample


def _validate_dense_compatibility(
    layer: DenseRationalLayer,
    initialization: DenseInitialization,
) -> None:
    for name, parameter in (
        ("eta", layer.eta),
        ("gamma", layer.gamma),
        ("chi", layer.chi),
    ):
        value = getattr(initialization, name)
        if value <= parameter.eps:
            raise ValueError(
                f"Dense initialization {name} must be greater than its "
                "PositiveParam eps."
            )
    if isinstance(initialization, ConstantDenseInitialization):
        parameters = [layer.evidence.w_pos, layer.evidence.w_neg]
        if layer.evidence.b_pos is not None:
            parameters.extend((layer.evidence.b_pos, layer.evidence.b_neg))
        for parameter in parameters:
            if (
                isinstance(parameter, PositiveParam)
                and initialization.value <= parameter.eps
            ):
                raise ValueError(
                    "Constant dense value must be greater than the "
                    "PositiveParam eps."
                )
        return
    if isinstance(initialization, NonnegativeXavierUniformInitialization):
        bound = initialization.gain * math.sqrt(
            6.0 / (layer.evidence_input_dim + layer.out_features)
        )
        parameters = [layer.evidence.w_pos, layer.evidence.w_neg]
        if layer.evidence.b_pos is not None:
            parameters.extend((layer.evidence.b_pos, layer.evidence.b_neg))
        for parameter in parameters:
            if isinstance(parameter, PositiveParam) and bound <= parameter.eps:
                raise ValueError(
                    "Xavier bound must be greater than the PositiveParam eps."
                )
        return
    raise TypeError("Unsupported dense initialization.")


def _validate_identity_compatibility(
    layer: DenseRationalLayer,
    initialization: IdentityInitialization,
) -> None:
    if layer.in_features != layer.out_features:
        raise ValueError("Identity initialization requires a square layer.")
    if not isinstance(layer.evidence.w_pos, ExactNonnegativeParam) or not isinstance(
        layer.evidence.w_neg,
        ExactNonnegativeParam,
    ):
        raise ValueError(
            "Identity initialization requires exact_nonnegative evidence "
            "parameters."
        )
    if layer.include_bias and (
        not isinstance(layer.evidence.b_pos, ExactNonnegativeParam)
        or not isinstance(layer.evidence.b_neg, ExactNonnegativeParam)
    ):
        raise ValueError(
            "Identity initialization requires exact_nonnegative bias "
            "parameters."
        )
    if initialization.response_rate <= layer.eta.eps:
        raise ValueError(
            "Identity response_rate must be greater than eta/gamma eps."
        )
    if initialization.response_rate <= layer.gamma.eps:
        raise ValueError(
            "Identity response_rate must be greater than eta/gamma eps."
        )
    if initialization.chi <= layer.chi.eps:
        raise ValueError("Identity chi must be greater than chi eps.")


def _audit_layer(
    layer: DenseRationalLayer,
    *,
    index: int,
    followed_by_output: bool,
    applied_initialization: AppliedInitialization,
) -> LayerInitializationAudit:
    positive = layer.evidence.raw_positive_weights().detach()
    negative = layer.evidence.raw_negative_weights().detach()
    positive_bias = layer.evidence.raw_positive_bias().detach()
    negative_bias = layer.evidence.raw_negative_bias().detach()
    eta = layer.eta().detach()
    gamma = layer.gamma().detach()

    exact_zero_count = int(
        torch.count_nonzero(positive == 0).item()
        + torch.count_nonzero(negative == 0).item()
        + (
            torch.count_nonzero(positive_bias == 0).item()
            + torch.count_nonzero(negative_bias == 0).item()
            if layer.include_bias
            else 0
        )
    )
    diagonal = torch.diagonal(positive)
    positive_diagonal_count = int(torch.count_nonzero(diagonal > 0).item())
    eta_gamma_error = float(torch.max(torch.abs(eta - gamma)).item())

    identity_error: float | None = None
    if applied_initialization == "identity":
        expected = torch.zeros(
            (layer.out_features, layer.in_features),
            dtype=positive.dtype,
            device=positive.device,
        )
        expected.copy_(
            torch.eye(
                layer.out_features,
                dtype=positive.dtype,
                device=positive.device,
            )
        )
        scaled_positive = eta.unsqueeze(1) * positive / gamma.unsqueeze(1)
        identity_error = max(
            float(torch.max(torch.abs(scaled_positive - expected)).item()),
            float(torch.max(torch.abs(negative)).item()),
            float(torch.max(torch.abs(positive_bias)).item()),
            float(torch.max(torch.abs(negative_bias)).item()),
        )

    return LayerInitializationAudit(
        index=index,
        in_features=layer.in_features,
        out_features=layer.out_features,
        followed_by_output=followed_by_output,
        applied_initialization=applied_initialization,
        weight_parameterization=layer.evidence.weight_parameterization,
        include_bias=layer.include_bias,
        exact_zero_count=exact_zero_count,
        positive_diagonal_count=positive_diagonal_count,
        eta_gamma_max_abs_diff=eta_gamma_error,
        algebraic_identity_error=identity_error,
    )


def _check_layer(layer: DenseRationalLayer) -> None:
    if not isinstance(layer, DenseRationalLayer):
        raise TypeError("layer must be a DenseRationalLayer.")


def _check_layer_index(index: int) -> None:
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError("index must be an int.")
    if index < 0:
        raise ValueError("index must be nonnegative.")


__all__ = [
    "AppliedInitialization",
    "ConstantDenseInitialization",
    "DenseInitialization",
    "IdentityInitialization",
    "InitializationAudit",
    "LayerInitializationAudit",
    "NetworkInitialization",
    "NonnegativeXavierUniformInitialization",
    "initialize_dense_layer_",
    "initialize_identity_layer_",
    "initialize_rational_network_",
]
