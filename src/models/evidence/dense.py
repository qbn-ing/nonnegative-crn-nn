from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn

from models.basis import rate
from models.positive import ExactNonnegativeParam, PositiveParam
from utils.validation import (
    check_non_empty_str,
    check_nonnegative_float,
    check_positive_int,
)

from .base import BaseEvidenceLayer, EvidenceState
from .rational_specs import RationalLayerSpec, rational_response_spec
from .specs import (
    EvidenceLayerSpec,
    evidence_bias,
    evidence_edge,
    evidence_layer_spec,
)


WeightParameterization = Literal["positive", "exact_nonnegative"]
NonnegativeParameter = PositiveParam | ExactNonnegativeParam


class DenseEvidenceLayer(BaseEvidenceLayer):
    """Dense nonnegative affine E/I aggregation.

    The numerical map is

    ``E = h @ W_E.T + e`` and ``I = h @ W_I.T + i``.

    ``e`` and ``i`` are explicit trainable nonnegative vectors. They are not
    input features and are not columns of ``W_E`` or ``W_I``. Hard structural
    pruning applies only to variable-input edges.
    """

    def __init__(
        self,
        *,
        input_dim: int | None = None,
        output_dim: int | None = None,
        init_rate: float = 0.1,
        include_bias: bool = False,
        bias_init_rate: float | None = None,
        eps: float = 1e-8,
        weight_parameterization: WeightParameterization = "positive",
        name: str = "dense_evidence",
        check_nonnegative_phi: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        n_basis: int | None = None,
        n_classes: int | None = None,
    ) -> None:
        super().__init__()
        resolved_input = _resolve_dimension(
            "input_dim",
            input_dim,
            "n_basis",
            n_basis,
        )
        resolved_output = _resolve_dimension(
            "output_dim",
            output_dim,
            "n_classes",
            n_classes,
        )
        check_positive_int("input_dim", resolved_input)
        check_positive_int("output_dim", resolved_output)
        check_non_empty_str("name", name)
        check_nonnegative_float("eps", eps)
        check_nonnegative_float("init_rate", init_rate)
        if not isinstance(include_bias, bool):
            raise TypeError("include_bias must be a bool.")
        if not isinstance(check_nonnegative_phi, bool):
            raise TypeError("check_nonnegative_phi must be a bool.")
        if weight_parameterization not in {"positive", "exact_nonnegative"}:
            raise ValueError(
                "weight_parameterization must be 'positive' or "
                "'exact_nonnegative'."
            )

        bias_rate = init_rate if bias_init_rate is None else bias_init_rate
        check_nonnegative_float("bias_init_rate", bias_rate)
        for label, value in (
            ("init_rate", init_rate),
            ("bias_init_rate", bias_rate),
        ):
            if (
                weight_parameterization == "positive"
                and value <= eps
                and (label == "init_rate" or include_bias)
            ):
                raise ValueError(
                    f"{label} must be greater than eps for PositiveParam."
                )

        self._input_dim = resolved_input
        self._output_dim = resolved_output
        self._name = name
        self.check_nonnegative_phi = check_nonnegative_phi
        self.weight_parameterization = weight_parameterization
        self.include_bias = include_bias

        self.w_pos = _make_nonnegative_parameter(
            parameterization=weight_parameterization,
            shape=(resolved_output, resolved_input),
            init=init_rate,
            eps=eps,
            dtype=dtype,
            device=device,
        )
        self.w_neg = _make_nonnegative_parameter(
            parameterization=weight_parameterization,
            shape=(resolved_output, resolved_input),
            init=init_rate,
            eps=eps,
            dtype=dtype,
            device=device,
        )
        if include_bias:
            self.b_pos: NonnegativeParameter | None = (
                _make_nonnegative_parameter(
                    parameterization=weight_parameterization,
                    shape=(resolved_output,),
                    init=bias_rate,
                    eps=eps,
                    dtype=dtype,
                    device=device,
                )
            )
            self.b_neg: NonnegativeParameter | None = (
                _make_nonnegative_parameter(
                    parameterization=weight_parameterization,
                    shape=(resolved_output,),
                    init=bias_rate,
                    eps=eps,
                    dtype=dtype,
                    device=device,
                )
            )
        else:
            self.b_pos = None
            self.b_neg = None

        self.register_buffer(
            "w_pos_hard_mask",
            torch.ones(
                (resolved_output, resolved_input),
                dtype=dtype,
                device=device,
            ),
        )
        self.register_buffer(
            "w_neg_hard_mask",
            torch.ones(
                (resolved_output, resolved_input),
                dtype=dtype,
                device=device,
            ),
        )

    def forward(self, phi: torch.Tensor) -> EvidenceState:
        self.validate_input(phi)
        if self.check_nonnegative_phi and torch.any(phi < 0):
            raise ValueError("DenseEvidenceLayer input must be nonnegative.")

        w_pos, w_neg = self.effective_weights()
        if w_pos.device != phi.device or w_neg.device != phi.device:
            raise ValueError(
                "DenseEvidenceLayer parameters and input are on different "
                "devices."
            )
        w_pos = w_pos.to(dtype=phi.dtype)
        w_neg = w_neg.to(dtype=phi.dtype)
        E = phi @ w_pos.t()
        I = phi @ w_neg.t()
        if self.include_bias:
            b_pos, b_neg = self.effective_biases()
            E = E + b_pos.to(dtype=phi.dtype).unsqueeze(0)
            I = I + b_neg.to(dtype=phi.dtype).unsqueeze(0)

        state = EvidenceState(E=E, I=I)
        self.validate_state(state)
        return state

    def raw_positive_weights(self) -> torch.Tensor:
        return self.w_pos()

    def raw_negative_weights(self) -> torch.Tensor:
        return self.w_neg()

    def effective_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.w_pos() * self.w_pos_hard_mask,
            self.w_neg() * self.w_neg_hard_mask,
        )

    def positive_weights(self) -> torch.Tensor:
        return self.effective_weights()[0]

    def negative_weights(self) -> torch.Tensor:
        return self.effective_weights()[1]

    def raw_positive_bias(self) -> torch.Tensor:
        if self.b_pos is None:
            return self.w_pos().new_zeros((self.output_dim,))
        return self.b_pos()

    def raw_negative_bias(self) -> torch.Tensor:
        if self.b_neg is None:
            return self.w_neg().new_zeros((self.output_dim,))
        return self.b_neg()

    def effective_biases(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.raw_positive_bias(), self.raw_negative_bias()

    def rate_l1_loss(self, reduction: str = "sum") -> torch.Tensor:
        values = list(self.effective_weights())
        if self.include_bias:
            values.extend(self.effective_biases())
        total = sum(value.sum() for value in values)
        if reduction == "sum":
            return total
        if reduction == "mean":
            return total / sum(value.numel() for value in values)
        raise ValueError("reduction must be 'sum' or 'mean'.")

    def evidence_spec(self) -> EvidenceLayerSpec:
        return self._make_evidence_spec(export=False)

    def export_evidence_spec(self) -> EvidenceLayerSpec:
        return self._make_evidence_spec(export=True)

    def _make_evidence_spec(self, *, export: bool) -> EvidenceLayerSpec:
        w_pos, w_neg = self.effective_weights()
        edges = []
        for kind, values, mask, parameter_name in (
            ("positive", w_pos, self.w_pos_hard_mask, "w_pos"),
            ("negative", w_neg, self.w_neg_hard_mask, "w_neg"),
        ):
            for output_index in range(self.output_dim):
                for input_index in range(self.input_dim):
                    if not bool(mask[output_index, input_index].item()):
                        continue
                    value = float(values[output_index, input_index].detach().item())
                    if export and value <= 0.0:
                        continue
                    target_prefix = "E" if kind == "positive" else "I"
                    edges.append(
                        evidence_edge(
                            source_input_index=input_index,
                            target_output_index=output_index,
                            kind=kind,
                            rate_ref=rate(
                                f"a_{target_prefix}_{output_index}_{input_index}",
                                value=value if export else None,
                                trainable=not export,
                                param_ref=(
                                    None
                                    if export
                                    else self._parameter_ref(
                                        parameter_name,
                                        output_index,
                                        input_index,
                                    )
                                ),
                            ),
                            source_input_species=(
                                f"Phi_{input_index}" if export else None
                            ),
                            target_species=(
                                f"{target_prefix}_{output_index}"
                                if export
                                else None
                            ),
                        )
                    )

        biases = []
        if self.include_bias:
            b_pos, b_neg = self.effective_biases()
            for kind, values, parameter_name in (
                ("positive", b_pos, "b_pos"),
                ("negative", b_neg, "b_neg"),
            ):
                for output_index in range(self.output_dim):
                    value = float(values[output_index].detach().item())
                    if export and value <= 0.0:
                        continue
                    target_prefix = "E" if kind == "positive" else "I"
                    biases.append(
                        evidence_bias(
                            target_output_index=output_index,
                            kind=kind,
                            rate_ref=rate(
                                f"b_{target_prefix}_{output_index}",
                                value=value if export else None,
                                trainable=not export,
                                param_ref=(
                                    None
                                    if export
                                    else self._vector_parameter_ref(
                                        parameter_name,
                                        output_index,
                                    )
                                ),
                            ),
                            target_species=(
                                f"{target_prefix}_{output_index}"
                                if export
                                else None
                            ),
                        )
                    )

        return evidence_layer_spec(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            edges=edges,
            biases=biases,
            name=self._name,
        )

    @torch.no_grad()
    def active_edge_masks(
        self,
        threshold: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        check_nonnegative_float("threshold", threshold)
        w_pos, w_neg = self.effective_weights()
        return w_pos.detach() > threshold, w_neg.detach() > threshold

    @torch.no_grad()
    def apply_hard_pruning(
        self,
        *,
        threshold: float = 1e-3,
        keep_per_class: bool = True,
    ) -> dict[str, Any]:
        """Prune variable edges; explicit biases are deliberately preserved."""

        check_nonnegative_float("threshold", threshold)
        old_pos = self.w_pos_hard_mask.detach() > 0.0
        old_neg = self.w_neg_hard_mask.detach() > 0.0
        raw_pos = self.raw_positive_weights().detach() * old_pos
        raw_neg = self.raw_negative_weights().detach() * old_neg
        new_pos = raw_pos > threshold
        new_neg = raw_neg > threshold
        if keep_per_class:
            self._keep_best_edge_per_output(raw_pos, new_pos)
            self._keep_best_edge_per_output(raw_neg, new_neg)
        self.w_pos_hard_mask.copy_(new_pos.to(self.w_pos_hard_mask.dtype))
        self.w_neg_hard_mask.copy_(new_neg.to(self.w_neg_hard_mask.dtype))
        return _pruning_summary(old_pos, old_neg, new_pos, new_neg)

    @staticmethod
    def _keep_best_edge_per_output(
        weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        for output_index in range(weights.shape[0]):
            if bool(mask[output_index].any()):
                continue
            input_index = int(torch.argmax(weights[output_index]).item())
            mask[output_index, input_index] = True

    @torch.no_grad()
    def reset_hard_pruning(self) -> None:
        self.w_pos_hard_mask.fill_(1.0)
        self.w_neg_hard_mask.fill_(1.0)

    def structure_info(self) -> dict[str, Any]:
        info = super().structure_info()
        active_pos, active_neg = self.active_edge_masks(threshold=0.0)
        info.update(
            {
                "dense": True,
                "has_trainable_rates": True,
                # Compatibility field retained for callers that predate the
                # explicit weight_parameterization string.
                "positive_param": self.weight_parameterization == "positive",
                "weight_parameterization": self.weight_parameterization,
                "check_nonnegative_phi": self.check_nonnegative_phi,
                "include_bias": self.include_bias,
                "bias_parameter_count": (
                    2 * self.output_dim if self.include_bias else 0
                ),
                "hard_pruned": bool(
                    (self.w_pos_hard_mask == 0.0).any().item()
                    or (self.w_neg_hard_mask == 0.0).any().item()
                ),
                "active_positive_edges": int(active_pos.sum().item()),
                "active_negative_edges": int(active_neg.sum().item()),
            }
        )
        return info

    def _parameter_ref(self, name: str, row: int, column: int) -> str:
        field = "raw" if self.weight_parameterization == "positive" else "value"
        return f"{name}.{field}[{row}, {column}]"

    def _vector_parameter_ref(self, name: str, index: int) -> str:
        field = "raw" if self.weight_parameterization == "positive" else "value"
        return f"{name}.{field}[{index}]"

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def n_basis(self) -> int:
        return self.input_dim

    @property
    def n_classes(self) -> int:
        return self.output_dim

    @property
    def n_species(self) -> int:
        return 2 * self.output_dim

    @property
    def n_edges(self) -> int:
        pos, neg = self.active_edge_masks(threshold=0.0)
        return int(pos.sum().item() + neg.sum().item())

    @property
    def n_positive_edges(self) -> int:
        return int(self.active_edge_masks(threshold=0.0)[0].sum().item())

    @property
    def n_negative_edges(self) -> int:
        return int(self.active_edge_masks(threshold=0.0)[1].sum().item())

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"n_basis={self.n_basis}, n_classes={self.n_classes}, "
            f"include_bias={self.include_bias}, name={self._name!r}, "
            f"weight_parameterization={self.weight_parameterization!r}, "
            f"check_nonnegative_phi={self.check_nonnegative_phi}"
        )


@dataclass(frozen=True)
class DenseRationalState:
    E: torch.Tensor
    I: torch.Tensor
    r: torch.Tensor

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, torch.Tensor) for value in (self.E, self.I, self.r)
        ):
            raise TypeError("E, I, and r must be torch.Tensor objects.")
        if self.E.ndim != 2:
            raise ValueError("E, I, and r must be 2D tensors.")
        if self.E.shape != self.I.shape or self.E.shape != self.r.shape:
            raise ValueError("E, I, and r must have the same shape.")

    @property
    def batch_size(self) -> int:
        return int(self.r.shape[0])

    @property
    def width(self) -> int:
        return int(self.r.shape[1])

    def detach(self) -> DenseRationalState:
        return DenseRationalState(
            E=self.E.detach(),
            I=self.I.detach(),
            r=self.r.detach(),
        )


class DenseRationalLayer(nn.Module):
    """Dense nonnegative affine E/I-to-r map."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        init_rate: float = 0.1,
        bias_init_rate: float | None = None,
        weight_parameterization: WeightParameterization = "positive",
        eta_init: float | list[float] | torch.Tensor = 1.0,
        gamma_init: float | list[float] | torch.Tensor = 1.0,
        chi_init: float | list[float] | torch.Tensor = 1.0,
        evidence_eps: float = 1e-8,
        response_eps: float = 1e-8,
        denom_eps: float = 1e-12,
        name: str = "dense_rational",
        response_prefix: str = "R",
        check_nonnegative_input: bool = True,
        include_bias: bool | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        include_constant_channel: bool | None = None,
        constant_value: float = 1.0,
    ) -> None:
        super().__init__()
        check_positive_int("in_features", in_features)
        check_positive_int("out_features", out_features)
        check_non_empty_str("name", name)
        check_non_empty_str("response_prefix", response_prefix)
        check_nonnegative_float("denom_eps", denom_eps)
        if not isinstance(check_nonnegative_input, bool):
            raise TypeError("check_nonnegative_input must be a bool.")
        resolved_bias = _resolve_bias_flag(
            include_bias,
            include_constant_channel,
        )
        if constant_value != 1.0:
            raise ValueError(
                "constant_value is a legacy argument and must equal 1.0; "
                "v9 represents affine offsets as explicit bias parameters."
            )

        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        self.response_prefix = response_prefix
        self.denom_eps = float(denom_eps)
        self.check_nonnegative_input = check_nonnegative_input
        self.include_bias = resolved_bias

        self.evidence = DenseEvidenceLayer(
            input_dim=in_features,
            output_dim=out_features,
            init_rate=init_rate,
            include_bias=resolved_bias,
            bias_init_rate=bias_init_rate,
            eps=evidence_eps,
            weight_parameterization=weight_parameterization,
            name=f"{name}_evidence",
            check_nonnegative_phi=check_nonnegative_input,
            dtype=dtype,
            device=device,
        )
        self.eta = PositiveParam(
            shape=(out_features,),
            init=eta_init,
            eps=response_eps,
            dtype=dtype,
            device=device,
        )
        self.gamma = PositiveParam(
            shape=(out_features,),
            init=gamma_init,
            eps=response_eps,
            dtype=dtype,
            device=device,
        )
        self.chi = PositiveParam(
            shape=(out_features,),
            init=chi_init,
            eps=response_eps,
            dtype=dtype,
            device=device,
        )

    def forward(self, h: torch.Tensor) -> DenseRationalState:
        evidence = self.evidence(h)
        eta = self.eta().to(dtype=evidence.E.dtype)
        gamma = self.gamma().to(dtype=evidence.E.dtype)
        chi = self.chi().to(dtype=evidence.E.dtype)
        denominator = gamma.unsqueeze(0) + chi.unsqueeze(0) * evidence.I
        if self.denom_eps > 0.0:
            denominator = denominator.clamp_min(self.denom_eps)
        r = eta.unsqueeze(0) * evidence.E / denominator
        return DenseRationalState(E=evidence.E, I=evidence.I, r=r)

    @property
    def evidence_input_dim(self) -> int:
        return self.in_features

    @property
    def include_constant_channel(self) -> bool:
        return self.include_bias

    @property
    def constant_value(self) -> float:
        return 1.0

    def evidence_input(self, h: torch.Tensor) -> torch.Tensor:
        """Compatibility helper; v9 returns variable inputs unchanged."""

        if not isinstance(h, torch.Tensor):
            raise TypeError("h must be a torch.Tensor.")
        if h.ndim != 2 or h.shape[1] != self.in_features:
            raise ValueError(
                f"h must have shape (batch_size, {self.in_features})."
            )
        return h

    def response_rates(self) -> dict[str, torch.Tensor]:
        return {"eta": self.eta(), "gamma": self.gamma(), "chi": self.chi()}

    def rational_spec(self, *, export: bool = False) -> RationalLayerSpec:
        evidence_spec = (
            self.evidence.export_evidence_spec()
            if export
            else self.evidence.evidence_spec()
        )
        if export:
            values = {
                "eta": self.eta().detach().cpu(),
                "gamma": self.gamma().detach().cpu(),
                "chi": self.chi().detach().cpu(),
            }
            rates = {
                name: tuple(
                    rate(
                        f"{name}_{index}",
                        value=float(tensor[index].item()),
                        trainable=False,
                    )
                    for index in range(self.out_features)
                )
                for name, tensor in values.items()
            }
        else:
            rates = {
                name: tuple(
                    rate(
                        f"{name}_{index}",
                        value=None,
                        trainable=True,
                        param_ref=f"{name}.raw[{index}]",
                    )
                    for index in range(self.out_features)
                )
                for name in ("eta", "gamma", "chi")
            }
        return RationalLayerSpec(
            name=self.name,
            input_dim=self.in_features,
            width=self.out_features,
            evidence=evidence_spec,
            response=rational_response_spec(
                eta=rates["eta"],
                gamma=rates["gamma"],
                chi=rates["chi"],
                response_prefix=self.response_prefix,
            ),
            include_bias=self.include_bias,
            bias_catalyst_initial=1.0,
        )

    def compile_spec(self) -> RationalLayerSpec:
        return self.rational_spec()

    def export_rational_spec(self) -> RationalLayerSpec:
        return self.rational_spec(export=True)

    def rate_l1_loss(self, reduction: str = "sum") -> torch.Tensor:
        evidence_loss = self.evidence.rate_l1_loss(reduction=reduction)
        response = list(self.response_rates().values())
        response_loss = sum(value.sum() for value in response)
        if reduction == "mean":
            response_loss = response_loss / sum(
                value.numel() for value in response
            )
        return evidence_loss + response_loss

    def structure_info(self) -> dict[str, Any]:
        spec = self.rational_spec()
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "include_bias": self.include_bias,
            "n_bias_parameters": (
                2 * self.out_features if self.include_bias else 0
            ),
            "n_species": spec.n_species,
            "n_reactions": spec.n_reactions,
            "weight_parameterization": self.evidence.weight_parameterization,
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"include_bias={self.include_bias}, name={self.name!r}"
        )


def _make_nonnegative_parameter(
    *,
    parameterization: WeightParameterization,
    shape: tuple[int, ...],
    init: float,
    eps: float,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> NonnegativeParameter:
    if parameterization == "positive":
        return PositiveParam(
            shape=shape,
            init=init,
            eps=eps,
            dtype=dtype,
            device=device,
        )
    return ExactNonnegativeParam(
        shape=shape,
        init=init,
        dtype=dtype,
        device=device,
    )


def _resolve_dimension(
    canonical_name: str,
    canonical_value: int | None,
    legacy_name: str,
    legacy_value: int | None,
) -> int:
    if canonical_value is None and legacy_value is None:
        raise TypeError(f"{canonical_name} is required.")
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


def _resolve_bias_flag(
    include_bias: bool | None,
    include_constant_channel: bool | None,
) -> bool:
    if include_bias is not None and not isinstance(include_bias, bool):
        raise TypeError("include_bias must be a bool or None.")
    if (
        include_constant_channel is not None
        and not isinstance(include_constant_channel, bool)
    ):
        raise TypeError("include_constant_channel must be a bool or None.")
    if (
        include_bias is not None
        and include_constant_channel is not None
        and include_bias != include_constant_channel
    ):
        raise ValueError(
            "include_bias and include_constant_channel must agree when both "
            "are set."
        )
    if include_bias is not None:
        return include_bias
    if include_constant_channel is not None:
        return include_constant_channel
    return False


def _pruning_summary(
    old_pos: torch.Tensor,
    old_neg: torch.Tensor,
    new_pos: torch.Tensor,
    new_neg: torch.Tensor,
) -> dict[str, Any]:
    before_pos = int(old_pos.sum().item())
    before_neg = int(old_neg.sum().item())
    after_pos = int(new_pos.sum().item())
    after_neg = int(new_neg.sum().item())
    return {
        "positive": {
            "before": before_pos,
            "after": after_pos,
            "pruned": before_pos - after_pos,
            "total": int(old_pos.numel()),
        },
        "negative": {
            "before": before_neg,
            "after": after_neg,
            "pruned": before_neg - after_neg,
            "total": int(old_neg.numel()),
        },
        "total": {
            "before": before_pos + before_neg,
            "after": after_pos + after_neg,
            "pruned": before_pos + before_neg - after_pos - after_neg,
            "total": int(old_pos.numel() + old_neg.numel()),
        },
    }


__all__ = [
    "DenseEvidenceLayer",
    "DenseRationalLayer",
    "DenseRationalState",
    "WeightParameterization",
]
