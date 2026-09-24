from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from models.composition.rational_network import RationalNetwork
from models.evidence.dense import DenseRationalLayer
from models.initialization import (
    IdentityInitialization,
    LayerInitializationAudit,
    initialize_identity_layer_,
)
from utils.validation import check_nonnegative_float, check_positive_int


@dataclass(frozen=True)
class FunctionPreservationAudit:
    max_abs_r_error: float
    max_abs_Z_error: float
    max_abs_Z_free_error: float
    passed: bool
    atol: float
    rtol: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_abs_r_error": self.max_abs_r_error,
            "max_abs_Z_error": self.max_abs_Z_error,
            "max_abs_Z_free_error": self.max_abs_Z_free_error,
            "passed": self.passed,
            "atol": self.atol,
            "rtol": self.rtol,
        }


@dataclass(frozen=True)
class DepthExtensionReport:
    old_depth: int
    new_depth: int
    inserted_indices: tuple[int, ...]
    inserted_layers: tuple[LayerInitializationAudit, ...]
    function_audit: FunctionPreservationAudit | None
    all_rational_layers_trainable: bool
    optimizer_rebuild_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_depth": self.old_depth,
            "new_depth": self.new_depth,
            "inserted_indices": self.inserted_indices,
            "inserted_layers": [
                layer.to_dict() for layer in self.inserted_layers
            ],
            "function_audit": (
                None
                if self.function_audit is None
                else self.function_audit.to_dict()
            ),
            "all_rational_layers_trainable": (
                self.all_rational_layers_trainable
            ),
            "optimizer_rebuild_required": self.optimizer_rebuild_required,
        }


@torch.no_grad()
def deepen_network_(
    network: RationalNetwork,
    *,
    target_depth: int,
    initialization: IdentityInitialization | None = None,
    reference_h: torch.Tensor | None = None,
    atol: float = 1e-7,
    rtol: float = 1e-6,
) -> DepthExtensionReport:
    """Insert exact identity transitions immediately before class evidence.

    The final rational layer is retained as the class-evidence layer, and all
    old and inserted rational-layer parameters are left trainable.  Because an
    optimizer created before insertion cannot contain the new parameters, the
    caller must construct the next-stage optimizer from the deepened model.
    """

    if not isinstance(network, RationalNetwork):
        raise TypeError("network must be a RationalNetwork.")
    check_positive_int("target_depth", target_depth)
    check_nonnegative_float("atol", atol)
    check_nonnegative_float("rtol", rtol)
    if target_depth <= network.depth:
        raise ValueError(
            "target_depth must be greater than the current network depth."
        )
    if initialization is None:
        initialization = IdentityInitialization()
    if not isinstance(initialization, IdentityInitialization):
        raise TypeError("initialization must be an IdentityInitialization.")
    if reference_h is not None:
        network.validate_input(reference_h)

    old_depth = network.depth
    old_layers = tuple(network.layers)
    old_layer_prefixes = tuple(layer.response_prefix for layer in old_layers)
    old_output_prefix = network.output_layer.response_prefix
    old_audit = getattr(network, "initialization_audit", None)
    old_requires_grad = {
        id(parameter): parameter.requires_grad
        for layer in old_layers
        for parameter in layer.parameters()
    }
    was_training = network.training

    before = (
        _detached_terminal_outputs(network, reference_h)
        if reference_h is not None
        else None
    )

    terminal = old_layers[-1]
    insertion_width = terminal.in_features
    inserted_count = target_depth - old_depth
    first_inserted_index = old_depth - 1
    inserted_indices = tuple(
        range(first_inserted_index, first_inserted_index + inserted_count)
    )

    inserted_layers: list[DenseRationalLayer] = []
    inserted_audits: list[LayerInitializationAudit] = []
    for index in inserted_indices:
        layer = _make_identity_transition(
            network=network,
            template=terminal,
            width=insertion_width,
            index=index,
            initialization=initialization,
        )
        inserted_layers.append(layer)
        inserted_audits.append(
            initialize_identity_layer_(
                layer,
                initialization,
                index=index,
                followed_by_output=False,
            )
        )

    new_layers = (
        old_layers[:-1]
        + tuple(inserted_layers)
        + (terminal,)
    )

    try:
        network.layers = nn.ModuleList(new_layers)
        _renumber_response_prefixes_(network)
        for layer in network.layers:
            layer.requires_grad_(True)
        network.initialization_audit = None

        after = (
            _detached_terminal_outputs(network, reference_h)
            if reference_h is not None
            else None
        )
        function_audit = (
            _function_preservation_audit(
                before,
                after,
                atol=atol,
                rtol=rtol,
            )
            if before is not None and after is not None
            else None
        )
        if function_audit is not None and not function_audit.passed:
            raise RuntimeError(
                "Identity insertion failed the function-preservation audit: "
                f"{function_audit.to_dict()}."
            )
    except Exception:
        network.layers = nn.ModuleList(old_layers)
        for layer, prefix in zip(network.layers, old_layer_prefixes):
            layer.response_prefix = prefix
            for parameter in layer.parameters():
                parameter.requires_grad_(
                    old_requires_grad[id(parameter)]
                )
        network.output_layer.response_prefix = old_output_prefix
        network.initialization_audit = old_audit
        network.train(was_training)
        raise

    network.train(was_training)
    all_trainable = all(
        parameter.requires_grad
        for layer in network.layers
        for parameter in layer.parameters()
    )
    return DepthExtensionReport(
        old_depth=old_depth,
        new_depth=network.depth,
        inserted_indices=inserted_indices,
        inserted_layers=tuple(inserted_audits),
        function_audit=function_audit,
        all_rational_layers_trainable=all_trainable,
    )


def _make_identity_transition(
    *,
    network: RationalNetwork,
    template: DenseRationalLayer,
    width: int,
    index: int,
    initialization: IdentityInitialization,
) -> DenseRationalLayer:
    prototype = template.evidence.raw_positive_weights()
    response_eps = template.eta.eps
    if initialization.response_rate <= response_eps:
        raise ValueError(
            "Identity response_rate must be greater than the inherited "
            "response eps."
        )
    if initialization.chi <= response_eps:
        raise ValueError(
            "Identity chi must be greater than the inherited response eps."
        )
    return DenseRationalLayer(
        in_features=width,
        out_features=width,
        init_rate=0.0,
        weight_parameterization="exact_nonnegative",
        eta_init=initialization.response_rate,
        gamma_init=initialization.response_rate,
        chi_init=initialization.chi,
        evidence_eps=0.0,
        response_eps=response_eps,
        denom_eps=template.denom_eps,
        name=f"{network.name}_continuation{index}",
        response_prefix=f"R{index}",
        check_nonnegative_input=template.check_nonnegative_input,
        include_bias=template.include_bias,
        dtype=prototype.dtype,
        device=prototype.device,
    )


def _renumber_response_prefixes_(network: RationalNetwork) -> None:
    for index, layer in enumerate(network.layers):
        layer.response_prefix = f"R{index}"
    network.output_layer.response_prefix = f"R{network.depth - 1}"


def _detached_terminal_outputs(
    network: RationalNetwork,
    reference_h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    was_training = network.training
    network.eval()
    try:
        state = network(reference_h)
    finally:
        network.train(was_training)
    return (
        state.final_rational_state.r.detach().clone(),
        state.Z.detach().clone(),
        state.Z_free.detach().clone(),
    )


def _function_preservation_audit(
    before: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    after: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> FunctionPreservationAudit:
    r_error = _max_abs_error(before[0], after[0])
    z_error = _max_abs_error(before[1], after[1])
    z_free_error = _max_abs_error(before[2], after[2])
    passed = all(
        torch.allclose(old, new, atol=atol, rtol=rtol)
        for old, new in zip(before, after)
    )
    return FunctionPreservationAudit(
        max_abs_r_error=r_error,
        max_abs_Z_error=z_error,
        max_abs_Z_free_error=z_free_error,
        passed=passed,
        atol=atol,
        rtol=rtol,
    )


def _max_abs_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).item())


__all__ = [
    "DepthExtensionReport",
    "FunctionPreservationAudit",
    "deepen_network_",
]
