from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

from models.composition.rational_network import RationalNetwork
from models.depth import deepen_network_
from models.initialization import (
    IdentityInitialization,
    InitializationAudit,
    LayerInitializationAudit,
    NonnegativeXavierUniformInitialization,
    initialize_dense_layer_,
    initialize_identity_layer_,
)


@dataclass(frozen=True)
class InitialFunctionAudit:
    reference_samples: int
    max_abs_r_error: float
    max_abs_Z_error: float
    max_abs_Z_free_error: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class FunctionChangeAudit:
    max_abs_r_error: float
    max_abs_Z_error: float
    max_abs_Z_free_error: float
    allclose: bool
    atol: float
    rtol: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class DenseDepthExtensionReport:
    old_depth: int
    new_depth: int
    inserted_indices: tuple[int, ...]
    inserted_layers: tuple[LayerInitializationAudit, ...]
    insertion_initialization: str
    function_preservation_required: bool
    function_change: FunctionChangeAudit
    all_rational_layers_trainable: bool
    optimizer_rebuild_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_depth": self.old_depth,
            "new_depth": self.new_depth,
            "inserted_indices": self.inserted_indices,
            "inserted_layers": [item.to_dict() for item in self.inserted_layers],
            "insertion_initialization": self.insertion_initialization,
            "function_preservation_required": self.function_preservation_required,
            "function_change": self.function_change.to_dict(),
            "all_rational_layers_trainable": self.all_rational_layers_trainable,
            "optimizer_rebuild_required": self.optimizer_rebuild_required,
        }


def identity_initialization(xavier_gain: float) -> IdentityInitialization:
    return IdentityInitialization(
        dense=NonnegativeXavierUniformInitialization(gain=xavier_gain),
        response_rate=1.0,
        chi=1.0,
    )


def initialize_rational_network_(
    network: RationalNetwork,
    *,
    model_seed: int,
    xavier_gain: float,
    identity_hidden: bool,
) -> InitializationAudit:
    """Initialize paired boundary layers and optional identity hidden layers."""

    dense = NonnegativeXavierUniformInitialization(gain=xavier_gain)
    identity = identity_initialization(xavier_gain)
    terminal_index = network.depth - 1
    audits: list[LayerInitializationAudit] = []
    for index, layer in enumerate(network.layers):
        followed_by_output = index == terminal_index
        if followed_by_output:
            layer_seed = model_seed + 23
        elif index == 0:
            layer_seed = model_seed + 11
        else:
            layer_seed = model_seed + 1_000 + index
        if identity_hidden and index > 0 and not followed_by_output:
            audit = initialize_identity_layer_(
                layer,
                identity,
                index=index,
                followed_by_output=False,
            )
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(layer_seed)
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


@torch.no_grad()
def compare_initial_function(
    shallow: nn.Module,
    deep: nn.Module,
    reference_inputs: torch.Tensor,
    *,
    atol: float = 1e-7,
    rtol: float = 1e-6,
) -> InitialFunctionAudit:
    shallow.eval()
    deep.eval()
    left = _network_state(shallow(reference_inputs))
    right = _network_state(deep(reference_inputs))
    return InitialFunctionAudit(
        reference_samples=int(reference_inputs.shape[0]),
        max_abs_r_error=_max_abs(
            left.final_rational_state.r,
            right.final_rational_state.r,
        ),
        max_abs_Z_error=_max_abs(left.Z, right.Z),
        max_abs_Z_free_error=_max_abs(left.Z_free, right.Z_free),
        passed=bool(
            torch.allclose(
                left.final_rational_state.r,
                right.final_rational_state.r,
                atol=atol,
                rtol=rtol,
            )
            and torch.allclose(left.Z, right.Z, atol=atol, rtol=rtol)
            and torch.allclose(left.Z_free, right.Z_free, atol=atol, rtol=rtol)
        ),
    )


@torch.no_grad()
def boundary_parameter_fingerprint(network: RationalNetwork) -> str:
    digest = hashlib.sha256()
    for layer in (network.layers[0], network.layers[-1]):
        for tensor in (
            layer.evidence.raw_positive_weights(),
            layer.evidence.raw_negative_weights(),
            layer.evidence.raw_positive_bias(),
            layer.evidence.raw_negative_bias(),
            layer.eta(),
            layer.gamma(),
            layer.chi(),
        ):
            value = tensor.detach().cpu().contiguous()
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def extend_with_paired_dense_(
    network: RationalNetwork,
    *,
    target_depth: int,
    reference_h: torch.Tensor | None,
    identity: IdentityInitialization,
    paired_network_factory: Callable[
        [int], tuple[RationalNetwork, InitializationAudit]
    ],
    atol: float = 1e-7,
    rtol: float = 1e-6,
) -> DenseDepthExtensionReport:
    """Insert layers once, then fill them from the paired dense full-start model."""

    before = _detached_outputs(network, reference_h)
    identity_report = deepen_network_(
        network,
        target_depth=target_depth,
        initialization=identity,
        reference_h=reference_h,
        atol=atol,
        rtol=rtol,
    )
    with torch.random.fork_rng(devices=[]):
        paired, paired_audit = paired_network_factory(target_depth)
    if paired.depth != target_depth:
        raise RuntimeError("paired network factory returned the wrong depth.")
    dense_audits: list[LayerInitializationAudit] = []
    for index in identity_report.inserted_indices:
        network.layers[index].load_state_dict(
            paired.layers[index].state_dict(),
            strict=True,
        )
        dense_audits.append(paired_audit.layers[index])
    after = _detached_outputs(network, reference_h)
    return DenseDepthExtensionReport(
        old_depth=identity_report.old_depth,
        new_depth=identity_report.new_depth,
        inserted_indices=identity_report.inserted_indices,
        inserted_layers=tuple(dense_audits),
        insertion_initialization="paired_nonnegative_xavier_uniform",
        function_preservation_required=False,
        function_change=_function_change(before, after, atol=atol, rtol=rtol),
        all_rational_layers_trainable=all(
            parameter.requires_grad
            for layer in network.layers
            for parameter in layer.parameters()
        ),
    )


@torch.no_grad()
def _detached_outputs(
    network: RationalNetwork,
    reference_h: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if reference_h is None:
        return None
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


def _function_change(
    before: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    after: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    *,
    atol: float,
    rtol: float,
) -> FunctionChangeAudit:
    if before is None or after is None:
        return FunctionChangeAudit(0.0, 0.0, 0.0, False, atol, rtol)
    errors = [_max_abs(left, right) for left, right in zip(before, after)]
    return FunctionChangeAudit(
        max_abs_r_error=errors[0],
        max_abs_Z_error=errors[1],
        max_abs_Z_free_error=errors[2],
        allclose=all(
            torch.allclose(left, right, atol=atol, rtol=rtol)
            for left, right in zip(before, after)
        ),
        atol=atol,
        rtol=rtol,
    )


def _network_state(state: Any) -> Any:
    nested = getattr(state, "network", None)
    return state if nested is None else nested


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).item())


__all__ = [
    "DenseDepthExtensionReport",
    "FunctionChangeAudit",
    "InitialFunctionAudit",
    "boundary_parameter_fingerprint",
    "compare_initial_function",
    "extend_with_paired_dense_",
    "identity_initialization",
    "initialize_rational_network_",
]
