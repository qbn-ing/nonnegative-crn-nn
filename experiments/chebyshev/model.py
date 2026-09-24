from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import torch
import torch.nn as nn

from experiments.common.model_lifecycle import (
    DenseDepthExtensionReport,
    FunctionChangeAudit,
    InitialFunctionAudit,
    boundary_parameter_fingerprint as _boundary_fingerprint,
    compare_initial_function as _compare_initial_function,
    extend_with_paired_dense_,
    identity_initialization as _identity_initialization,
    initialize_rational_network_,
)
from models.builders import DenseRationalConfig, build_rational_network
from models.composition.rational_network import RationalNetwork
from models.initialization import IdentityInitialization, InitializationAudit

from .data import encode_inputs, encoded_input_dim
from .protocol import RunSpec


ChebyshevDepthExtensionReport = DenseDepthExtensionReport


class EncodedRationalClassifier(nn.Module):
    """Raw Chebyshev coordinates, task encoding, rational network, then O."""

    def __init__(self, spec: RunSpec, network: RationalNetwork) -> None:
        super().__init__()
        self.task = spec.task
        self.network = network

    @property
    def depth(self) -> int:
        return self.network.depth

    def forward(self, x_raw: torch.Tensor):
        return self.network(encode_inputs(x_raw, self.task))

    def structure_info(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "raw_input_dim": 2,
            "input_encoding": self.task.input_encoding,
            "chebyshev_degree": self.task.degree,
            "depth": self.depth,
            "network": self.network.structure_info(),
        }

    def export_model_spec(self):
        return self.network.export_model_spec()


@dataclass(frozen=True)
class ModelBuild:
    model: EncodedRationalClassifier
    model_seed: int
    initialization_audit: InitializationAudit
    boundary_parameter_fingerprint: str
    initial_function_audit: InitialFunctionAudit | None


def build_model(
    spec: RunSpec,
    *,
    reference_raw: torch.Tensor | None = None,
    deterministic: bool = True,
) -> ModelBuild:
    """Build semantically paired boundary layers for all four I-by-C cells."""

    del deterministic
    identity_hidden = spec.condition == "identity_full_start"
    model, audit = _build_initialized(
        spec,
        depth=spec.initial_depth,
        identity_hidden=identity_hidden,
    )
    function_audit: InitialFunctionAudit | None = None
    if identity_hidden:
        shallow, _ = _build_initialized(spec, depth=2, identity_hidden=False)
        if reference_raw is None:
            reference_raw = torch.linspace(0.0, 1.0, 16).reshape(8, 2)
        function_audit = compare_initial_function(shallow, model, reference_raw)
        if not function_audit.passed:
            raise RuntimeError(
                "identity full-start failed the paired D2 function audit: "
                f"{function_audit.to_dict()}."
            )
    return ModelBuild(
        model=model,
        model_seed=spec.seed,
        initialization_audit=audit,
        boundary_parameter_fingerprint=boundary_parameter_fingerprint(model),
        initial_function_audit=function_audit,
    )


def identity_initialization(spec: RunSpec) -> IdentityInitialization:
    return _identity_initialization(spec.xavier_gain)


def depth_extension_factory(
    spec: RunSpec,
    *,
    deterministic: bool = True,
) -> Callable[
    [RationalNetwork, int, torch.Tensor | None],
    DenseDepthExtensionReport,
] | None:
    del deterministic
    if spec.condition != "dense_continuation":
        return None

    def extend(
        network: RationalNetwork,
        target_depth: int,
        reference_h: torch.Tensor | None,
    ) -> DenseDepthExtensionReport:
        return extend_dense_continuation_model_(
            network,
            spec,
            target_depth=target_depth,
            reference_h=reference_h,
        )

    return extend


@torch.no_grad()
def extend_dense_continuation_model_(
    network: RationalNetwork,
    spec: RunSpec,
    *,
    target_depth: int,
    reference_h: torch.Tensor | None,
    deterministic: bool = True,
    atol: float = 1e-7,
    rtol: float = 1e-6,
) -> DenseDepthExtensionReport:
    del deterministic
    if spec.condition != "dense_continuation":
        raise ValueError("dense extension requires dense_continuation.")
    paired_spec = replace(spec, condition="dense_full_start")

    def paired_factory(
        depth: int,
    ) -> tuple[RationalNetwork, InitializationAudit]:
        paired_model, paired_audit = _build_initialized(
            paired_spec,
            depth=depth,
            identity_hidden=False,
        )
        return paired_model.network, paired_audit

    return extend_with_paired_dense_(
        network,
        target_depth=target_depth,
        reference_h=reference_h,
        identity=identity_initialization(spec),
        paired_network_factory=paired_factory,
        atol=atol,
        rtol=rtol,
    )


def compare_initial_function(
    shallow: EncodedRationalClassifier,
    deep: EncodedRationalClassifier,
    reference_raw: torch.Tensor,
    *,
    atol: float = 1e-7,
    rtol: float = 1e-6,
) -> InitialFunctionAudit:
    return _compare_initial_function(
        shallow,
        deep,
        reference_raw,
        atol=atol,
        rtol=rtol,
    )


def boundary_parameter_fingerprint(model: EncodedRationalClassifier) -> str:
    return _boundary_fingerprint(model.network)


def _build_initialized(
    spec: RunSpec,
    *,
    depth: int,
    identity_hidden: bool,
) -> tuple[EncodedRationalClassifier, InitializationAudit]:
    config = DenseRationalConfig(
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
    widths = (2,) if depth == 1 else (spec.width,) * (depth - 1) + (2,)
    network = build_rational_network(
        input_dim=encoded_input_dim(spec.task),
        widths=widths,
        config=config,
        initialization=None,
        name=f"chebyshev_{spec.task.name}",
        layer_name_prefix="rational",
        check_finite_input=True,
        check_nonnegative_input=True,
    )
    audit = initialize_rational_network_(
        network,
        model_seed=spec.seed,
        xavier_gain=spec.xavier_gain,
        identity_hidden=identity_hidden,
    )
    return EncodedRationalClassifier(spec, network), audit


__all__ = [
    "ChebyshevDepthExtensionReport",
    "DenseDepthExtensionReport",
    "EncodedRationalClassifier",
    "FunctionChangeAudit",
    "InitialFunctionAudit",
    "ModelBuild",
    "boundary_parameter_fingerprint",
    "build_model",
    "compare_initial_function",
    "depth_extension_factory",
    "extend_dense_continuation_model_",
    "identity_initialization",
]
