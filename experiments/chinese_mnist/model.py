from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import torch

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

from .protocol import RunSpec


@dataclass(frozen=True)
class ModelBuild:
    network: RationalNetwork
    model_seed: int
    initialization_audit: InitializationAudit
    boundary_parameter_fingerprint: str
    initial_function_audit: InitialFunctionAudit | None

    @property
    def model(self) -> RationalNetwork:
        return self.network

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "model_seed": self.model_seed,
            "input_dim": self.network.input_dim,
            "depth": self.network.depth,
            "widths": self.network.widths,
            "n_classes": self.network.n_classes,
            "include_trainable_affine_bias": all(
                layer.include_bias for layer in self.network.layers
            ),
            "weight_parameterization": tuple(
                layer.evidence.weight_parameterization
                for layer in self.network.layers
            ),
            "boundary_parameter_fingerprint": self.boundary_parameter_fingerprint,
            "initialization": self.initialization_audit.to_dict(),
            "initial_function_audit": (
                None
                if self.initial_function_audit is None
                else self.initial_function_audit.to_dict()
            ),
        }


def build_model(
    spec: RunSpec,
    *,
    input_dim: int = 784,
    reference_h: torch.Tensor | None = None,
) -> ModelBuild:
    """Build paired boundary layers for the four I-by-C cells."""

    identity_hidden = spec.condition == "identity_full_start"
    network, audit = _build_initialized(
        spec,
        depth=spec.initial_depth,
        input_dim=input_dim,
        identity_hidden=identity_hidden,
    )
    function_audit: InitialFunctionAudit | None = None
    if identity_hidden:
        shallow, _ = _build_initialized(
            spec,
            depth=2,
            input_dim=input_dim,
            identity_hidden=False,
        )
        if reference_h is None:
            reference_h = torch.linspace(0.0, 1.0, 8 * input_dim).reshape(
                8, input_dim
            )
        function_audit = compare_initial_function(shallow, network, reference_h)
        if not function_audit.passed:
            raise RuntimeError(
                "identity full-start failed the paired D2 function audit: "
                f"{function_audit.to_dict()}"
            )
    return ModelBuild(
        network=network,
        model_seed=spec.model_seed,
        initialization_audit=audit,
        boundary_parameter_fingerprint=boundary_parameter_fingerprint(network),
        initial_function_audit=function_audit,
    )


def identity_initialization(spec: RunSpec) -> IdentityInitialization:
    return _identity_initialization(spec.xavier_gain)


def depth_extension_factory(
    spec: RunSpec,
    *,
    input_dim: int,
) -> Callable[
    [RationalNetwork, int, torch.Tensor | None],
    DenseDepthExtensionReport,
] | None:
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
            input_dim=input_dim,
            target_depth=target_depth,
            reference_h=reference_h,
        )

    return extend


@torch.no_grad()
def extend_dense_continuation_model_(
    network: RationalNetwork,
    spec: RunSpec,
    *,
    input_dim: int,
    target_depth: int,
    reference_h: torch.Tensor | None,
    atol: float = 1e-7,
    rtol: float = 1e-6,
) -> DenseDepthExtensionReport:
    if spec.condition != "dense_continuation":
        raise ValueError("dense extension requires dense_continuation.")
    paired_spec = replace(spec, condition="dense_full_start")

    def paired_factory(
        depth: int,
    ) -> tuple[RationalNetwork, InitializationAudit]:
        return _build_initialized(
            paired_spec,
            depth=depth,
            input_dim=input_dim,
            identity_hidden=False,
        )

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
    shallow: RationalNetwork,
    deep: RationalNetwork,
    reference_h: torch.Tensor,
    *,
    atol: float = 1e-7,
    rtol: float = 1e-6,
) -> InitialFunctionAudit:
    return _compare_initial_function(
        shallow,
        deep,
        reference_h,
        atol=atol,
        rtol=rtol,
    )


def boundary_parameter_fingerprint(network: RationalNetwork) -> str:
    return _boundary_fingerprint(network)


def _build_initialized(
    spec: RunSpec,
    *,
    depth: int,
    input_dim: int,
    identity_hidden: bool,
) -> tuple[RationalNetwork, InitializationAudit]:
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
    widths = (
        (spec.num_classes,)
        if depth == 1
        else (spec.width,) * (depth - 1) + (spec.num_classes,)
    )
    network = build_rational_network(
        input_dim=input_dim,
        widths=widths,
        config=config,
        initialization=None,
        name="chinese_mnist_network",
        layer_name_prefix="rational",
        check_finite_input=True,
        check_nonnegative_input=True,
    )
    audit = initialize_rational_network_(
        network,
        model_seed=spec.model_seed,
        xavier_gain=spec.xavier_gain,
        identity_hidden=identity_hidden,
    )
    return network, audit


__all__ = [
    "DenseDepthExtensionReport",
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
