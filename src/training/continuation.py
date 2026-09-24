from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn

from models.composition.rational_network import RationalNetwork
from models.depth import DepthExtensionReport, deepen_network_
from models.initialization import IdentityInitialization
from utils.validation import check_nonnegative_float, check_positive_int

from .loop import EpochResult, evaluate, train_steps
from .progress import ProgressSetting


OptimizerFactory = Callable[
    [Iterable[nn.Parameter]],
    torch.optim.Optimizer,
]
StageCompletedCallback = Callable[
    ["ContinuationStageResult", nn.Module],
    None,
]


@dataclass(frozen=True)
class DepthStageConfig:
    """One fixed-budget stage of effective-depth continuation."""

    target_depth: int
    optimizer_steps: int

    def __post_init__(self) -> None:
        check_positive_int("target_depth", self.target_depth)
        check_positive_int("optimizer_steps", self.optimizer_steps)

    def to_dict(self) -> dict[str, int]:
        return {
            "target_depth": self.target_depth,
            "optimizer_steps": self.optimizer_steps,
        }


@dataclass(frozen=True)
class DepthContinuationSchedule:
    """Strictly increasing depth stages with an explicit total step budget."""

    stages: tuple[DepthStageConfig, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stages, tuple) or not self.stages:
            raise ValueError("stages must be a non-empty tuple.")
        if not all(
            isinstance(stage, DepthStageConfig)
            for stage in self.stages
        ):
            raise TypeError(
                "stages must contain only DepthStageConfig objects."
            )
        for previous, current in zip(self.stages, self.stages[1:]):
            if current.target_depth <= previous.target_depth:
                raise ValueError(
                    "Stage target depths must be strictly increasing."
                )

    @property
    def total_optimizer_steps(self) -> int:
        return sum(stage.optimizer_steps for stage in self.stages)

    @property
    def target_depths(self) -> tuple[int, ...]:
        return tuple(stage.target_depth for stage in self.stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [stage.to_dict() for stage in self.stages],
            "target_depths": self.target_depths,
            "total_optimizer_steps": self.total_optimizer_steps,
        }

    @classmethod
    def from_sequences(
        cls,
        target_depths: Iterable[int],
        optimizer_steps: Iterable[int],
    ) -> DepthContinuationSchedule:
        depths = tuple(target_depths)
        budgets = tuple(optimizer_steps)
        if len(depths) != len(budgets):
            raise ValueError(
                "target_depths and optimizer_steps must have equal length."
            )
        return cls(
            stages=tuple(
                DepthStageConfig(
                    target_depth=depth,
                    optimizer_steps=steps,
                )
                for depth, steps in zip(depths, budgets)
            )
        )


@dataclass(frozen=True)
class OptimizerStageSchedule:
    """Nondecreasing stages, including matched resets at unchanged depth."""

    stages: tuple[DepthStageConfig, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stages, tuple) or not self.stages:
            raise ValueError("stages must be a non-empty tuple.")
        if not all(isinstance(stage, DepthStageConfig) for stage in self.stages):
            raise TypeError("stages must contain only DepthStageConfig objects.")
        for previous, current in zip(self.stages, self.stages[1:]):
            if current.target_depth < previous.target_depth:
                raise ValueError("Stage target depths must be nondecreasing.")

    @property
    def total_optimizer_steps(self) -> int:
        return sum(stage.optimizer_steps for stage in self.stages)

    @property
    def target_depths(self) -> tuple[int, ...]:
        return tuple(stage.target_depth for stage in self.stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [stage.to_dict() for stage in self.stages],
            "target_depths": self.target_depths,
            "total_optimizer_steps": self.total_optimizer_steps,
            "allows_equal_target_depths": True,
        }


StageSchedule = DepthContinuationSchedule | OptimizerStageSchedule
DepthExtensionFactory = Callable[
    [RationalNetwork, int, torch.Tensor | None],
    Any,
]


@dataclass(frozen=True)
class ContinuationStageResult:
    """Training and audit record for one completed depth stage."""

    index: int
    config: DepthStageConfig
    global_step_start: int
    global_step_end: int
    depth_extension: DepthExtensionReport | None
    optimizer_parameter_count: int
    all_rational_layers_trainable: bool
    train: EpochResult
    validation: EpochResult | None

    @property
    def optimizer_steps_completed(self) -> int:
        return self.global_step_end - self.global_step_start

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "config": self.config.to_dict(),
            "global_step_start": self.global_step_start,
            "global_step_end": self.global_step_end,
            "optimizer_steps_completed": self.optimizer_steps_completed,
            "depth_extension": (
                None
                if self.depth_extension is None
                else self.depth_extension.to_dict()
            ),
            "optimizer_parameter_count": self.optimizer_parameter_count,
            "all_rational_layers_trainable": (
                self.all_rational_layers_trainable
            ),
            "train": _epoch_result_to_dict(self.train),
            "validation": (
                None
                if self.validation is None
                else _epoch_result_to_dict(self.validation)
            ),
        }


@dataclass(frozen=True)
class DepthContinuationResult:
    """Complete fixed-budget record of an effective-depth run."""

    initial_depth: int
    final_depth: int
    schedule: StageSchedule
    stages: tuple[ContinuationStageResult, ...]

    @property
    def total_optimizer_steps_completed(self) -> int:
        return sum(
            stage.optimizer_steps_completed
            for stage in self.stages
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_depth": self.initial_depth,
            "final_depth": self.final_depth,
            "schedule": self.schedule.to_dict(),
            "total_optimizer_steps_completed": (
                self.total_optimizer_steps_completed
            ),
            "stages": [stage.to_dict() for stage in self.stages],
        }


def run_depth_continuation(
    model: nn.Module,
    network: RationalNetwork,
    train_loader: Any,
    *,
    schedule: StageSchedule,
    optimizer_factory: OptimizerFactory,
    val_loader: Any | None = None,
    reference_h: torch.Tensor | None = None,
    identity_initialization: IdentityInitialization | None = None,
    device: torch.device | str | None = None,
    loss_kwargs: dict[str, Any] | None = None,
    max_grad_norm: float | None = None,
    collect_output_diagnostics: bool = True,
    collect_gradient_diagnostics: bool = True,
    collect_classification_metrics: bool = True,
    grad_small_threshold: float = 1e-12,
    grad_large_threshold: float = 1e3,
    function_atol: float = 1e-7,
    function_rtol: float = 1e-6,
    on_stage_completed: StageCompletedCallback | None = None,
    progress: ProgressSetting = "auto",
) -> DepthContinuationResult:
    """Train a shallow model and deepen it through exact identity insertion.

    The first schedule depth must equal the network's current depth.  Every
    later stage inserts identity layers immediately before the terminal class
    evidence layer.  A fresh optimizer is built from all trainable model
    parameters for every stage, so old and newly inserted rational layers are
    always trained jointly.
    """

    if not isinstance(model, nn.Module):
        raise TypeError(
            f"model must be a torch.nn.Module. Got {type(model).__name__}."
        )
    if not isinstance(network, RationalNetwork):
        raise TypeError("network must be a RationalNetwork.")
    if not isinstance(
        schedule,
        (DepthContinuationSchedule, OptimizerStageSchedule),
    ):
        raise TypeError(
            "schedule must be a supported stage schedule."
        )
    if not callable(optimizer_factory):
        raise TypeError("optimizer_factory must be callable.")
    if on_stage_completed is not None and not callable(on_stage_completed):
        raise TypeError("on_stage_completed must be callable or None.")
    if reference_h is not None and not isinstance(reference_h, torch.Tensor):
        raise TypeError("reference_h must be a torch.Tensor or None.")
    if identity_initialization is None:
        identity_initialization = IdentityInitialization()
    if not isinstance(identity_initialization, IdentityInitialization):
        raise TypeError(
            "identity_initialization must be an IdentityInitialization."
        )
    check_nonnegative_float("function_atol", function_atol)
    check_nonnegative_float("function_rtol", function_rtol)

    initial_depth = network.depth
    if schedule.stages[0].target_depth != initial_depth:
        raise ValueError(
            "The first stage target_depth must equal the network's current "
            f"depth ({initial_depth})."
        )
    _validate_network_belongs_to_model(model, network)

    if device is not None:
        model.to(device)
        if reference_h is not None:
            reference_h = reference_h.to(device)
    if reference_h is not None:
        network.validate_input(reference_h)

    global_step = 0
    stage_results: list[ContinuationStageResult] = []

    for index, stage in enumerate(schedule.stages):
        extension = enter_depth_stage_(
            network,
            stage,
            reference_h=reference_h,
            identity_initialization=identity_initialization,
            atol=function_atol,
            rtol=function_rtol,
        )
        _validate_network_belongs_to_model(model, network)

        trainable_parameters = tuple(
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        if not trainable_parameters:
            raise ValueError("model has no trainable parameters.")
        optimizer = optimizer_factory(trainable_parameters)
        _validate_optimizer(
            optimizer,
            expected_parameters=trainable_parameters,
        )

        start = global_step
        train_result = train_steps(
            model,
            train_loader,
            optimizer,
            steps=stage.optimizer_steps,
            device=device,
            loss_kwargs=loss_kwargs,
            max_grad_norm=max_grad_norm,
            collect_output_diagnostics=collect_output_diagnostics,
            collect_gradient_diagnostics=collect_gradient_diagnostics,
            collect_classification_metrics=collect_classification_metrics,
            grad_small_threshold=grad_small_threshold,
            grad_large_threshold=grad_large_threshold,
            progress=progress,
            progress_desc=f"Depth {stage.target_depth}",
        )
        global_step += stage.optimizer_steps

        validation_result = (
            None
            if val_loader is None
            else evaluate(
                model,
                val_loader,
                device=device,
                loss_kwargs=loss_kwargs,
                collect_output_diagnostics=collect_output_diagnostics,
                collect_classification_metrics=(
                    collect_classification_metrics
                ),
                progress=False,
            )
        )
        all_trainable = all(
            parameter.requires_grad
            for layer in network.layers
            for parameter in layer.parameters()
        )
        stage_result = ContinuationStageResult(
            index=index,
            config=stage,
            global_step_start=start,
            global_step_end=global_step,
            depth_extension=extension,
            optimizer_parameter_count=sum(
                parameter.numel()
                for parameter in trainable_parameters
            ),
            all_rational_layers_trainable=all_trainable,
            train=train_result,
            validation=validation_result,
        )
        stage_results.append(stage_result)
        if on_stage_completed is not None:
            on_stage_completed(stage_result, model)

    result = DepthContinuationResult(
        initial_depth=initial_depth,
        final_depth=network.depth,
        schedule=schedule,
        stages=tuple(stage_results),
    )
    if (
        result.total_optimizer_steps_completed
        != schedule.total_optimizer_steps
    ):
        raise RuntimeError(
            "Completed optimizer-step count does not match the schedule."
        )
    return result


def enter_depth_stage_(
    network: RationalNetwork,
    stage: DepthStageConfig,
    *,
    reference_h: torch.Tensor | None = None,
    identity_initialization: IdentityInitialization | None = None,
    depth_extension_factory: DepthExtensionFactory | None = None,
    atol: float = 1e-7,
    rtol: float = 1e-6,
) -> Any | None:
    """Apply the shared stage transition and expose every layer to training."""

    if not isinstance(network, RationalNetwork):
        raise TypeError("network must be a RationalNetwork.")
    if not isinstance(stage, DepthStageConfig):
        raise TypeError("stage must be a DepthStageConfig.")
    if stage.target_depth < network.depth:
        raise RuntimeError("The schedule attempted to reduce network depth.")
    extension: Any | None = None
    if stage.target_depth > network.depth:
        if depth_extension_factory is None:
            extension = deepen_network_(
                network,
                target_depth=stage.target_depth,
                initialization=(
                    identity_initialization or IdentityInitialization()
                ),
                reference_h=reference_h,
                atol=atol,
                rtol=rtol,
            )
        else:
            extension = depth_extension_factory(
                network,
                stage.target_depth,
                reference_h,
            )
        if network.depth != stage.target_depth:
            raise RuntimeError(
                "The depth extension factory did not reach the scheduled depth."
            )
    for layer in network.layers:
        layer.requires_grad_(True)
    return extension


def _validate_network_belongs_to_model(
    model: nn.Module,
    network: RationalNetwork,
) -> None:
    model_parameter_ids = {
        id(parameter)
        for parameter in model.parameters()
    }
    network_parameter_ids = {
        id(parameter)
        for parameter in network.parameters()
    }
    missing = network_parameter_ids - model_parameter_ids
    if missing:
        raise ValueError(
            "network must be the supplied model or a registered submodule of "
            "the supplied model."
        )


def _validate_optimizer(
    optimizer: Any,
    *,
    expected_parameters: tuple[nn.Parameter, ...],
) -> None:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError(
            "optimizer_factory must return a torch.optim.Optimizer."
        )
    expected_ids = {id(parameter) for parameter in expected_parameters}
    actual = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    actual_ids = {id(parameter) for parameter in actual}
    if len(actual) != len(actual_ids):
        raise ValueError(
            "optimizer contains duplicate parameter references."
        )
    if actual_ids != expected_ids:
        missing = len(expected_ids - actual_ids)
        unexpected = len(actual_ids - expected_ids)
        raise ValueError(
            "optimizer must contain every trainable model parameter exactly "
            f"once (missing={missing}, unexpected={unexpected})."
        )


def _epoch_result_to_dict(result: EpochResult) -> dict[str, Any]:
    return {
        "n_batches": result.n_batches,
        "n_samples": result.n_samples,
        "metrics": dict(result.metrics),
    }


__all__ = [
    "ContinuationStageResult",
    "DepthExtensionFactory",
    "DepthContinuationResult",
    "DepthContinuationSchedule",
    "DepthStageConfig",
    "OptimizerStageSchedule",
    "OptimizerFactory",
    "StageSchedule",
    "StageCompletedCallback",
    "enter_depth_stage_",
    "run_depth_continuation",
]
