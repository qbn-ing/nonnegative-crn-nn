from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import torch
import torch.nn as nn

from models.composition.rational_network import RationalNetwork
from models.initialization import IdentityInitialization
from reporting.checkpoint import load_checkpoint
from reporting.run import RunRecorder, start_run
from training.continuation import (
    DepthContinuationSchedule,
    DepthStageConfig,
    OptimizerStageSchedule,
    StageSchedule,
    enter_depth_stage_,
)
from training.loop import (
    CyclingBatchStream,
    EpochResult,
    EvaluationBatchCallback,
    OptimizerStepResult,
    evaluate,
    train_steps,
)
from training.optim import (
    OptimizerParameterGroupAudit,
    build_optimizer_parameter_groups,
)
from training.progress import ProgressSetting
from training.schedule import StepProtocol, TargetMode
from utils.serialization import to_jsonable, write_json
from utils.validation import (
    check_nonnegative_float,
    check_positive_float,
    check_positive_int,
)

from .protocol import (
    CheckpointDecision,
    ExperimentProtocol,
    ValidationCheckpointSelector,
)


SoftTargetFactory = Callable[
    [Any, torch.Tensor, float, int],
    torch.Tensor,
]


class DepthExtensionRecord(Protocol):
    """Serializable audit returned by a task-specific depth extension."""

    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class SerializedDepthExtension:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


DepthExtensionFactory = Callable[
    [RationalNetwork, int, torch.Tensor | None],
    DepthExtensionRecord,
]


@dataclass(frozen=True)
class AdamWProtocolConfig:
    """AdamW settings with explicit policies for every parameter domain."""

    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    positive_weight_decay: float = 0.0
    exact_nonnegative_weight_decay: float = 0.0
    ordinary_weight_decay: float | None = None
    amsgrad: bool = False

    def __post_init__(self) -> None:
        for name in ("beta1", "beta2"):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1).")
        check_positive_float("eps", self.eps)
        check_nonnegative_float(
            "positive_weight_decay",
            self.positive_weight_decay,
        )
        check_nonnegative_float(
            "exact_nonnegative_weight_decay",
            self.exact_nonnegative_weight_decay,
        )
        if self.ordinary_weight_decay is not None:
            check_nonnegative_float(
                "ordinary_weight_decay",
                self.ordinary_weight_decay,
            )
        if not isinstance(self.amsgrad, bool):
            raise TypeError("amsgrad must be a bool.")


@dataclass(frozen=True)
class ProtocolRunnerConfig:
    """Execution controls that do not alter the scientific step protocol."""

    evaluation_interval: int = 100
    max_grad_norm: float | None = 5.0
    collect_output_diagnostics: bool = True
    collect_gradient_diagnostics: bool = True
    collect_classification_metrics: bool = True
    include_packages_in_environment: bool = True
    overwrite_output: bool = False
    resume_from_latest: bool = False
    manage_training_status: bool = True

    def __post_init__(self) -> None:
        check_positive_int("evaluation_interval", self.evaluation_interval)
        if self.max_grad_norm is not None:
            check_nonnegative_float("max_grad_norm", self.max_grad_norm)
        for name in (
            "collect_output_diagnostics",
            "collect_gradient_diagnostics",
            "collect_classification_metrics",
            "include_packages_in_environment",
            "overwrite_output",
            "resume_from_latest",
            "manage_training_status",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool.")
        if self.overwrite_output and self.resume_from_latest:
            raise ValueError(
                "overwrite_output and resume_from_latest are mutually exclusive."
            )


@dataclass(frozen=True)
class ProtocolEvaluationRecord:
    global_step: int
    stage_index: int
    scheduled_depth: int
    actual_depth: int
    target_mode: TargetMode
    beta: float | None
    learning_rate: float
    validation: EpochResult
    checkpoint: CheckpointDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_step": self.global_step,
            "stage_index": self.stage_index,
            "scheduled_depth": self.scheduled_depth,
            "actual_depth": self.actual_depth,
            "target_mode": self.target_mode,
            "beta": self.beta,
            "learning_rate": self.learning_rate,
            "validation": _epoch_result_to_dict(self.validation),
            "checkpoint": asdict(self.checkpoint),
        }


@dataclass(frozen=True)
class ProtocolStageResult:
    index: int
    target_depth: int
    global_step_start: int
    global_step_end: int
    depth_extension: DepthExtensionRecord | None
    optimizer_groups: OptimizerParameterGroupAudit
    all_rational_layers_trainable: bool
    train: EpochResult
    validation: EpochResult

    @property
    def optimizer_steps_completed(self) -> int:
        return self.global_step_end - self.global_step_start

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "target_depth": self.target_depth,
            "global_step_start": self.global_step_start,
            "global_step_end": self.global_step_end,
            "optimizer_steps_completed": self.optimizer_steps_completed,
            "depth_extension": (
                None
                if self.depth_extension is None
                else self.depth_extension.to_dict()
            ),
            "optimizer_groups": self.optimizer_groups.to_dict(),
            "all_rational_layers_trainable": (
                self.all_rational_layers_trainable
            ),
            "train": _epoch_result_to_dict(self.train),
            "validation": _epoch_result_to_dict(self.validation),
        }


@dataclass(frozen=True)
class ProtocolRunResult:
    initial_depth: int
    final_depth: int
    total_optimizer_steps: int
    best_global_step: int
    best_validation_loss: float
    best_checkpoint: str
    stages: tuple[ProtocolStageResult, ...]
    evaluations: tuple[ProtocolEvaluationRecord, ...]
    selected_validation: EpochResult
    test: EpochResult
    test_evaluation_count: int
    diagnostic_global_best_step: int | None = None
    diagnostic_global_best_loss: float | None = None
    diagnostic_global_best_checkpoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_depth": self.initial_depth,
            "final_depth": self.final_depth,
            "total_optimizer_steps": self.total_optimizer_steps,
            "best_global_step": self.best_global_step,
            "best_validation_loss": self.best_validation_loss,
            "best_checkpoint": self.best_checkpoint,
            "stages": [stage.to_dict() for stage in self.stages],
            "evaluations": [item.to_dict() for item in self.evaluations],
            "selected_validation": _epoch_result_to_dict(
                self.selected_validation
            ),
            "test": _epoch_result_to_dict(self.test),
            "test_evaluation_count": self.test_evaluation_count,
            "diagnostic_global_best_step": self.diagnostic_global_best_step,
            "diagnostic_global_best_loss": self.diagnostic_global_best_loss,
            "diagnostic_global_best_checkpoint": (
                self.diagnostic_global_best_checkpoint
            ),
        }


def run_training_protocol(
    model: nn.Module,
    network: RationalNetwork,
    train_loader: Any,
    val_loader: Any,
    test_loader: Any,
    *,
    output_dir: str | Path,
    step_protocol: StepProtocol,
    continuation_schedule: StageSchedule,
    soft_target_factory: SoftTargetFactory,
    experiment_protocol: ExperimentProtocol | None = None,
    optimizer_config: AdamWProtocolConfig | None = None,
    runner_config: ProtocolRunnerConfig | None = None,
    reference_h: torch.Tensor | None = None,
    identity_initialization: IdentityInitialization | None = None,
    depth_extension_factory: DepthExtensionFactory | None = None,
    depth_extension_strategy: str = "identity",
    device: torch.device | str | None = None,
    loss_kwargs: Mapping[str, Any] | None = None,
    project_root: str | Path | None = None,
    seed_report: Mapping[str, Any] | None = None,
    checkpoint_generators: Mapping[str, torch.Generator] | None = None,
    on_test_batch_completed: EvaluationBatchCallback | None = None,
    progress: ProgressSetting = "auto",
) -> ProtocolRunResult:
    """Execute the fixed-budget paper training and selection protocol.

    The runner keeps minibatch order continuous across depth stages, applies a
    single global warm-up/cosine learning-rate schedule, delegates task-specific
    soft-target construction to ``soft_target_factory``, rebuilds AdamW after
    each optimizer stage, selects only final-depth post-curriculum
    checkpoints, restores the selected model, and evaluates the test set once.
    """

    _validate_runner_inputs(
        model=model,
        network=network,
        step_protocol=step_protocol,
        continuation_schedule=continuation_schedule,
        soft_target_factory=soft_target_factory,
        reference_h=reference_h,
        depth_extension_factory=depth_extension_factory,
        depth_extension_strategy=depth_extension_strategy,
    )
    optimizer_config = optimizer_config or AdamWProtocolConfig()
    runner_config = runner_config or ProtocolRunnerConfig()
    experiment_protocol = experiment_protocol or ExperimentProtocol(
        minimum_selection_step=step_protocol.soft_steps,
    )
    _validate_selection_protocol(
        experiment_protocol,
        step_protocol=step_protocol,
    )
    _validate_stage_boundary(
        continuation_schedule,
        step_protocol=step_protocol,
    )

    resolved_loss_kwargs = dict(loss_kwargs or {})
    initial_depth = network.depth
    final_depth = continuation_schedule.stages[-1].target_depth
    runner_payload = asdict(runner_config)
    for execution_key in (
        "overwrite_output",
        "resume_from_latest",
        "manage_training_status",
    ):
        runner_payload.pop(execution_key)
    resolved_config = {
        "step_protocol": step_protocol.to_dict(),
        "experiment_protocol": experiment_protocol.to_dict(),
        "continuation_schedule": continuation_schedule.to_dict(),
        "optimizer": asdict(optimizer_config),
        "runner": runner_payload,
        "initial_depth": initial_depth,
        "final_depth": final_depth,
        "depth_extension_strategy": depth_extension_strategy,
    }
    recorder = start_run(
        output_dir,
        config=resolved_config,
        project_root=project_root,
        seed_report=seed_report,
        include_packages=runner_config.include_packages_in_environment,
        overwrite=runner_config.overwrite_output,
        resume=runner_config.resume_from_latest,
        manage_status=runner_config.manage_training_status,
    )

    try:
        if device is not None:
            model.to(device)
            if reference_h is not None:
                reference_h = reference_h.to(device)
        if reference_h is not None:
            network.validate_input(reference_h)

        selector = ValidationCheckpointSelector(experiment_protocol)
        evaluations: list[ProtocolEvaluationRecord] = []
        stages: list[ProtocolStageResult] = []
        evaluated_positions: set[tuple[int, int]] = set()
        best_checkpoint: Path | None = None
        diagnostic_global_best_checkpoint: Path | None = None
        diagnostic_global_best_step: int | None = None
        diagnostic_global_best_loss: float | None = None
        global_step = 0
        current_learning_rate = step_protocol.learning_rate(0)
        start_stage_index = 0
        resume_path = recorder.output_dir / "protocol_resume_state.json"
        latest_path = recorder.output_dir / "latest_stage_checkpoint.pt"

        if runner_config.resume_from_latest:
            resume_state = _read_resume_state(resume_path)
            completed_stage_index = int(resume_state["completed_stage_index"])
            if not 0 <= completed_stage_index < len(continuation_schedule.stages):
                raise ValueError("resume stage index is outside the schedule.")
            for stage in continuation_schedule.stages[: completed_stage_index + 1]:
                enter_depth_stage_(
                    network,
                    stage,
                    reference_h=reference_h,
                    identity_initialization=identity_initialization,
                    depth_extension_factory=depth_extension_factory,
                )
            if not latest_path.is_file():
                raise FileNotFoundError(
                    "resume requires latest_stage_checkpoint.pt."
                )
            load_checkpoint(latest_path, model, map_location=device or "cpu")
            global_step = int(resume_state["global_step"])
            expected_step = sum(
                stage.optimizer_steps
                for stage in continuation_schedule.stages[
                    : completed_stage_index + 1
                ]
            )
            if global_step != expected_step:
                raise ValueError("resume global step does not match the schedule.")
            evaluations = [
                _evaluation_from_dict(item)
                for item in resume_state.get("evaluations", [])
            ]
            stages = [
                _stage_from_dict(item)
                for item in resume_state.get("stages", [])
            ]
            if len(stages) != completed_stage_index + 1:
                raise ValueError("resume stage history is incomplete.")
            evaluated_positions = {
                (record.global_step, record.actual_depth)
                for record in evaluations
            }
            selector.best_loss = _optional_float(resume_state.get("best_loss"))
            selector.best_step = _optional_int(resume_state.get("best_step"))
            best_checkpoint = _optional_path(resume_state.get("best_checkpoint"))
            diagnostic_global_best_loss = _optional_float(
                resume_state.get("diagnostic_global_best_loss")
            )
            diagnostic_global_best_step = _optional_int(
                resume_state.get("diagnostic_global_best_step")
            )
            diagnostic_global_best_checkpoint = _optional_path(
                resume_state.get("diagnostic_global_best_checkpoint")
            )
            start_stage_index = completed_stage_index + 1
            current_learning_rate = step_protocol.learning_rate(
                min(global_step, step_protocol.total_steps - 1)
            )

        batch_stream = CyclingBatchStream(train_loader)
        if runner_config.resume_from_latest:
            data_state = resume_state.get("data_state")
            if not isinstance(data_state, dict):
                raise ValueError("resume data_state is missing.")
            yielded = int(data_state.get("batches_yielded", -1))
            if yielded < 0:
                raise ValueError("resume batches_yielded is invalid.")
            for _ in range(yielded):
                next(batch_stream)
            if batch_stream.cycles_completed != int(
                data_state.get("cycles_completed", -1)
            ):
                raise ValueError("reconstructed loader cycle count is inconsistent.")
            load_checkpoint(
                latest_path,
                model,
                map_location=device or "cpu",
                restore_rng=True,
                strict_rng=False,
            )

        def evaluate_position(
            *,
            stage_index: int,
            scheduled_depth: int,
            optimizer: torch.optim.Optimizer,
            force: bool = False,
        ) -> ProtocolEvaluationRecord:
            nonlocal best_checkpoint
            nonlocal diagnostic_global_best_checkpoint
            nonlocal diagnostic_global_best_step
            nonlocal diagnostic_global_best_loss
            position = (global_step, network.depth)
            if not force and position in evaluated_positions:
                return _find_evaluation(evaluations, position)
            validation = evaluate(
                model,
                val_loader,
                device=device,
                loss_kwargs=resolved_loss_kwargs,
                collect_output_diagnostics=(
                    runner_config.collect_output_diagnostics
                ),
                collect_classification_metrics=(
                    runner_config.collect_classification_metrics
                ),
                progress=False,
            )
            target_mode, beta = step_protocol.curriculum(
                min(global_step, step_protocol.total_steps - 1)
            )
            decision = selector.consider(
                global_step=global_step,
                validation_loss=validation.classification,
                at_final_depth=network.depth == final_depth,
            )
            record = ProtocolEvaluationRecord(
                global_step=global_step,
                stage_index=stage_index,
                scheduled_depth=scheduled_depth,
                actual_depth=network.depth,
                target_mode=target_mode,
                beta=beta,
                learning_rate=current_learning_rate,
                validation=validation,
                checkpoint=decision,
            )
            evaluations.append(record)
            evaluated_positions.add(position)
            recorder.log_event("validation_completed", record.to_dict())
            if (
                diagnostic_global_best_loss is None
                or validation.classification < diagnostic_global_best_loss
            ):
                diagnostic_global_best_loss = float(validation.classification)
                diagnostic_global_best_step = global_step
                diagnostic_global_best_checkpoint = recorder.save_checkpoint(
                    model,
                    filename="diagnostic_global_best_checkpoint.pt",
                    optimizer=optimizer,
                    global_step=global_step,
                    config=resolved_config,
                    extra={
                        "diagnostic_only": True,
                        "stage_index": stage_index,
                        "actual_depth": network.depth,
                        "validation": _epoch_result_to_dict(validation),
                    },
                    generators=checkpoint_generators,
                )
            if decision.selected:
                best_checkpoint = recorder.save_checkpoint(
                    model,
                    filename="best_checkpoint.pt",
                    optimizer=optimizer,
                    global_step=global_step,
                    config=resolved_config,
                    extra={
                        "stage_index": stage_index,
                        "actual_depth": network.depth,
                        "validation": _epoch_result_to_dict(validation),
                        "selection_metric": "classification",
                    },
                    generators=checkpoint_generators,
                )
            model.train()
            return record

        for stage_index, stage in enumerate(continuation_schedule.stages):
            if stage_index < start_stage_index:
                continue
            extension: DepthExtensionRecord | None = enter_depth_stage_(
                network,
                stage,
                reference_h=reference_h,
                identity_initialization=identity_initialization,
                depth_extension_factory=depth_extension_factory,
            )

            parameter_groups, group_audit = (
                build_optimizer_parameter_groups(
                    model,
                    positive_weight_decay=(
                        optimizer_config.positive_weight_decay
                    ),
                    exact_nonnegative_weight_decay=(
                        optimizer_config.exact_nonnegative_weight_decay
                    ),
                    ordinary_weight_decay=(
                        optimizer_config.ordinary_weight_decay
                    ),
                )
            )
            optimizer = torch.optim.AdamW(
                parameter_groups,
                lr=step_protocol.peak_lr,
                betas=(optimizer_config.beta1, optimizer_config.beta2),
                eps=optimizer_config.eps,
                amsgrad=optimizer_config.amsgrad,
            )
            stage_start = global_step
            stage_end = stage_start + stage.optimizer_steps

            if extension is not None or stage_index == 0:
                evaluate_position(
                    stage_index=stage_index,
                    scheduled_depth=stage.target_depth,
                    optimizer=optimizer,
                )

            def on_step_started(
                zero_based_step: int,
                active_optimizer: torch.optim.Optimizer,
            ) -> None:
                nonlocal current_learning_rate
                current_learning_rate = step_protocol.learning_rate(
                    zero_based_step
                )
                for group in active_optimizer.param_groups:
                    group["lr"] = current_learning_rate

            def transform_target(
                inputs: Any,
                hard_target: torch.Tensor,
                zero_based_step: int,
            ) -> torch.Tensor:
                mode, beta = step_protocol.curriculum(zero_based_step)
                if mode == "hard":
                    return hard_target
                assert beta is not None
                return soft_target_factory(
                    inputs,
                    hard_target,
                    float(beta),
                    zero_based_step,
                )

            latest_step_result: OptimizerStepResult | None = None

            def on_step_completed(
                step_result: OptimizerStepResult,
                active_model: nn.Module,
                active_optimizer: torch.optim.Optimizer,
            ) -> None:
                nonlocal global_step, latest_step_result
                if active_model is not model:
                    raise RuntimeError("Step callback received another model.")
                global_step = step_result.global_step
                latest_step_result = step_result
                should_evaluate = (
                    global_step % runner_config.evaluation_interval == 0
                    or global_step == step_protocol.soft_steps
                    or global_step == stage_end
                    or global_step == step_protocol.total_steps
                )
                if should_evaluate:
                    evaluate_position(
                        stage_index=stage_index,
                        scheduled_depth=stage.target_depth,
                        optimizer=active_optimizer,
                    )

            train_result = train_steps(
                model,
                batch_stream,
                optimizer,
                steps=stage.optimizer_steps,
                device=device,
                loss_kwargs=resolved_loss_kwargs,
                max_grad_norm=runner_config.max_grad_norm,
                collect_output_diagnostics=(
                    runner_config.collect_output_diagnostics
                ),
                collect_gradient_diagnostics=(
                    runner_config.collect_gradient_diagnostics
                ),
                collect_classification_metrics=(
                    runner_config.collect_classification_metrics
                ),
                progress=progress,
                progress_desc=f"Depth {stage.target_depth}",
                global_step_start=stage_start,
                on_step_started=on_step_started,
                training_target_transform=transform_target,
                on_step_completed=on_step_completed,
            )
            if latest_step_result is None or global_step != stage_end:
                raise RuntimeError("The optimizer stage did not complete.")
            final_evaluation = evaluate_position(
                stage_index=stage_index,
                scheduled_depth=stage.target_depth,
                optimizer=optimizer,
            )
            stage_result = ProtocolStageResult(
                index=stage_index,
                target_depth=stage.target_depth,
                global_step_start=stage_start,
                global_step_end=global_step,
                depth_extension=extension,
                optimizer_groups=group_audit,
                all_rational_layers_trainable=all(
                    parameter.requires_grad
                    for layer in network.layers
                    for parameter in layer.parameters()
                ),
                train=train_result,
                validation=final_evaluation.validation,
            )
            stages.append(stage_result)
            recorder.record_stage(stage_result, model)
            data_state = {
                "batches_yielded": batch_stream.batches_yielded,
                "cycles_completed": batch_stream.cycles_completed,
            }
            recorder.save_checkpoint(
                model,
                filename="latest_stage_checkpoint.pt",
                optimizer=optimizer,
                global_step=global_step,
                config=resolved_config,
                data_state=data_state,
                extra={"stage": stage_result.to_dict()},
                generators=checkpoint_generators,
            )
            write_json(
                {
                    "schema_version": 1,
                    "completed_stage_index": stage_index,
                    "global_step": global_step,
                    "data_state": data_state,
                    "best_loss": selector.best_loss,
                    "best_step": selector.best_step,
                    "best_checkpoint": (
                        None
                        if best_checkpoint is None
                        else best_checkpoint.as_posix()
                    ),
                    "diagnostic_global_best_loss": (
                        diagnostic_global_best_loss
                    ),
                    "diagnostic_global_best_step": (
                        diagnostic_global_best_step
                    ),
                    "diagnostic_global_best_checkpoint": (
                        None
                        if diagnostic_global_best_checkpoint is None
                        else diagnostic_global_best_checkpoint.as_posix()
                    ),
                    "stages": [item.to_dict() for item in stages],
                    "evaluations": [item.to_dict() for item in evaluations],
                },
                resume_path,
            )

        if global_step != step_protocol.total_steps:
            raise RuntimeError(
                "Completed optimizer-step count does not match the protocol."
            )
        if network.depth != final_depth:
            raise RuntimeError("Training ended before reaching final depth.")
        if best_checkpoint is None or selector.best_step is None:
            raise RuntimeError(
                "No final-depth post-curriculum checkpoint was selected."
            )
        load_checkpoint(best_checkpoint, model, map_location=device or "cpu")
        if device is not None:
            model.to(device)
        selected_validation = evaluate(
            model,
            val_loader,
            device=device,
            loss_kwargs=resolved_loss_kwargs,
            collect_output_diagnostics=(
                runner_config.collect_output_diagnostics
            ),
            collect_classification_metrics=(
                runner_config.collect_classification_metrics
            ),
            progress=False,
        )
        selector.register_test_evaluation()
        test_result = evaluate(
            model,
            test_loader,
            device=device,
            loss_kwargs=resolved_loss_kwargs,
            collect_output_diagnostics=(
                runner_config.collect_output_diagnostics
            ),
            collect_classification_metrics=(
                runner_config.collect_classification_metrics
            ),
            on_batch_completed=on_test_batch_completed,
            progress=False,
        )
        assert selector.best_loss is not None
        result = ProtocolRunResult(
            initial_depth=initial_depth,
            final_depth=network.depth,
            total_optimizer_steps=global_step,
            best_global_step=selector.best_step,
            best_validation_loss=selector.best_loss,
            best_checkpoint=best_checkpoint.as_posix(),
            stages=tuple(stages),
            evaluations=tuple(evaluations),
            selected_validation=selected_validation,
            test=test_result,
            test_evaluation_count=selector.test_evaluations,
            diagnostic_global_best_step=diagnostic_global_best_step,
            diagnostic_global_best_loss=diagnostic_global_best_loss,
            diagnostic_global_best_checkpoint=(
                None
                if diagnostic_global_best_checkpoint is None
                else diagnostic_global_best_checkpoint.as_posix()
            ),
        )
        write_json(
            result.to_dict(),
            recorder.output_dir / "protocol_history.json",
        )
        summary = {
            "status": "completed",
            "selection": {
                "criterion": "minimum hard validation cross-entropy",
                "minimum_step": experiment_protocol.minimum_selection_step,
                "requires_final_depth": True,
                "selected_step": selector.best_step,
                "selected_loss": selector.best_loss,
            },
            "test_evaluation_count": selector.test_evaluations,
            "diagnostic_global_best": {
                "selection_role": "descriptive_only_not_used_for_test",
                "step": diagnostic_global_best_step,
                "loss": diagnostic_global_best_loss,
                "checkpoint": (
                    None
                    if diagnostic_global_best_checkpoint is None
                    else diagnostic_global_best_checkpoint.as_posix()
                ),
            },
            "result": result.to_dict(),
        }
        recorder.save_summary(summary)
        recorder.mark_completed(summary)
        return result
    except BaseException as error:
        recorder.mark_failed(error)
        raise


def _validate_runner_inputs(
    *,
    model: nn.Module,
    network: RationalNetwork,
    step_protocol: StepProtocol,
    continuation_schedule: StageSchedule,
    soft_target_factory: SoftTargetFactory,
    reference_h: torch.Tensor | None,
    depth_extension_factory: DepthExtensionFactory | None,
    depth_extension_strategy: str,
) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not isinstance(network, RationalNetwork):
        raise TypeError("network must be a RationalNetwork.")
    if not isinstance(step_protocol, StepProtocol):
        raise TypeError("step_protocol must be a StepProtocol.")
    if not isinstance(
        continuation_schedule,
        (DepthContinuationSchedule, OptimizerStageSchedule),
    ):
        raise TypeError(
            "continuation_schedule must be a DepthContinuationSchedule or "
            "OptimizerStageSchedule."
        )
    if not callable(soft_target_factory):
        raise TypeError("soft_target_factory must be callable.")
    if reference_h is not None and not isinstance(reference_h, torch.Tensor):
        raise TypeError("reference_h must be a torch.Tensor or None.")
    if depth_extension_factory is not None and not callable(
        depth_extension_factory
    ):
        raise TypeError("depth_extension_factory must be callable or None.")
    if not isinstance(depth_extension_strategy, str) or not (
        depth_extension_strategy.strip()
    ):
        raise ValueError("depth_extension_strategy must be a non-empty string.")
    if continuation_schedule.stages[0].target_depth != network.depth:
        raise ValueError(
            "The first continuation stage must match the current depth."
        )
    if (
        continuation_schedule.total_optimizer_steps
        != step_protocol.total_steps
    ):
        raise ValueError(
            "Continuation and global optimizer-step budgets must match."
        )
    model_parameter_ids = {id(value) for value in model.parameters()}
    network_parameter_ids = {id(value) for value in network.parameters()}
    if not network_parameter_ids.issubset(model_parameter_ids):
        raise ValueError(
            "network must be the model or a registered model submodule."
        )


def _validate_selection_protocol(
    protocol: ExperimentProtocol,
    *,
    step_protocol: StepProtocol,
) -> None:
    if not isinstance(protocol, ExperimentProtocol):
        raise TypeError("experiment_protocol must be an ExperimentProtocol.")
    if not protocol.require_final_depth:
        raise ValueError("The paper protocol requires final-depth selection.")
    if protocol.minimum_selection_step < step_protocol.soft_steps:
        raise ValueError(
            "Checkpoint selection must begin after the soft curriculum."
        )
    if protocol.minimum_selection_step > step_protocol.total_steps:
        raise ValueError("minimum_selection_step exceeds the step budget.")


def _validate_stage_boundary(
    schedule: StageSchedule,
    *,
    step_protocol: StepProtocol,
) -> None:
    if len(schedule.stages) <= 1:
        return
    final_depth_start = sum(
        stage.optimizer_steps for stage in schedule.stages[:-1]
    )
    if final_depth_start != step_protocol.soft_steps:
        raise ValueError(
            "The final optimizer stage must start at the soft-to-hard boundary."
        )


def _find_evaluation(
    evaluations: list[ProtocolEvaluationRecord],
    position: tuple[int, int],
) -> ProtocolEvaluationRecord:
    for record in reversed(evaluations):
        if (record.global_step, record.actual_depth) == position:
            return record
    raise RuntimeError("Evaluation cache is inconsistent.")


def _epoch_result_to_dict(result: EpochResult) -> dict[str, Any]:
    return {
        "n_batches": result.n_batches,
        "n_samples": result.n_samples,
        "metrics": to_jsonable(result.metrics),
    }


def _epoch_result_from_dict(payload: Mapping[str, Any]) -> EpochResult:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise TypeError("serialized epoch result has no metrics object.")
    return EpochResult(
        n_batches=int(payload["n_batches"]),
        n_samples=int(payload["n_samples"]),
        metrics=dict(metrics),
    )


def _evaluation_from_dict(payload: Mapping[str, Any]) -> ProtocolEvaluationRecord:
    checkpoint = payload.get("checkpoint")
    validation = payload.get("validation")
    if not isinstance(checkpoint, dict) or not isinstance(validation, dict):
        raise TypeError("serialized evaluation record is incomplete.")
    return ProtocolEvaluationRecord(
        global_step=int(payload["global_step"]),
        stage_index=int(payload["stage_index"]),
        scheduled_depth=int(payload["scheduled_depth"]),
        actual_depth=int(payload["actual_depth"]),
        target_mode=str(payload["target_mode"]),  # type: ignore[arg-type]
        beta=_optional_float(payload.get("beta")),
        learning_rate=float(payload["learning_rate"]),
        validation=_epoch_result_from_dict(validation),
        checkpoint=CheckpointDecision(
            selected=bool(checkpoint["selected"]),
            reason=str(checkpoint["reason"]),
            best_loss=_optional_float(checkpoint.get("best_loss")),
            best_step=_optional_int(checkpoint.get("best_step")),
        ),
    )


def _stage_from_dict(payload: Mapping[str, Any]) -> ProtocolStageResult:
    optimizer = payload.get("optimizer_groups")
    train = payload.get("train")
    validation = payload.get("validation")
    if not all(isinstance(item, dict) for item in (optimizer, train, validation)):
        raise TypeError("serialized stage record is incomplete.")
    extension_payload = payload.get("depth_extension")
    extension = (
        None
        if extension_payload is None
        else SerializedDepthExtension(dict(extension_payload))
    )
    assert isinstance(optimizer, dict)
    assert isinstance(train, dict)
    assert isinstance(validation, dict)
    audit = OptimizerParameterGroupAudit(
        positive_names=tuple(optimizer["positive_names"]),
        exact_nonnegative_names=tuple(optimizer["exact_nonnegative_names"]),
        ordinary_names=tuple(optimizer["ordinary_names"]),
        positive_scalars=int(optimizer["positive_scalars"]),
        exact_nonnegative_scalars=int(optimizer["exact_nonnegative_scalars"]),
        ordinary_scalars=int(optimizer["ordinary_scalars"]),
    )
    return ProtocolStageResult(
        index=int(payload["index"]),
        target_depth=int(payload["target_depth"]),
        global_step_start=int(payload["global_step_start"]),
        global_step_end=int(payload["global_step_end"]),
        depth_extension=extension,
        optimizer_groups=audit,
        all_rational_layers_trainable=bool(
            payload["all_rational_layers_trainable"]
        ),
        train=_epoch_result_from_dict(train),
        validation=_epoch_result_from_dict(validation),
    )


def _read_resume_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("resume requires protocol_resume_state.json.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read resume state: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported protocol resume state.")
    return payload


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_path(value: Any) -> Path | None:
    return None if value is None else Path(str(value))


__all__ = [
    "AdamWProtocolConfig",
    "DepthExtensionFactory",
    "DepthExtensionRecord",
    "OptimizerStageSchedule",
    "ProtocolEvaluationRecord",
    "ProtocolRunResult",
    "ProtocolRunnerConfig",
    "ProtocolStageResult",
    "SerializedDepthExtension",
    "SoftTargetFactory",
    "StageSchedule",
    "run_training_protocol",
]
