from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import traceback
from typing import Any

import numpy as np
import torch

from experiments.common import (
    AdamWProtocolConfig,
    ExperimentProtocol,
    LoaderBundle,
    OptimizerStageSchedule,
    ProtocolRunnerConfig,
    compile_crn_artifacts,
    make_classification_loaders,
    mark_experiment_failed,
    reference_case_from_state,
    run_training_protocol,
    start_experiment_run,
)
from training.continuation import DepthStageConfig
from training.schedule import StepProtocol
from utils.seed import set_global_seed
from utils.serialization import write_json

from .data import (
    DatasetPack,
    edge_metrics,
    encode_inputs,
    make_dataset,
    soft_targets,
)
from .model import (
    ModelBuild,
    build_model,
    depth_extension_factory,
    identity_initialization,
)
from .protocol import RunSpec


@dataclass(frozen=True)
class ChebyshevRunResult:
    run_id: str
    output_dir: str
    elapsed_seconds: float
    metrics: dict[str, Any]


def _run_one(
    spec: RunSpec,
    output_root: str | Path,
    *,
    device: torch.device | str = "cpu",
    deterministic: bool = True,
    save_crn: bool = False,
    overwrite: bool = False,
    resume: bool = False,
    include_packages_in_environment: bool = False,
    progress: bool | str = "auto",
) -> ChebyshevRunResult:
    """Run one paired cell through the common fixed-budget training loop."""

    run_dir = Path(output_root).resolve() / spec.run_id
    session = start_experiment_run(
        run_dir,
        protocol=spec.to_dict(),
        overwrite=overwrite,
        resume=resume,
    )
    seed_report = set_global_seed(
        spec.seed,
        deterministic=deterministic,
        warn_only=deterministic,
    )
    dataset = make_dataset(spec)
    loaders = _make_loaders(spec, dataset)
    reference_raw = dataset.x_val[: min(64, len(dataset.x_val))]
    build = build_model(
        spec,
        reference_raw=reference_raw,
        deterministic=deterministic,
    )

    write_json(dataset.summary(), run_dir / "dataset.json")
    write_json(
        _initialization_payload(spec, build),
        run_dir / "initialization_audit.json",
    )

    step_protocol = StepProtocol(
        total_steps=spec.total_steps,
        peak_lr=spec.learning_rate,
        warmup_steps=spec.warmup_steps,
        min_lr_ratio=spec.min_lr_ratio,
        soft_fraction=spec.soft_steps / spec.total_steps,
        beta_min=spec.beta_min,
        beta_max=spec.beta_max,
    )
    if step_protocol.soft_steps != spec.soft_steps:
        raise RuntimeError("resolved soft curriculum boundary is inconsistent.")
    schedule = OptimizerStageSchedule(
        stages=tuple(
            DepthStageConfig(target_depth=depth, optimizer_steps=steps)
            for depth, steps in zip(
                spec.optimizer_stage_depths,
                spec.stage_steps,
            )
        )
    )

    def task_soft_targets(
        inputs: Any,
        _hard_target: torch.Tensor,
        beta: float,
        _step: int,
    ) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("Chebyshev inputs must be tensors.")
        return soft_targets(inputs, degree=spec.task.degree, beta=beta)

    extension_factory = depth_extension_factory(
        spec,
        deterministic=deterministic,
    )
    if spec.condition == "dense_continuation":
        extension_strategy = "paired_nonnegative_xavier_uniform"
    elif spec.condition == "continuation":
        extension_strategy = "exact_identity"
    else:
        extension_strategy = "none"

    protocol_result = run_training_protocol(
        build.model,
        build.model.network,
        loaders.train,
        loaders.validation,
        loaders.test,
        output_dir=run_dir / "training",
        step_protocol=step_protocol,
        continuation_schedule=schedule,
        soft_target_factory=task_soft_targets,
        experiment_protocol=ExperimentProtocol(
            minimum_selection_step=spec.soft_steps,
        ),
        optimizer_config=AdamWProtocolConfig(
            positive_weight_decay=0.0,
            exact_nonnegative_weight_decay=0.0,
            ordinary_weight_decay=0.0,
        ),
        runner_config=ProtocolRunnerConfig(
            evaluation_interval=spec.eval_interval,
            max_grad_norm=spec.grad_clip,
            include_packages_in_environment=include_packages_in_environment,
            resume_from_latest=resume,
            manage_training_status=False,
        ),
        reference_h=encode_inputs(reference_raw, spec.task),
        identity_initialization=identity_initialization(spec),
        depth_extension_factory=extension_factory,
        depth_extension_strategy=extension_strategy,
        device=device,
        project_root=Path(__file__).resolve().parents[2],
        seed_report=seed_report,
        checkpoint_generators={"train_loader": loaders.generator},
        progress=progress,
    )

    grid = _evaluate_grid(
        build.model,
        dataset,
        device=device,
        batch_size=max(256, spec.batch_size),
        tolerance_cells=spec.edge_tolerance_cells,
    )
    counts, export_payload = _crn_artifacts(
        build.model,
        dataset,
        run_dir=run_dir,
        device=device,
        save_crn=save_crn,
    )
    write_json(counts, run_dir / "crn_counts.json")
    metrics = _metrics_payload(
        spec,
        dataset,
        protocol_result,
        grid=grid,
        crn_counts=counts,
        crn_export=export_payload,
    )
    write_json(metrics, run_dir / "metrics.json")
    extensions = [
        stage.depth_extension.to_dict()
        for stage in protocol_result.stages
        if stage.depth_extension is not None
    ]
    initialization = _initialization_payload(spec, build)
    initialization["extensions"] = extensions
    write_json(initialization, run_dir / "initialization_audit.json")
    elapsed = session.elapsed_seconds
    status = {
        "status": "completed",
        "run_id": spec.run_id,
        "optimizer_steps_completed": protocol_result.total_optimizer_steps,
        "test_evaluation_count": protocol_result.test_evaluation_count,
        "elapsed_seconds": elapsed,
    }
    session.complete(status)
    return ChebyshevRunResult(
        run_id=spec.run_id,
        output_dir=run_dir.as_posix(),
        elapsed_seconds=elapsed,
        metrics=metrics,
    )


def run_one(
    spec: RunSpec,
    output_root: str | Path,
    *,
    device: torch.device | str = "cpu",
    deterministic: bool = True,
    save_crn: bool = False,
    overwrite: bool = False,
    resume: bool = False,
    include_packages_in_environment: bool = False,
    progress: bool | str = "auto",
) -> ChebyshevRunResult:
    try:
        return _run_one(
            spec,
            output_root,
            device=device,
            deterministic=deterministic,
            save_crn=save_crn,
            overwrite=overwrite,
            resume=resume,
            include_packages_in_environment=include_packages_in_environment,
            progress=progress,
        )
    except BaseException as error:
        mark_experiment_failed(
            Path(output_root) / spec.run_id,
            error,
            run_id=spec.run_id,
            traceback_text=traceback.format_exc(),
        )
        raise


def _make_loaders(spec: RunSpec, dataset: DatasetPack) -> LoaderBundle:
    return make_classification_loaders(
        dataset.x_train,
        dataset.y_train,
        dataset.x_val,
        dataset.y_val,
        dataset.x_test,
        dataset.y_test,
        batch_size=spec.batch_size,
        batch_seed=spec.batch_seed,
    )


@torch.no_grad()
def _evaluate_grid(
    model,
    dataset: DatasetPack,
    *,
    device: torch.device | str,
    batch_size: int,
    tolerance_cells: int,
) -> dict[str, Any]:
    from data import make_tensor_dataloader

    loader = make_tensor_dataloader(
        dataset.x_grid,
        dataset.y_grid,
        batch_size=batch_size,
        shuffle=False,
    )
    predictions: list[torch.Tensor] = []
    model.eval()
    for x_raw, _ in loader:
        state = model(x_raw.to(device))
        predictions.append(state.predicted_classes.detach().cpu())
    predicted = torch.cat(predictions).numpy()
    target = dataset.y_grid.detach().cpu().numpy()
    payload: dict[str, Any] = {
        "accuracy": float(np.mean(predicted == target)),
    }
    payload.update(
        edge_metrics(
            predicted,
            target,
            grid_shape=dataset.grid_shape,
            tolerance_cells=tolerance_cells,
        )
    )
    return payload


@torch.no_grad()
def _crn_artifacts(
    model,
    dataset: DatasetPack,
    *,
    run_dir: Path,
    device: torch.device | str,
    save_crn: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    reference = None
    output_path = None
    if save_crn:
        x_raw = dataset.x_test[:1].to(device)
        state = model(x_raw)
        encoded = encode_inputs(x_raw, model.task)
        reference = reference_case_from_state(
            state,
            crn_input=encoded[0],
            target=int(dataset.y_test[0].item()),
            sample_index=0,
        )
        output_path = run_dir / "best_model.crn"
    return compile_crn_artifacts(
        model,
        model.network,
        reference=reference,
        output_path=output_path,
    )


def _metrics_payload(
    spec: RunSpec,
    dataset: DatasetPack,
    protocol_result,
    *,
    grid: dict[str, Any],
    crn_counts: dict[str, Any],
    crn_export: dict[str, Any] | None,
) -> dict[str, Any]:
    validation = protocol_result.selected_validation.metrics
    test = protocol_result.test.metrics
    return {
        "run_id": spec.run_id,
        "task": spec.task.name,
        "degree": spec.task.degree,
        "input_encoding": spec.task.input_encoding,
        "width": spec.width,
        "depth": spec.depth,
        "seed": spec.seed,
        "condition": spec.condition,
        "factor_code": spec.factor_code,
        "factors": spec.factors,
        "dataset_fingerprint": dataset.fingerprint,
        "total_optimizer_steps": protocol_result.total_optimizer_steps,
        "best_global_step": protocol_result.best_global_step,
        "best_validation_loss": protocol_result.best_validation_loss,
        "diagnostic_global_best_step": (
            protocol_result.diagnostic_global_best_step
        ),
        "diagnostic_global_best_loss": (
            protocol_result.diagnostic_global_best_loss
        ),
        "test_evaluation_count": protocol_result.test_evaluation_count,
        "validation": dict(validation),
        "test": dict(test),
        "grid": grid,
        "crn_counts": crn_counts,
        "crn_export": crn_export,
    }


def _initialization_payload(spec: RunSpec, build: ModelBuild) -> dict[str, Any]:
    return {
        "condition": spec.condition,
        "factor_code": spec.factor_code,
        "factors": spec.factors,
        "model_seed": build.model_seed,
        "boundary_parameter_fingerprint": build.boundary_parameter_fingerprint,
        "layers": build.initialization_audit.to_dict(),
        "initial_function_audit": (
            None
            if build.initial_function_audit is None
            else build.initial_function_audit.to_dict()
        ),
        "extensions": [],
    }


__all__ = ["ChebyshevRunResult", "run_one"]
