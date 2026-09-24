from __future__ import annotations

import math
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
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
from training.losses import concentration_probs, output_to_concentrations
from training.schedule import StepProtocol
from utils.seed import set_global_seed
from utils.serialization import write_json

from .data import PreparedFold, prepare_fold
from .model import ModelBuild, build_model, depth_extension_factory, identity_initialization
from .protocol import RunSpec


@dataclass(frozen=True)
class ChineseMnistRunResult:
    run_id: str
    output_dir: str
    elapsed_seconds: float
    metrics: dict[str, Any]


def _run_one(
    spec: RunSpec,
    output_root: str | Path,
    *,
    data_root: str | Path,
    device: torch.device | str = "cpu",
    deterministic: bool = True,
    save_crn: bool = False,
    overwrite: bool = False,
    resume: bool = False,
    include_packages_in_environment: bool = False,
    progress: bool | str = "auto",
) -> ChineseMnistRunResult:
    torch.set_num_threads(int(os.environ.get("NR_TORCH_THREADS", "2")))
    run_dir = Path(output_root).resolve() / spec.run_id
    session = start_experiment_run(
        run_dir,
        protocol=spec.to_dict(),
        overwrite=overwrite,
        resume=resume,
    )
    seed_report = set_global_seed(
        spec.model_seed,
        deterministic=deterministic,
        warn_only=deterministic,
    )
    fold = prepare_fold(data_root, spec)
    if fold.n_classes != spec.num_classes:
        raise RuntimeError("fold class count does not match the protocol.")
    loaders = _make_loaders(spec, fold)
    reference_h = torch.as_tensor(
        fold.x_validation[: min(32, len(fold.x_validation))],
        dtype=torch.float32,
    )
    build = build_model(spec, input_dim=fold.input_dim, reference_h=reference_h)
    steps_per_epoch = math.ceil(len(fold.x_train) / spec.batch_size)
    if spec.nominal_epochs is not None:
        expected_steps = steps_per_epoch * spec.nominal_epochs
        if expected_steps != spec.total_steps:
            raise RuntimeError(
                "nominal epoch budget and optimizer-step budget disagree: "
                f"{expected_steps} != {spec.total_steps}"
            )

    write_json(fold.summary(spec), run_dir / "dataset.json")
    write_json(build.to_dict(), run_dir / "initialization_audit.json")
    step_protocol = StepProtocol(
        total_steps=spec.total_steps,
        peak_lr=spec.learning_rate,
        warmup_steps=spec.warmup_steps,
        min_lr_ratio=spec.min_lr_ratio,
        soft_fraction=spec.soft_steps / spec.total_steps,
        beta_min=1.0,
        beta_max=32.0,
    )
    if step_protocol.soft_steps != spec.soft_steps:
        raise RuntimeError("resolved curriculum boundary is inconsistent.")
    schedule = OptimizerStageSchedule(
        stages=tuple(
            DepthStageConfig(target_depth=depth, optimizer_steps=steps)
            for depth, steps in zip(spec.optimizer_stage_depths, spec.stage_steps)
        )
    )

    def label_smoothed_targets(
        _inputs: Any,
        hard_target: torch.Tensor,
        beta: float,
        _step: int,
    ) -> torch.Tensor:
        alpha = spec.label_smoothing_start / float(beta)
        one_hot = torch.nn.functional.one_hot(
            hard_target.to(torch.long), num_classes=spec.num_classes
        ).to(torch.float32)
        return one_hot * (1.0 - alpha) + alpha / spec.num_classes

    captured_labels: list[torch.Tensor] = []
    captured_probabilities: list[torch.Tensor] = []

    def capture_test_batch(_inputs: Any, target: torch.Tensor, output: Any) -> None:
        concentrations = output_to_concentrations(output)
        probabilities = concentration_probs(
            concentrations, eps=1e-8, check_nonnegative=True
        )
        captured_labels.append(target.detach().cpu().reshape(-1))
        captured_probabilities.append(probabilities.detach().cpu())

    extension_factory = depth_extension_factory(spec, input_dim=fold.input_dim)
    if spec.condition == "dense_continuation":
        extension_strategy = "paired_nonnegative_xavier_uniform"
    elif spec.condition == "continuation":
        extension_strategy = "exact_identity"
    else:
        extension_strategy = "none"

    protocol_result = run_training_protocol(
        build.model,
        build.network,
        loaders.train,
        loaders.validation,
        loaders.test,
        output_dir=run_dir / "training",
        step_protocol=step_protocol,
        continuation_schedule=schedule,
        soft_target_factory=label_smoothed_targets,
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
        reference_h=reference_h,
        identity_initialization=identity_initialization(spec),
        depth_extension_factory=extension_factory,
        depth_extension_strategy=extension_strategy,
        device=device,
        project_root=Path(__file__).resolve().parents[2],
        seed_report=seed_report,
        checkpoint_generators={"train_loader": loaders.generator},
        on_test_batch_completed=capture_test_batch,
        progress=progress,
    )
    predictions = _save_predictions(
        run_dir / "predictions_test.npz",
        fold,
        captured_labels,
        captured_probabilities,
    )
    crn_counts, crn_export = _crn_artifacts(
        build,
        fold,
        run_dir=run_dir,
        device=device,
        save_crn=save_crn,
    )
    write_json(crn_counts, run_dir / "crn_counts.json")
    initialization = build.to_dict()
    initialization["extensions"] = [
        stage.depth_extension.to_dict()
        for stage in protocol_result.stages
        if stage.depth_extension is not None
    ]
    write_json(initialization, run_dir / "initialization_audit.json")
    metrics = {
        "schema_version": 1,
        "run_id": spec.run_id,
        "profile": spec.profile_name,
        "architecture": spec.architecture_name,
        "roles": spec.roles,
        "width": spec.width,
        "depth": spec.depth,
        "condition": spec.condition,
        "factor_code": spec.factor_code,
        "factors": spec.factors,
        "fold": spec.fold,
        "model_seed": spec.model_seed,
        "dataset_fingerprint": fold.dataset_fingerprint,
        "split_fingerprint": fold.split_fingerprint,
        "train_samples": len(fold.train_indices),
        "validation_samples": len(fold.validation_indices),
        "test_samples": len(fold.test_indices),
        "steps_per_nominal_epoch": steps_per_epoch,
        "nominal_epochs": spec.nominal_epochs,
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
        "validation": dict(protocol_result.selected_validation.metrics),
        "test": dict(protocol_result.test.metrics),
        "prediction_archive": predictions,
        "crn_counts": crn_counts,
        "crn_export": crn_export,
    }
    write_json(metrics, run_dir / "metrics.json")
    elapsed = session.elapsed_seconds
    status = {
        "status": "completed",
        "run_id": spec.run_id,
        "optimizer_steps_completed": protocol_result.total_optimizer_steps,
        "test_evaluation_count": protocol_result.test_evaluation_count,
        "test_predictions_captured_in_same_evaluation_pass": True,
        "elapsed_seconds": elapsed,
    }
    session.complete(status)
    return ChineseMnistRunResult(
        run_id=spec.run_id,
        output_dir=run_dir.as_posix(),
        elapsed_seconds=elapsed,
        metrics=metrics,
    )


def run_one(
    spec: RunSpec,
    output_root: str | Path,
    *,
    data_root: str | Path,
    device: torch.device | str = "cpu",
    deterministic: bool = True,
    save_crn: bool = False,
    overwrite: bool = False,
    resume: bool = False,
    include_packages_in_environment: bool = False,
    progress: bool | str = "auto",
) -> ChineseMnistRunResult:
    try:
        return _run_one(
            spec,
            output_root,
            data_root=data_root,
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


def _make_loaders(spec: RunSpec, fold: PreparedFold) -> LoaderBundle:
    return make_classification_loaders(
        torch.from_numpy(fold.x_train),
        torch.from_numpy(fold.y_train),
        torch.from_numpy(fold.x_validation),
        torch.from_numpy(fold.y_validation),
        torch.from_numpy(fold.x_test),
        torch.from_numpy(fold.y_test),
        batch_size=spec.batch_size,
        batch_seed=spec.batch_seed,
    )


def _save_predictions(
    path: Path,
    fold: PreparedFold,
    label_chunks: list[torch.Tensor],
    probability_chunks: list[torch.Tensor],
) -> dict[str, Any]:
    if not label_chunks or not probability_chunks:
        raise RuntimeError("the single test evaluation did not capture predictions.")
    labels = torch.cat(label_chunks).numpy().astype(np.int64, copy=False)
    probabilities = torch.cat(probability_chunks).numpy().astype(np.float32, copy=False)
    expected_labels = fold.y_test
    if not np.array_equal(labels, expected_labels):
        raise AssertionError("captured test prediction order does not match split order.")
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    writers = fold.all_groups[fold.test_indices].astype(np.int64, copy=False)
    np.savez_compressed(
        path,
        sample_index=fold.test_indices,
        writer_id=writers,
        label=labels,
        prediction=predictions,
        probabilities=probabilities,
    )
    return {
        "path": path.as_posix(),
        "n_samples": len(labels),
        "n_classes": probabilities.shape[1],
        "writer_count": len(np.unique(writers)),
        "same_pass_as_test_metrics": True,
    }


@torch.no_grad()
def _crn_artifacts(
    build: ModelBuild,
    fold: PreparedFold,
    *,
    run_dir: Path,
    device: torch.device | str,
    save_crn: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    reference = None
    output_path = None
    if save_crn:
        x = torch.from_numpy(fold.x_test[:1]).to(device)
        state = build.model(x)
        reference = reference_case_from_state(
            state,
            crn_input=x[0],
            target=int(fold.y_test[0]),
            sample_index=int(fold.test_indices[0]),
        )
        output_path = run_dir / "best_model.crn"
    return compile_crn_artifacts(
        build.network,
        build.network,
        reference=reference,
        output_path=output_path,
    )


__all__ = ["ChineseMnistRunResult", "run_one"]
