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
    adaptive_pysb_validation,
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

from .data import PreparedFold, load_dataset, prepare_fold
from .model import ModelBuild, build_model
from .protocol import RunSpec


@dataclass(frozen=True)
class SingleLayerRunResult:
    run_id: str
    output_dir: str
    elapsed_seconds: float
    metrics: dict[str, Any]


def _run_one(
    spec: RunSpec,
    output_root: str | Path,
    *,
    st003390_csv: str | Path,
    device: torch.device | str = "cpu",
    deterministic: bool = True,
    save_crn: bool = False,
    pysb_samples: int = 0,
    overwrite: bool = False,
    resume: bool = False,
    include_packages_in_environment: bool = False,
    progress: bool | str = "auto",
) -> SingleLayerRunResult:
    """Run one D1 fold with fold-local preprocessing and fixed step budget."""

    if pysb_samples < 0:
        raise ValueError("pysb_samples must be nonnegative.")
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
    dataset = load_dataset(spec, st003390_csv=st003390_csv)
    fold = prepare_fold(spec, dataset)
    loaders = _make_loaders(spec, fold)
    build = build_model(
        spec,
        n_inputs=fold.n_features,
        n_classes=fold.n_classes,
    )

    write_json(fold.summary(spec), run_dir / "dataset.json")
    write_json(build.to_dict(), run_dir / "model.json")

    step_protocol = StepProtocol(
        total_steps=spec.total_steps,
        peak_lr=spec.learning_rate,
        warmup_steps=spec.warmup_steps,
        min_lr_ratio=spec.min_lr_ratio,
        soft_fraction=spec.soft_fraction,
        beta_min=1.0,
        beta_max=32.0,
    )
    schedule = OptimizerStageSchedule(
        stages=(
            DepthStageConfig(target_depth=1, optimizer_steps=spec.total_steps),
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
            hard_target.to(torch.long),
            num_classes=fold.n_classes,
        ).to(torch.float32)
        return one_hot * (1.0 - alpha) + alpha / fold.n_classes

    reference_raw = torch.as_tensor(
        fold.x_validation[: min(32, len(fold.x_validation))],
        dtype=torch.float32,
    )
    with torch.no_grad():
        reference_h = build.model.features(reference_raw)
    protocol_result = run_training_protocol(
        build.model,
        build.model.network,
        loaders.train,
        loaders.validation,
        loaders.test,
        output_dir=run_dir / "training",
        step_protocol=step_protocol,
        continuation_schedule=schedule,
        soft_target_factory=label_smoothed_targets,
        experiment_protocol=ExperimentProtocol(
            minimum_selection_step=step_protocol.soft_steps,
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
        device=device,
        project_root=Path(__file__).resolve().parents[2],
        seed_report=seed_report,
        checkpoint_generators={"train_loader": loaders.generator},
        progress=progress,
    )

    crn_counts, crn_export = _compile_crn(
        build,
        fold,
        run_dir=run_dir,
        device=device,
        save_crn=save_crn,
    )
    write_json(crn_counts, run_dir / "crn_counts.json")
    pysb = _validate_pysb(
        build,
        fold,
        run_dir=run_dir,
        device=device,
        n_samples=pysb_samples,
    )
    decision_boundary = _save_circle_decision_boundary(
        build,
        fold,
        run_dir=run_dir,
        device=device,
    )
    metrics = {
        "schema_version": 1,
        "run_id": spec.run_id,
        "dataset": spec.dataset,
        "representation": spec.representation,
        "depth": 1,
        "repeat": spec.repeat,
        "fold": spec.fold,
        "model_seed": spec.model_seed,
        "dataset_fingerprint": fold.fingerprint,
        "raw_input_dim": fold.n_features,
        "basis_dim": build.basis_dim,
        "n_classes": fold.n_classes,
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
        "crn_counts": crn_counts,
        "crn_export": crn_export,
        "pysb_validation": pysb,
        "decision_boundary": decision_boundary,
    }
    write_json(metrics, run_dir / "metrics.json")
    elapsed = session.elapsed_seconds
    session.complete(
        {
            "optimizer_steps_completed": protocol_result.total_optimizer_steps,
            "test_evaluation_count": protocol_result.test_evaluation_count,
        }
    )
    return SingleLayerRunResult(
        run_id=spec.run_id,
        output_dir=run_dir.as_posix(),
        elapsed_seconds=elapsed,
        metrics=metrics,
    )


def run_one(
    spec: RunSpec,
    output_root: str | Path,
    *,
    st003390_csv: str | Path,
    device: torch.device | str = "cpu",
    deterministic: bool = True,
    save_crn: bool = False,
    pysb_samples: int = 0,
    overwrite: bool = False,
    resume: bool = False,
    include_packages_in_environment: bool = False,
    progress: bool | str = "auto",
) -> SingleLayerRunResult:
    try:
        return _run_one(
            spec,
            output_root,
            st003390_csv=st003390_csv,
            device=device,
            deterministic=deterministic,
            save_crn=save_crn,
            pysb_samples=pysb_samples,
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


@torch.no_grad()
def _compile_crn(
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
        x = torch.as_tensor(fold.x_test[:1], dtype=torch.float32, device=device)
        state = build.model(x)
        reference = reference_case_from_state(
            state,
            crn_input=x[0],
            target=int(fold.y_test[0]),
            sample_index=int(fold.indices.test[0]),
        )
        output_path = run_dir / "best_model.crn"
    return compile_crn_artifacts(
        build.model,
        build.model.network,
        reference=reference,
        output_path=output_path,
        count_metadata={
            "raw_input_dim": build.raw_input_dim,
            "basis_dim": build.basis_dim,
        },
    )


@torch.no_grad()
def _validate_pysb(
    build: ModelBuild,
    fold: PreparedFold,
    *,
    run_dir: Path,
    device: torch.device | str,
    n_samples: int,
) -> dict[str, Any] | None:
    if n_samples == 0:
        return None
    count = min(n_samples, len(fold.x_test))
    cases: list[dict[str, Any]] = []
    model_spec = build.model.export_model_spec()
    for position in range(count):
        x = torch.as_tensor(
            fold.x_test[position : position + 1],
            dtype=torch.float32,
            device=device,
        )
        state = build.model(x)
        reference = reference_case_from_state(
            state,
            crn_input=x[0],
            target=int(fold.y_test[position]),
            sample_index=int(fold.indices.test[position]),
        )
        cases.append(
            {
                "reference": reference.to_dict(),
                "pysb": adaptive_pysb_validation(
                    model_spec,
                    reference,
                    initial_t_end=20.0,
                    max_t_end=320.0,
                    n_timepoints=201,
                    atol=1e-4,
                    rtol=1e-4,
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "n_cases": len(cases),
        "n_pysb_passed": sum(item["pysb"]["passed"] for item in cases),
        "n_pysb_errors": 0,
        "passed": all(item["pysb"]["passed"] for item in cases),
        "cases": cases,
    }
    write_json(payload, run_dir / "pysb_validation.json")
    return payload


@torch.no_grad()
def _save_circle_decision_boundary(
    build: ModelBuild,
    fold: PreparedFold,
    *,
    run_dir: Path,
    device: torch.device | str,
    grid_size: int = 301,
) -> dict[str, Any] | None:
    """Save a fold-specific dense probability grid without another test pass."""

    if fold.dataset.name != "circles":
        return None
    if grid_size < 33:
        raise ValueError("circle decision grid must be at least 33x33.")
    source = np.asarray(fold.dataset.X, dtype=np.float64)
    span = np.maximum(source.max(axis=0) - source.min(axis=0), 1e-6)
    lower = source.min(axis=0) - 0.08 * span
    upper = source.max(axis=0) + 0.08 * span
    x_axis = np.linspace(lower[0], upper[0], grid_size, dtype=np.float64)
    y_axis = np.linspace(upper[1], lower[1], grid_size, dtype=np.float64)
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
    raw_grid = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)
    transformed = fold.preprocessor.transform(raw_grid)
    probabilities: list[torch.Tensor] = []
    build.model.eval()
    for start in range(0, len(transformed), 4_096):
        batch = torch.from_numpy(transformed[start : start + 4_096]).to(device)
        state = build.model(batch)
        probabilities.append(state.normalized_scores().detach().cpu())
    scores = torch.cat(probabilities, dim=0).numpy().astype(np.float32)
    class_one = scores[:, 1].reshape(grid_size, grid_size)
    archive_path = run_dir / "decision_boundary_grid.npz"
    np.savez_compressed(
        archive_path,
        x_axis=x_axis.astype(np.float32),
        y_axis=y_axis.astype(np.float32),
        class1_probability=class_one,
        predicted_class=(class_one >= 0.5).astype(np.uint8),
        test_x=source[fold.indices.test].astype(np.float32),
        test_y=fold.y_test.astype(np.int64),
    )
    png_path = run_dir / "decision_boundary.png"
    _write_circle_png(
        png_path,
        class_one,
        test_x=source[fold.indices.test],
        test_y=fold.y_test,
        lower=lower,
        upper=upper,
    )
    payload = {
        "schema_version": 1,
        "grid_size": grid_size,
        "coordinate_space": "original make_circles coordinates",
        "probability": "normalized class-1 active-output concentration",
        "archive": archive_path.name,
        "preview": png_path.name,
        "x_range": [float(lower[0]), float(upper[0])],
        "y_range": [float(lower[1]), float(upper[1])],
    }
    write_json(payload, run_dir / "decision_boundary.json")
    return payload


def _write_circle_png(
    path: Path,
    class_one: np.ndarray,
    *,
    test_x: np.ndarray,
    test_y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    from PIL import Image, ImageDraw

    probability = np.clip(np.asarray(class_one, dtype=np.float64), 0.0, 1.0)
    red = (238.0 * (1.0 - probability) + 72.0 * probability).astype(np.uint8)
    green = (154.0 * (1.0 - probability) + 135.0 * probability).astype(np.uint8)
    blue = (88.0 * (1.0 - probability) + 210.0 * probability).astype(np.uint8)
    image = Image.fromarray(np.stack((red, green, blue), axis=2), mode="RGB")
    draw = ImageDraw.Draw(image)
    height, width = probability.shape
    for point, label in zip(test_x, test_y, strict=True):
        px = int(round((float(point[0]) - lower[0]) / (upper[0] - lower[0]) * (width - 1)))
        py = int(round((upper[1] - float(point[1])) / (upper[1] - lower[1]) * (height - 1)))
        color = (255, 255, 255) if int(label) == 0 else (0, 0, 0)
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
    image.save(path)


__all__ = ["SingleLayerRunResult", "run_one"]
