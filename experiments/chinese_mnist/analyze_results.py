from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.common import (
    aggregate_metric,
    exact_sign_test_pvalue,
    planned_run_directories,
)
from reporting import write_csv_rows
from utils.serialization import write_json


RUN_METRICS = (
    "test_accuracy",
    "test_bacc",
    "test_f1_macro",
    "test_auc_macro",
    "test_nll",
)
WRITER_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "nll")
CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("D2-D1", "d1_k15_i0_c0", "d2_w24_i0_c0"),
    ("D3-D2", "d2_w24_i0_c0", "d3_w24_i1_c0"),
    ("D5-D3", "d3_w24_i1_c0", "d5_w24_i1_c0"),
    ("D5-D1", "d1_k15_i0_c0", "d5_w24_i1_c0"),
    ("D5 I@C0", "d5_w24_i0_c0", "d5_w24_i1_c0"),
    ("D5 C@I0", "d5_w24_i0_c0", "d5_w24_i0_c1"),
    ("D5 11-00", "d5_w24_i0_c0", "d5_w24_i1_c1"),
    ("D5 11-10", "d5_w24_i1_c0", "d5_w24_i1_c1"),
    ("D5 11-01", "d5_w24_i0_c1", "d5_w24_i1_c1"),
    ("D7 I@C0", "d7_w24_i0_c0", "d7_w24_i1_c0"),
    ("D7 C@I0", "d7_w24_i0_c0", "d7_w24_i0_c1"),
    ("D7 11-00", "d7_w24_i0_c0", "d7_w24_i1_c1"),
    ("D7 11-10", "d7_w24_i1_c0", "d7_w24_i1_c1"),
    ("D7 11-01", "d7_w24_i0_c1", "d7_w24_i1_c1"),
    ("D10 I@C0", "d10_w24_i0_c0", "d10_w24_i1_c0"),
    ("D10 C@I0", "d10_w24_i0_c0", "d10_w24_i0_c1"),
    ("D10 11-00", "d10_w24_i0_c0", "d10_w24_i1_c1"),
    ("D10 11-10", "d10_w24_i1_c0", "d10_w24_i1_c1"),
    ("D10 11-01", "d10_w24_i0_c1", "d10_w24_i1_c1"),
)


def analyze_results(
    output_root: str | Path,
    *,
    analysis_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    destination = (
        Path(analysis_dir).resolve()
        if analysis_dir is not None
        else root / "analysis"
    )
    destination.mkdir(parents=True, exist_ok=True)
    runs, prediction_sets, issues = _load_runs(root)
    aggregates = _aggregate_runs(runs)
    writer_rows = _writer_metrics(prediction_sets)
    paired_rows, fold_difference_rows = _writer_comparisons(root, writer_rows)
    confusion_rows = _confusion_rows(prediction_sets)
    write_csv_rows(runs, destination / "runs.csv")
    write_csv_rows(aggregates, destination / "aggregate.csv")
    write_csv_rows(writer_rows, destination / "writer_metrics.csv")
    write_csv_rows(paired_rows, destination / "writer_paired_comparisons.csv")
    write_csv_rows(
        fold_difference_rows, destination / "fold_paired_differences.csv"
    )
    write_csv_rows(confusion_rows, destination / "confusion.csv")
    write_csv_rows(issues, destination / "integrity_issues.csv")
    summary = {
        "completed_runs": len(runs),
        "architectures_with_predictions": len(prediction_sets),
        "aggregate_rows": len(aggregates),
        "writer_metric_rows": len(writer_rows),
        "writer_paired_rows": len(paired_rows),
        "fold_paired_difference_rows": len(fold_difference_rows),
        "integrity_issue_count": len(issues),
    }
    write_json(summary, destination / "summary.json")
    (destination / "report.md").write_text(
        _render_report(summary, aggregates, paired_rows, issues), encoding="utf-8"
    )
    return summary


def _load_runs(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, np.ndarray]]], list[dict[str, str]]]:
    runs: list[dict[str, Any]] = []
    predictions: dict[str, list[dict[str, np.ndarray]]] = defaultdict(list)
    issues: list[dict[str, str]] = []
    run_dirs, plan_issues = planned_run_directories(root)
    issues.extend(plan_issues)
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            protocol = json.loads((run_dir / "resolved_protocol.json").read_text(encoding="utf-8"))
            status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
            with np.load(run_dir / "predictions_test.npz") as archive:
                prediction = {key: np.asarray(archive[key]).copy() for key in archive.files}
        except (OSError, KeyError, json.JSONDecodeError, ValueError) as exc:
            issues.append(
                {"run_id": run_dir.name, "issue": "unreadable_completed_artifact", "detail": str(exc)}
            )
            continue
        if status.get("status") != "completed":
            issues.append(
                {"run_id": run_dir.name, "issue": "metrics_without_completed_status", "detail": str(status.get("status"))}
            )
            continue
        try:
            row = _run_row(protocol, metrics)
            _validate_prediction(prediction, row)
        except (KeyError, TypeError, ValueError, AssertionError) as exc:
            issues.append(
                {"run_id": run_dir.name, "issue": "invalid_completed_artifact_schema", "detail": str(exc)}
            )
            continue
        runs.append(row)
        prediction["fold"] = np.asarray([int(row["fold"])], dtype=np.int64)
        predictions[str(row["architecture"])].append(prediction)

    by_fold_depth: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        by_fold_depth[(int(row["fold"]), int(row["depth"]))].append(row)
    for key, group in by_fold_depth.items():
        fingerprints = {row["split_fingerprint"] for row in group}
        if len(fingerprints) != 1:
            issues.append(
                {"run_id": str(key), "issue": "paired_split_fingerprint_mismatch", "detail": str(sorted(fingerprints))}
            )
    return sorted(runs, key=lambda row: row["run_id"]), dict(predictions), issues


def _run_row(protocol: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    test = metrics["test"]
    crn = metrics["crn_counts"]
    return {
        "run_id": str(protocol["run_id"]),
        "profile": str(protocol["profile_name"]),
        "architecture": str(protocol["architecture_name"]),
        "roles": "|".join(protocol["roles"]),
        "width": int(protocol["width"]),
        "depth": int(protocol["depth"]),
        "condition": str(protocol["condition"]),
        "factor_code": str(protocol["factor_code"]),
        "fold": int(protocol["fold"]),
        "model_seed": int(protocol["model_seed"]),
        "dataset_fingerprint": str(metrics["dataset_fingerprint"]),
        "split_fingerprint": str(metrics["split_fingerprint"]),
        "total_optimizer_steps": int(metrics["total_optimizer_steps"]),
        "best_global_step": int(metrics["best_global_step"]),
        "best_validation_loss": float(metrics["best_validation_loss"]),
        "test_accuracy": _metric(test, "accuracy", "acc"),
        "test_bacc": _metric(test, "recall_macro"),
        "test_f1_macro": _metric(test, "f1_macro"),
        "test_auc_macro": _metric(test, "auc_macro"),
        "test_nll": _metric(test, "classification"),
        "species_count": int(crn["species_count"]),
        "reaction_count": int(crn["reaction_count"]),
        "parameter_count": int(crn["parameter_count"]),
    }


def _metric(payload: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = payload.get(key)
        if value is not None and math.isfinite(float(value)):
            return float(value)
    return float("nan")


def _validate_prediction(prediction: dict[str, np.ndarray], row: dict[str, Any]) -> None:
    required = {"sample_index", "writer_id", "label", "prediction", "probabilities"}
    if not required.issubset(prediction):
        raise KeyError(f"prediction archive is missing {sorted(required - set(prediction))}")
    n = len(prediction["label"])
    if not all(len(prediction[key]) == n for key in required):
        raise ValueError("prediction archive arrays have inconsistent lengths.")
    probabilities = prediction["probabilities"]
    if probabilities.ndim != 2 or probabilities.shape[1] != 15:
        raise ValueError("prediction probabilities must have shape (n,15).")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("prediction probabilities are invalid.")
    if not np.array_equal(probabilities.argmax(axis=1), prediction["prediction"]):
        raise AssertionError("stored predictions do not match probabilities.")
    if len(np.unique(prediction["writer_id"])) != 20:
        raise ValueError(f"fold {row['fold']} does not contain 20 test writers.")


def _aggregate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        groups[str(row["architecture"])].append(row)
    output: list[dict[str, Any]] = []
    for architecture, group in sorted(groups.items()):
        for metric in RUN_METRICS:
            values = [float(row[metric]) for row in group if math.isfinite(float(row[metric]))]
            if not values:
                continue
            aggregate = aggregate_metric(values)
            output.append(
                {
                    "architecture": architecture,
                    "width": group[0]["width"],
                    "depth": group[0]["depth"],
                    "condition": group[0]["condition"],
                    "factor_code": group[0]["factor_code"],
                    "metric": metric,
                    **aggregate.to_dict(),
                }
            )
    return output


def _writer_metrics(
    prediction_sets: dict[str, list[dict[str, np.ndarray]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for architecture, archives in sorted(prediction_sets.items()):
        for archive in archives:
            fold = int(archive["fold"][0])
            for writer in sorted(np.unique(archive["writer_id"]).tolist()):
                mask = archive["writer_id"] == writer
                labels = archive["label"][mask].astype(np.int64)
                predictions = archive["prediction"][mask].astype(np.int64)
                probabilities = archive["probabilities"][mask]
                confusion = _confusion(labels, predictions, 15)
                recalls = np.diag(confusion) / np.maximum(1, confusion.sum(axis=1))
                precision = np.diag(confusion) / np.maximum(1, confusion.sum(axis=0))
                f1 = 2 * precision * recalls / np.maximum(1e-12, precision + recalls)
                true_probability = probabilities[np.arange(len(labels)), labels]
                output.append(
                    {
                        "architecture": architecture,
                        "fold": fold,
                        "writer_id": int(writer),
                        "n_samples": len(labels),
                        "accuracy": float(np.mean(labels == predictions)),
                        "balanced_accuracy": float(np.mean(recalls)),
                        "macro_f1": float(np.mean(f1)),
                        "nll": float(-np.log(np.clip(true_probability, 1e-12, 1.0)).mean()),
                    }
                )
    return output


def _writer_comparisons(
    root: Path,
    writer_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    profile_path = root / "profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.is_file() else {}
    replicates = int(profile.get("bootstrap_replicates", 2_000))
    base_seed = int(profile.get("bootstrap_seed", 20_260_731))
    by_architecture: dict[
        str,
        dict[tuple[int, int], dict[str, float]],
    ] = defaultdict(dict)
    for row in writer_rows:
        key = (int(row["fold"]), int(row["writer_id"]))
        by_architecture[str(row["architecture"])][key] = {
            metric: float(row[metric]) for metric in WRITER_METRICS
        }
    output: list[dict[str, Any]] = []
    fold_output: list[dict[str, Any]] = []
    for contrast, baseline, treatment in CONTRASTS:
        if baseline not in by_architecture or treatment not in by_architecture:
            continue
        baseline_rows = by_architecture[baseline]
        treatment_rows = by_architecture[treatment]
        writer_keys = sorted(set(baseline_rows) & set(treatment_rows))
        if not writer_keys:
            continue
        for metric in WRITER_METRICS:
            differences_by_fold: dict[int, np.ndarray] = {}
            for fold in sorted({key[0] for key in writer_keys}):
                keys = [key for key in writer_keys if key[0] == fold]
                differences_by_fold[fold] = np.asarray(
                    [
                        treatment_rows[key][metric]
                        - baseline_rows[key][metric]
                        for key in keys
                    ],
                    dtype=np.float64,
                )
            fold_means = np.asarray(
                [values.mean() for values in differences_by_fold.values()],
                dtype=np.float64,
            )
            aggregate = aggregate_metric(fold_means.tolist())
            wins = int(np.sum(fold_means > 1e-12))
            losses = int(np.sum(fold_means < -1e-12))
            ties = len(fold_means) - wins - losses
            low, high = _hierarchical_bootstrap_mean_ci(
                differences_by_fold,
                replicates=replicates,
                seed=_contrast_seed(base_seed, contrast, metric),
            )
            for fold, values in differences_by_fold.items():
                fold_output.append(
                    {
                        "contrast": contrast,
                        "baseline": baseline,
                        "treatment": treatment,
                        "metric": metric,
                        "fold": fold,
                        "n_writers": len(values),
                        "fold_mean_difference": float(values.mean()),
                        "fold_median_difference": float(np.median(values)),
                        "writer_differences": "|".join(
                            f"{value:.12g}" for value in values.tolist()
                        ),
                    }
                )
            output.append(
                {
                    "contrast": contrast,
                    "baseline": baseline,
                    "treatment": treatment,
                    "metric": metric,
                    "n_folds": len(fold_means),
                    "n_writers": len(writer_keys),
                    "mean_difference": aggregate.mean,
                    "median_fold_difference": aggregate.median,
                    "q1_fold_difference": aggregate.q1,
                    "q3_fold_difference": aggregate.q3,
                    "fold_iqr": aggregate.iqr,
                    "fold_standard_deviation": aggregate.standard_deviation,
                    "fold_t_ci95_low": aggregate.ci95_low,
                    "fold_t_ci95_high": aggregate.ci95_high,
                    "hierarchical_bootstrap_ci95_low": low,
                    "hierarchical_bootstrap_ci95_high": high,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "win_rate_excluding_ties": (
                        None if wins + losses == 0 else wins / (wins + losses)
                    ),
                    "sign_test_pvalue": (
                        None if wins + losses == 0 else exact_sign_test_pvalue(wins, losses)
                    ),
                    "bootstrap_replicates": replicates,
                    "primary_inference_unit": "outer_fold",
                }
            )
    return output, fold_output


def _hierarchical_bootstrap_mean_ci(
    values_by_fold: dict[int, np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    if not values_by_fold:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    folds = tuple(sorted(values_by_fold))
    samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.integers(0, len(folds), size=len(folds))
        fold_means: list[float] = []
        for index in selected.tolist():
            values = values_by_fold[folds[index]]
            writer_indices = rng.integers(0, len(values), size=len(values))
            fold_means.append(float(values[writer_indices].mean()))
        samples[replicate] = float(np.mean(fold_means))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _contrast_seed(base_seed: int, contrast: str, metric: str) -> int:
    digest = hashlib.sha256(f"{contrast}|{metric}".encode("utf-8")).digest()
    return int((base_seed + int.from_bytes(digest[:4], "little")) % (2**32))


def _confusion_rows(
    prediction_sets: dict[str, list[dict[str, np.ndarray]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for architecture, archives in sorted(prediction_sets.items()):
        labels = np.concatenate([archive["label"] for archive in archives]).astype(np.int64)
        predictions = np.concatenate([archive["prediction"] for archive in archives]).astype(np.int64)
        matrix = _confusion(labels, predictions, 15)
        row_sums = np.maximum(1, matrix.sum(axis=1, keepdims=True))
        normalized = matrix / row_sums
        for true_label in range(15):
            for predicted_label in range(15):
                output.append(
                    {
                        "architecture": architecture,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": int(matrix[true_label, predicted_label]),
                        "row_normalized": float(normalized[true_label, predicted_label]),
                    }
                )
    return output


def _confusion(labels: np.ndarray, predictions: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def _render_report(
    summary: dict[str, Any],
    aggregates: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    issues: list[dict[str, str]],
) -> str:
    lines = [
        "# Chinese-MNIST depth and optimization-rescue analysis",
        "",
        f"- Completed runs: {summary['completed_runs']}",
        f"- Writer-level paired rows: {summary['writer_paired_rows']}",
        f"- Integrity issues: {summary['integrity_issue_count']}",
        "",
        "## Test balanced accuracy",
        "",
        "| Architecture | n | Mean | SD |",
        "|---|---:|---:|---:|",
    ]
    for row in aggregates:
        if row["metric"] == "test_bacc":
            lines.append(
                f"| {row['architecture']} | {row['n']} | {row['mean']:.6f} | {row['standard_deviation']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Writer-paired balanced-accuracy contrasts",
            "",
            "| Contrast | n writers | Mean difference | 95% bootstrap CI | Sign p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        if row["metric"] == "balanced_accuracy":
            sign = row["sign_test_pvalue"]
            lines.append(
                f"| {row['contrast']} | {row['n_folds']} folds / {row['n_writers']} writers | "
                f"{row['mean_difference']:.6f} | "
                f"[{row['hierarchical_bootstrap_ci95_low']:.6f}, "
                f"{row['hierarchical_bootstrap_ci95_high']:.6f}] | "
                f"{'' if sign is None else f'{sign:.6g}'} |"
            )
    if issues:
        lines.extend(["", "## Integrity issues", ""])
        for issue in issues:
            lines.append(f"- `{issue['run_id']}`: {issue['issue']} {issue['detail']}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Chinese-MNIST experiment outputs.")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--analysis-dir", type=Path, default=None)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = analyze_results(args.output_root, analysis_dir=args.analysis_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


__all__ = ["analyze_results", "main"]
