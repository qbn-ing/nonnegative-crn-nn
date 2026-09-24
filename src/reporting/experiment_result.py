from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.loop import EpochResult, FitHistory
from utils.paths import normalize_path
from utils.serialization import to_jsonable, write_json

__all__ = [
    "DEFAULT_COMPACT_METRIC_KEYS",
    "ExperimentOutputPaths",
    "compact_count_report",
    "compact_epoch_result",
    "epoch_result_to_dict",
    "fit_history_to_dict",
    "fit_history_to_rows",
    "make_experiment_summary",
    "make_summary_for_level",
    "make_compact_experiment_summary",
    "SUMMARY_LEVELS",
    "validate_summary_level",
    "to_jsonable",
    "write_csv_rows",
    "write_json",
]

DEFAULT_COMPACT_METRIC_KEYS: tuple[str, ...] = (
    "total",
    "classification",
    "regularization",
    "accuracy",
    "acc",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "auc_macro",
    "aupr_macro",
    "true_prob_mean",
    "true_prob_min",
    "predicted_margin_mean",
    "predicted_margin_min",
    "true_margin_mean",
    "entropy_mean",
    "z_sum_mean",
    "z_sum_min",
    "has_negative_z",
    "has_nonfinite_z",
    "grad_global_l2_norm",
    "grad_max_abs_grad",
    "grad_mean_abs_grad",
    "grad_small_grad_fraction",
    "grad_large_grad_fraction",
    "grad_zero_grad_fraction",
    "grad_has_nan_or_inf",
    "grad_has_missing_grad",
    "pre_clip_grad_l2_norm",
    "constraint_projection_trigger_rate",
    "constraint_projected_elements_mean",
    "constraint_projected_elements_max",
    "constraint_managed_elements",
    "constraint_most_negative_pre_projection",
)


@dataclass(frozen=True)
class ExperimentOutputPaths:
    """Paths produced by experiment reporting utilities."""

    output_dir: str
    summary_json: str | None = None
    history_csv: str | None = None
    full_result_json: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "output_dir": self.output_dir,
            "summary_json": self.summary_json,
            "history_csv": self.history_csv,
            "full_result_json": self.full_result_json,
        }


def epoch_result_to_dict(result: EpochResult) -> dict[str, Any]:
    """Serialize an EpochResult without dropping any metric."""

    if not isinstance(result, EpochResult):
        raise TypeError(
            "result must be an EpochResult. "
            f"Got {type(result).__name__}."
        )

    return {
        "n_batches": int(result.n_batches),
        "n_samples": int(result.n_samples),
        "metrics": to_jsonable(dict(result.metrics)),
    }


def compact_epoch_result(
    result: EpochResult,
    *,
    metric_keys: Sequence[str] = DEFAULT_COMPACT_METRIC_KEYS,
) -> dict[str, Any]:
    """Serialize an EpochResult using a compact metric subset.

    If both ``accuracy`` and ``acc`` are requested and exist, ``accuracy`` is
    kept and ``acc`` is omitted to reduce duplicate fields.
    """

    if not isinstance(result, EpochResult):
        raise TypeError(
            "result must be an EpochResult. "
            f"Got {type(result).__name__}."
        )

    requested_keys = _normalize_metric_keys(metric_keys)

    metrics = dict(result.metrics)
    selected: dict[str, Any] = {}

    for key in requested_keys:
        if key == "acc" and "accuracy" in metrics:
            continue

        if key in metrics:
            selected[key] = to_jsonable(metrics[key])

    requested_accuracy = "accuracy" in requested_keys or "acc" in requested_keys

    if requested_accuracy and "accuracy" not in selected:
        if "accuracy" in metrics:
            selected["accuracy"] = to_jsonable(metrics["accuracy"])
        elif "acc" in metrics:
            selected["accuracy"] = to_jsonable(metrics["acc"])

    return {
        "n_batches": int(result.n_batches),
        "n_samples": int(result.n_samples),
        "metrics": selected,
    }


def fit_history_to_dict(history: FitHistory) -> dict[str, list[dict[str, Any] | None]]:
    """Serialize full training and validation history."""

    if not isinstance(history, FitHistory):
        raise TypeError(
            "history must be a FitHistory. "
            f"Got {type(history).__name__}."
        )

    return {
        "train": [epoch_result_to_dict(item) for item in history.train],
        "validation": [
            None if item is None else epoch_result_to_dict(item)
            for item in history.validation
        ],
    }


def fit_history_to_rows(
    history: FitHistory,
    *,
    metric_keys: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert FitHistory into flat rows suitable for CSV output.

    Each row corresponds to one split in one epoch:

    - epoch
    - split: train or validation
    - n_batches
    - n_samples
    - metric columns
    """

    if not isinstance(history, FitHistory):
        raise TypeError(
            "history must be a FitHistory. "
            f"Got {type(history).__name__}."
        )

    rows: list[dict[str, Any]] = []

    for epoch_index, train_result in enumerate(history.train, start=1):
        rows.append(
            _epoch_result_to_row(
                train_result,
                epoch=epoch_index,
                split="train",
                metric_keys=metric_keys,
            )
        )

        if epoch_index <= len(history.validation):
            validation_result = history.validation[epoch_index - 1]
            if validation_result is not None:
                rows.append(
                    _epoch_result_to_row(
                        validation_result,
                        epoch=epoch_index,
                        split="validation",
                        metric_keys=metric_keys,
                    )
                )

    return rows


def compact_count_report(count_report: Any) -> dict[str, Any]:
    """Extract the human-readable part of a model count report.

    Accepts either a ``ModelCountReport``-like object exposing ``to_dict()``,
    or an already serialized mapping.
    """

    payload = _to_mapping_payload(count_report, name="count_report")

    root = payload.get("root")
    if not isinstance(root, Mapping):
        raise TypeError("count_report must contain a mapping field named 'root'.")

    compact_root_keys = (
        "type",
        "name",
        "input_dim",
        "n_basis",
        "depth",
        "n_outputs",
        "n_classes",
        "widths",
        "hidden_widths",
        "n_species",
        "n_evidence_species",
        "n_response_species",
        "n_output_species",
        "n_reactions",
        "n_rational_reactions",
        "n_output_reactions",
    )

    compact = {
        "root": {
            key: to_jsonable(root[key])
            for key in compact_root_keys
            if key in root
        }
    }

    for key in (
        "total_parameter_scalars",
        "trainable_parameter_scalars",
        "frozen_parameter_scalars",
        "n_nodes",
    ):
        if key in payload:
            compact[key] = to_jsonable(payload[key])

    return compact


def make_experiment_summary(
    *,
    config: Mapping[str, Any],
    train_final: EpochResult,
    validation_final: EpochResult | None,
    test_result: EpochResult,
    count_report: Any | None = None,
    extra: Mapping[str, Any] | None = None,
    metric_keys: Sequence[str] = DEFAULT_COMPACT_METRIC_KEYS,
) -> dict[str, Any]:
    """Build a compact, human-readable experiment summary."""

    if not isinstance(config, Mapping):
        raise TypeError(
            "config must be a mapping. "
            f"Got {type(config).__name__}."
        )

    summary: dict[str, Any] = {
        "config": to_jsonable(config),
        "train_final": compact_epoch_result(
            train_final,
            metric_keys=metric_keys,
        ),
        "validation_final": (
            None
            if validation_final is None
            else compact_epoch_result(
                validation_final,
                metric_keys=metric_keys,
            )
        ),
        "test": compact_epoch_result(
            test_result,
            metric_keys=metric_keys,
        ),
    }

    if count_report is not None:
        summary["count_report"] = compact_count_report(count_report)

    if extra is not None:
        if not isinstance(extra, Mapping):
            raise TypeError(
                "extra must be a mapping or None. "
                f"Got {type(extra).__name__}."
            )
        summary["extra"] = to_jsonable(extra)

    return summary



SUMMARY_LEVELS = {"compact", "standard", "full"}
COMPACT_SUMMARY_METRIC_KEYS: tuple[str, ...] = (
    "total",
    "classification",
    "regularization",
    "accuracy",
    "acc",
    "f1_macro",
    "auc_macro",
    "aupr_macro",
    "true_prob_mean",
    "predicted_margin_mean",
)


def validate_summary_level(level: str) -> str:
    if level not in SUMMARY_LEVELS:
        raise ValueError(
            "summary_level must be one of "
            f"{sorted(SUMMARY_LEVELS)}. Got {level!r}."
        )
    return level


def make_summary_for_level(
    summary: Mapping[str, Any],
    *,
    level: str,
) -> dict[str, Any]:
    validate_summary_level(level)
    standard = to_jsonable(summary)
    if level == "full":
        standard["summary_level"] = "full"
        return standard
    if level == "standard":
        standard["summary_level"] = "standard"
        return _drop_none(standard)
    return _make_compact_summary(standard)


def _make_compact_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"summary_level": "compact"}

    config = summary.get("config")
    if isinstance(config, Mapping):
        for key in (
            "dataset",
            "seed",
            "variant",
            "epochs",
            "batch_size",
            "lr",
            "hidden_per_class",
            "basis_degree",
            "include_interactions",
            "enable_input_gate",
            "enable_feature_gate",
            "gate_l1_weight",
            "hard_prune",
            "prune_mode",
            "prune_retrain_epochs",
            "prune_input_threshold",
            "prune_basis_threshold",
            "prune_evidence_threshold",
            "early_stopping",
            "prune_early_stopping",
            "prune_early_stopping_scope",
        ):
            _copy_scalar(row, key, config.get(key))

    extra = summary.get("extra")
    if isinstance(extra, Mapping):
        dataset = extra.get("dataset")
        if isinstance(dataset, Mapping):
            _copy_scalar(row, "dataset", dataset.get("name"))
            _copy_scalar(row, "dataset_task", dataset.get("task"))
            _copy_scalar(row, "n_samples", dataset.get("n_samples"))
            _copy_scalar(row, "n_features", dataset.get("n_features"))
            _copy_scalar(row, "n_classes", dataset.get("n_classes"))

        _copy_scalar(row, "history_epochs", extra.get("history_epochs"))

        early = extra.get("early_stopping")
        if isinstance(early, Mapping):
            for key in ("enabled", "stopped_early", "best_epoch", "epochs_run"):
                _copy_scalar(row, f"early_stopping_{key}", early.get(key))

        pruning = extra.get("hard_pruning")
        if isinstance(pruning, Mapping):
            _summarize_pruning(row, "prune", pruning)
            _summarize_pruning_component_totals(row, "prune", pruning)

        trace = extra.get("layerwise_pruning_trace")
        if isinstance(trace, list):
            row["layerwise_n_stages"] = len(trace)
            for item in trace:
                if not isinstance(item, Mapping):
                    continue
                stage = item.get("stage")
                if not isinstance(stage, str):
                    continue
                prefix = f"stage_{stage}"
                _copy_scalar(row, f"{prefix}_retrain_epochs", item.get("retrain_epochs"))
                _copy_epoch_metrics(row, f"{prefix}_test_after_prune", item.get("test_after_prune"))
                _copy_epoch_metrics(row, f"{prefix}_test_after_retrain", item.get("test_after_retrain"))
                pruning_info = item.get("pruning")
                if isinstance(pruning_info, Mapping):
                    _summarize_pruning(row, prefix, pruning_info)
                    _summarize_pruning_component_totals(row, prefix, pruning_info)
            _summarize_layerwise_pruning_run(row, trace)

        before = extra.get("test_before_pruning")
        if isinstance(before, Mapping):
            _copy_epoch_metrics(row, "test_before_pruning", before)

    count_report = summary.get("count_report")
    if isinstance(count_report, Mapping):
        root = count_report.get("root")
        if isinstance(root, Mapping):
            _copy_scalar(row, "name", root.get("name"))
            _copy_scalar(row, "model_type", root.get("type"))
            for key in (
                "input_dim",
                "n_basis",
                "depth",
                "n_outputs",
                "n_classes",
                "n_species",
                "n_reactions",
            ):
                _copy_scalar(row, f"count_{key}", root.get(key))
            _copy_jsonable(row, "count_widths", root.get("widths"))
            _copy_jsonable(row, "count_hidden_widths", root.get("hidden_widths"))

    crn_export = summary.get("crn_export")
    if isinstance(crn_export, Mapping):
        _copy_scalar(row, "export_n_species", crn_export.get("species_count"))
        _copy_scalar(row, "export_n_reactions", crn_export.get("reaction_count"))
    else:
        # Fallback for in-memory summaries built before CRN export artifacts are
        # written.  The output writer overwrites these fields from crn_export
        # once export is available.
        if isinstance(count_report, Mapping):
            root = count_report.get("root")
            if isinstance(root, Mapping):
                _copy_scalar(row, "export_n_species", root.get("n_species"))
                _copy_scalar(row, "export_n_reactions", root.get("n_reactions"))

    _copy_epoch_metrics(row, "train", summary.get("train_final"))
    _copy_epoch_metrics(row, "val", summary.get("validation_final"))
    _copy_epoch_metrics(row, "test", summary.get("test"))
    return _drop_none(row)


def _copy_epoch_metrics(row: dict[str, Any], prefix: str, payload: Any) -> None:
    if not isinstance(payload, Mapping):
        return
    _copy_scalar(row, f"{prefix}_n_samples", payload.get("n_samples"))
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        return
    for key in COMPACT_SUMMARY_METRIC_KEYS:
        _copy_scalar(row, f"{prefix}_{key}", metrics.get(key))


def _summarize_pruning(row: dict[str, Any], prefix: str, pruning: Mapping[str, Any]) -> None:
    for key, value in pruning.items():
        if key == "components":
            continue
        _copy_scalar(row, f"{prefix}_{key}", value)
    components = pruning.get("components")
    if not isinstance(components, Mapping):
        return
    for component_name, component in components.items():
        if not isinstance(component, Mapping):
            continue
        for key in ("n_pruned", "n_before", "n_after"):
            _copy_scalar(row, f"{prefix}_{component_name}_{key}", component.get(key))
        after = component.get("after")
        if isinstance(after, Mapping):
            for key in ("input_dim", "n_basis", "n_evidence_species", "n_classes"):
                _copy_scalar(row, f"{prefix}_{component_name}_after_{key}", after.get(key))


def _summarize_pruning_component_totals(
    row: dict[str, Any],
    prefix: str,
    pruning: Mapping[str, Any],
) -> None:
    before = pruning.get("before")
    after = pruning.get("after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return

    before_counts = _active_component_counts(before)
    after_counts = _active_component_counts(after)

    for component in ("input", "basis", "evidence", "total"):
        before_value = before_counts.get(component)
        after_value = after_counts.get(component)
        _copy_scalar(row, f"{prefix}_before_{component}_components", before_value)
        _copy_scalar(row, f"{prefix}_after_{component}_components", after_value)
        if isinstance(before_value, int) and isinstance(after_value, int):
            _copy_scalar(
                row,
                f"{prefix}_removed_{component}_components",
                before_value - after_value,
            )


def _summarize_layerwise_pruning_run(
    row: dict[str, Any],
    trace: Sequence[Any],
) -> None:
    """Summarize the whole layerwise run from the stage trace.

    After layerwise pruning the final all-stage hard-pruning report is often a
    no-op because the model has already been pruned.  In that case its
    before/after counts are equal, so top-level ``prune_*`` fields must be
    derived from the first stage's pre-prune state and the last stage's
    post-prune state instead of from the final no-op report.
    """

    reports: list[Mapping[str, Any]] = []
    for item in trace:
        if not isinstance(item, Mapping):
            continue
        pruning = item.get("pruning")
        if not isinstance(pruning, Mapping):
            continue
        if isinstance(pruning.get("before"), Mapping) and isinstance(
            pruning.get("after"), Mapping
        ):
            reports.append(pruning)

    if not reports:
        return

    first_before = reports[0].get("before")
    last_after = reports[-1].get("after")
    if not isinstance(first_before, Mapping) or not isinstance(last_after, Mapping):
        return

    before_counts = _active_component_counts(first_before)
    after_counts = _active_component_counts(last_after)
    total_removed = 0
    for component in ("input", "basis", "evidence", "total"):
        before_value = before_counts.get(component)
        after_value = after_counts.get(component)
        _copy_scalar(row, f"prune_before_{component}_components", before_value)
        _copy_scalar(row, f"prune_after_{component}_components", after_value)
        if isinstance(before_value, int) and isinstance(after_value, int):
            removed = before_value - after_value
            _copy_scalar(row, f"prune_removed_{component}_components", removed)
            if component == "total":
                total_removed = int(removed)

    if total_removed > 0:
        row["prune_applied"] = True


def _active_component_counts(snapshot: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}

    input_gate = snapshot.get("input_gate")
    if isinstance(input_gate, Mapping):
        active = input_gate.get("active")
        if isinstance(active, int):
            counts["input"] = active

    basis_gate = snapshot.get("basis_gate")
    if isinstance(basis_gate, Mapping):
        active = basis_gate.get("active")
        if isinstance(active, int):
            counts["basis"] = active

    evidence_edges = snapshot.get("evidence_edges")
    if isinstance(evidence_edges, Mapping):
        active = evidence_edges.get("active")
        if isinstance(active, int):
            counts["evidence"] = active

    total = sum(value for value in counts.values() if isinstance(value, int))
    if counts:
        counts["total"] = int(total)
    return counts


def _copy_scalar(row: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (str, int, float, bool)):
        row[key] = value


def _copy_jsonable(row: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    row[key] = to_jsonable(value)


def _drop_none(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}

def write_csv_rows(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> Path:
    """Write flat dictionaries to a CSV file.

    The field order is stable: first-seen keys define the column order.
    """

    if isinstance(rows, (str, bytes)):
        raise TypeError("rows must be a sequence of mappings.")

    output_path = normalize_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = _collect_fieldnames(rows)

    if not fieldnames:
        output_path.write_text("", encoding="utf-8")
        return output_path

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: to_jsonable(row.get(key)) for key in fieldnames})

    return output_path


def _epoch_result_to_row(
    result: EpochResult,
    *,
    epoch: int,
    split: str,
    metric_keys: Sequence[str] | None,
) -> dict[str, Any]:
    if not isinstance(result, EpochResult):
        raise TypeError(
            "result must be an EpochResult. "
            f"Got {type(result).__name__}."
        )

    metrics = dict(result.metrics)

    if metric_keys is not None:
        requested_keys = _normalize_metric_keys(metric_keys)
        metrics = {key: metrics[key] for key in requested_keys if key in metrics}

    row: dict[str, Any] = {
        "epoch": int(epoch),
        "split": split,
        "n_batches": int(result.n_batches),
        "n_samples": int(result.n_samples),
    }

    for key, value in metrics.items():
        row[str(key)] = to_jsonable(value)

    return row


def _collect_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fieldnames: list[str] = []

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(
                "Every row must be a mapping. "
                f"Got {type(row).__name__} at index {index}."
            )

        for key in row.keys():
            key = str(key)
            if key not in fieldnames:
                fieldnames.append(key)

    return fieldnames


def _to_mapping_payload(value: Any, *, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload

    raise TypeError(
        f"{name} must be a mapping or expose to_dict() returning a mapping. "
        f"Got {type(value).__name__}."
    )


def _normalize_metric_keys(metric_keys: Sequence[str]) -> tuple[str, ...]:
    if isinstance(metric_keys, (str, bytes)):
        raise TypeError("metric_keys must be a sequence of metric names.")

    normalized = tuple(metric_keys)

    for key in normalized:
        if not isinstance(key, str):
            raise TypeError(
                "Every metric key must be a str. "
                f"Got {type(key).__name__}."
            )

    return normalized
