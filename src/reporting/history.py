from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.loop import EpochResult, FitHistory
from utils.paths import normalize_path
from utils.serialization import to_jsonable, write_json


@dataclass(frozen=True)
class HistoryPaths:
    output_dir: str
    history_json: str
    history_csv: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_dir": self.output_dir,
            "history_json": self.history_json,
            "history_csv": self.history_csv,
        }


def epoch_result_to_dict(result: EpochResult) -> dict[str, Any]:
    if not isinstance(result, EpochResult):
        raise TypeError("result must be an EpochResult.")
    return {
        "n_batches": int(result.n_batches),
        "n_samples": int(result.n_samples),
        "metrics": to_jsonable(result.metrics),
    }


def fit_history_to_dict(history: FitHistory) -> dict[str, Any]:
    if not isinstance(history, FitHistory):
        raise TypeError("history must be a FitHistory.")
    return {
        "epochs_completed": len(history.train),
        "best_epoch": history.best_epoch,
        "stopped_early": history.stopped_early,
        "train": [
            epoch_result_to_dict(result)
            for result in history.train
        ],
        "validation": [
            None
            if result is None
            else epoch_result_to_dict(result)
            for result in history.validation
        ],
        "early_stopping": (
            None
            if history.early_stopping is None
            else history.early_stopping.to_dict()
        ),
    }


def fit_history_to_rows(
    history: FitHistory,
    *,
    metric_keys: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(history, FitHistory):
        raise TypeError("history must be a FitHistory.")
    if metric_keys is not None and isinstance(
        metric_keys,
        (str, bytes),
    ):
        raise TypeError("metric_keys must be a sequence or None.")
    requested = (
        None
        if metric_keys is None
        else tuple(str(key) for key in metric_keys)
    )

    rows: list[dict[str, Any]] = []
    for epoch, train_result in enumerate(history.train, start=1):
        rows.append(
            _epoch_row(
                epoch,
                "train",
                train_result,
                metric_keys=requested,
            )
        )
        if epoch <= len(history.validation):
            validation_result = history.validation[epoch - 1]
            if validation_result is not None:
                rows.append(
                    _epoch_row(
                        epoch,
                        "validation",
                        validation_result,
                        metric_keys=requested,
                    )
                )
    return rows


def save_fit_history(
    history: FitHistory,
    output_dir: str | Path,
    *,
    json_filename: str = "history.json",
    csv_filename: str = "history.csv",
    metric_keys: Sequence[str] | None = None,
) -> HistoryPaths:
    """Save complete JSON history and a flat CSV view."""

    output_path = normalize_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = write_json(
        fit_history_to_dict(history),
        output_path / json_filename,
    )
    csv_path = output_path / csv_filename
    _write_rows(
        fit_history_to_rows(history, metric_keys=metric_keys),
        csv_path,
    )
    return HistoryPaths(
        output_dir=output_path.as_posix(),
        history_json=json_path.as_posix(),
        history_csv=csv_path.as_posix(),
    )


def _epoch_row(
    epoch: int,
    split: str,
    result: EpochResult,
    *,
    metric_keys: tuple[str, ...] | None,
) -> dict[str, Any]:
    metrics: Mapping[str, Any] = result.metrics
    selected = (
        metrics
        if metric_keys is None
        else {
            key: metrics[key]
            for key in metric_keys
            if key in metrics
        }
    )
    row: dict[str, Any] = {
        "epoch": epoch,
        "split": split,
        "n_batches": int(result.n_batches),
        "n_samples": int(result.n_samples),
    }
    row.update(
        {
            str(key): to_jsonable(value)
            for key, value in selected.items()
        }
    )
    return row


def _write_rows(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


__all__ = [
    "HistoryPaths",
    "epoch_result_to_dict",
    "fit_history_to_dict",
    "fit_history_to_rows",
    "save_fit_history",
]
