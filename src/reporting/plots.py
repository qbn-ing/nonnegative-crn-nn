from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from utils.paths import normalize_path
from utils.serialization import to_jsonable, write_json

__all__ = [
    "ConfusionMatrixArtifacts",
    "plot_history_metric",
    "write_confusion_matrix_artifacts",
]


@dataclass(frozen=True)
class ConfusionMatrixArtifacts:
    """Files and values produced for a confusion matrix report."""

    csv_path: str
    png_path: str
    json_path: str | None
    matrix: list[list[int]]
    labels: list[int]
    target_names: list[str] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "csv_path": self.csv_path,
            "png_path": self.png_path,
            "json_path": self.json_path,
            "matrix": self.matrix,
            "labels": self.labels,
            "target_names": self.target_names,
        }


def plot_history_metric(
    history_rows: Sequence[Mapping[str, Any]],
    output_file: str | Path,
    *,
    metric_key: str = "total",
    split_key: str = "split",
    epoch_key: str = "epoch",
    splits: Sequence[str] = ("train", "validation"),
    title: str | None = None,
) -> Path:
    """Plot one scalar metric from flat history rows.

    This function expects rows produced by ``fit_history_to_rows``. It is kept
    independent from the training loop so that experiments can reuse it without
    changing the core trainer.
    """

    if not metric_key:
        raise ValueError("metric_key must be non-empty.")
    if not split_key:
        raise ValueError("split_key must be non-empty.")
    if not epoch_key:
        raise ValueError("epoch_key must be non-empty.")

    rows = _validate_history_rows(history_rows)
    output_path = normalize_path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    series_by_split = _collect_metric_series(
        rows,
        metric_key=metric_key,
        split_key=split_key,
        epoch_key=epoch_key,
        splits=splits,
    )

    if not series_by_split:
        raise ValueError(
            f"No finite metric values found for metric_key={metric_key!r}."
        )

    plt = _import_pyplot()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for split, series in series_by_split.items():
        epochs = [item[0] for item in series]
        values = [item[1] for item in series]
        ax.plot(epochs, values, marker="o", label=split)

    ax.set_xlabel("epoch")
    ax.set_ylabel(metric_key)
    ax.set_title(title or f"{metric_key} curve")
    if len(series_by_split) > 1:
        ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def write_confusion_matrix_artifacts(
    *,
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    output_dir: str | Path,
    labels: Sequence[int] | None = None,
    target_names: Sequence[str] | None = None,
    csv_filename: str = "confusion_matrix.csv",
    png_filename: str = "confusion_matrix.png",
    json_filename: str | None = "confusion_matrix.json",
    title: str = "Confusion matrix",
) -> ConfusionMatrixArtifacts:
    """Write confusion matrix CSV/PNG, optionally with a compact JSON payload."""

    true_array = _as_1d_int_array(y_true, name="y_true")
    pred_array = _as_1d_int_array(y_pred, name="y_pred")

    if true_array.shape[0] != pred_array.shape[0]:
        raise ValueError(
            "y_true and y_pred must have the same length. "
            f"Got {true_array.shape[0]} and {pred_array.shape[0]}."
        )
    if true_array.shape[0] == 0:
        raise ValueError("Cannot build a confusion matrix from empty inputs.")

    label_list = _normalize_labels(labels, true_array, pred_array)
    target_name_list = _normalize_target_names(target_names, label_list)

    matrix = _confusion_matrix(true_array, pred_array, label_list)

    output_path = normalize_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / csv_filename
    png_path = output_path / png_filename
    json_path = None if json_filename is None else output_path / json_filename

    _write_confusion_matrix_csv(
        matrix,
        csv_path,
        labels=label_list,
        target_names=target_name_list,
    )
    _plot_confusion_matrix(
        matrix,
        png_path,
        labels=label_list,
        target_names=target_name_list,
        title=title,
    )

    payload = {
        "matrix": matrix.astype(int).tolist(),
        "labels": [int(item) for item in label_list],
        "target_names": target_name_list,
        "n_samples": int(true_array.shape[0]),
    }

    if json_path is not None:
        write_json(payload, json_path)

    return ConfusionMatrixArtifacts(
        csv_path=csv_path.as_posix(),
        png_path=png_path.as_posix(),
        json_path=None if json_path is None else json_path.as_posix(),
        matrix=payload["matrix"],
        labels=payload["labels"],
        target_names=target_name_list,
    )


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _validate_history_rows(
    history_rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if isinstance(history_rows, (str, bytes)):
        raise TypeError("history_rows must be a sequence of mappings.")

    rows = list(history_rows)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(
                "Every history row must be a mapping. "
                f"Got {type(row).__name__} at index {index}."
            )
    return rows


def _collect_metric_series(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
    split_key: str,
    epoch_key: str,
    splits: Sequence[str],
) -> dict[str, list[tuple[int, float]]]:
    requested_splits = tuple(str(split) for split in splits)
    result: dict[str, list[tuple[int, float]]] = {}

    for split in requested_splits:
        series: list[tuple[int, float]] = []

        for row in rows:
            if str(row.get(split_key)) != split:
                continue
            if metric_key not in row:
                continue

            value = row.get(metric_key)
            if value is None:
                continue

            value_float = float(value)
            if not np.isfinite(value_float):
                continue

            series.append((int(row[epoch_key]), value_float))

        if series:
            series.sort(key=lambda item: item[0])
            result[split] = series

    return result


def _as_1d_int_array(value: Sequence[int] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D. Got shape {array.shape}.")
    return array.astype(np.int64, copy=False)


def _normalize_labels(
    labels: Sequence[int] | None,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[int]:
    if labels is not None:
        label_list = [int(item) for item in labels]
    else:
        label_list = sorted(
            int(item) for item in set(y_true.tolist()) | set(y_pred.tolist())
        )

    if not label_list:
        raise ValueError("labels must not be empty.")

    if len(set(label_list)) != len(label_list):
        raise ValueError(f"labels must be unique. Got {label_list}.")

    return label_list


def _normalize_target_names(
    target_names: Sequence[str] | None,
    labels: Sequence[int],
) -> list[str] | None:
    if target_names is None:
        return None

    names = [str(item) for item in target_names]
    if len(names) != len(labels):
        raise ValueError(
            "target_names length must match labels length. "
            f"Got {len(names)} names and {len(labels)} labels."
        )
    return names


def _confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int],
) -> np.ndarray:
    index = {label: pos for pos, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)

    for expected, predicted in zip(y_true.tolist(), y_pred.tolist(), strict=True):
        if expected not in index or predicted not in index:
            continue
        matrix[index[expected], index[predicted]] += 1

    return matrix


def _write_confusion_matrix_csv(
    matrix: np.ndarray,
    path: Path,
    *,
    labels: Sequence[int],
    target_names: Sequence[str] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    display_names = _display_names(labels, target_names)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["expected\\predicted", *display_names])
        for row_name, row_values in zip(display_names, matrix.tolist(), strict=True):
            writer.writerow([row_name, *[int(value) for value in row_values]])


def _plot_confusion_matrix(
    matrix: np.ndarray,
    path: Path,
    *,
    labels: Sequence[int],
    target_names: Sequence[str] | None,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    display_names = _display_names(labels, target_names)

    plt = _import_pyplot()
    fig = plt.figure(figsize=(6.0, 5.2))
    ax = fig.add_subplot(1, 1, 1)

    image = ax.imshow(matrix, cmap="Blues", interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("expected")
    ax.set_xticks(range(len(display_names)))
    ax.set_yticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=45, ha="right")
    ax.set_yticklabels(display_names)

    max_value = int(matrix.max()) if matrix.size > 0 else 0
    threshold = 0.5 * max_value

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            text_color = "white" if max_value > 0 and value > threshold else "black"
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _display_names(
    labels: Sequence[int],
    target_names: Sequence[str] | None,
) -> list[str]:
    if target_names is not None:
        return [f"{label}:{name}" for label, name in zip(labels, target_names, strict=True)]
    return [str(label) for label in labels]
