from __future__ import annotations

from typing import Any, Sequence

import torch

from training.metrics import (
    classification_metrics as _training_classification_metrics,
)
from training.metrics import labels_from_score


def classification_metrics(
    *,
    y_true: Any,
    y_pred: Any | None = None,
    y_score: Any | None = None,
) -> dict[str, float]:
    """Paper-facing classification metrics with explicit aliases."""

    result = _training_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
    )
    result["accuracy"] = result["acc"]
    result["balanced_accuracy"] = result["recall_macro"]
    return result


def confusion_matrix(
    y_true: Any,
    y_pred: Any,
    *,
    labels: Sequence[int] | None = None,
) -> torch.Tensor:
    truth = torch.as_tensor(y_true).detach().cpu().reshape(-1).to(torch.long)
    prediction = (
        torch.as_tensor(y_pred).detach().cpu().reshape(-1).to(torch.long)
    )
    if truth.numel() != prediction.numel():
        raise ValueError("y_true and y_pred must have the same length.")
    if labels is None:
        label_tensor = torch.unique(
            torch.cat((truth, prediction))
        ).sort().values
    else:
        label_tensor = torch.as_tensor(
            list(labels),
            dtype=torch.long,
        )
    if label_tensor.numel() == 0:
        raise ValueError("At least one label is required.")
    matrix = torch.zeros(
        (label_tensor.numel(), label_tensor.numel()),
        dtype=torch.long,
    )
    index = {int(label): i for i, label in enumerate(label_tensor.tolist())}
    for expected, observed in zip(truth.tolist(), prediction.tolist()):
        if expected not in index or observed not in index:
            raise ValueError("Observed label is absent from labels.")
        matrix[index[expected], index[observed]] += 1
    return matrix


def per_class_recall(
    y_true: Any,
    y_pred: Any,
    *,
    labels: Sequence[int] | None = None,
) -> dict[int, float]:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    if labels is None:
        truth = torch.as_tensor(y_true).reshape(-1).to(torch.long)
        prediction = torch.as_tensor(y_pred).reshape(-1).to(torch.long)
        resolved = torch.unique(
            torch.cat((truth, prediction))
        ).sort().values.tolist()
    else:
        resolved = [int(value) for value in labels]
    totals = matrix.sum(dim=1)
    return {
        label: (
            float(matrix[i, i] / totals[i])
            if int(totals[i]) > 0
            else 0.0
        )
        for i, label in enumerate(resolved)
    }


__all__ = [
    "classification_metrics",
    "confusion_matrix",
    "labels_from_score",
    "per_class_recall",
]
