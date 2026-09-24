from __future__ import annotations

import math
from typing import Any

import torch


def classification_metrics(
    *,
    y_true: Any,
    y_pred: Any | None = None,
    y_score: Any | None = None,
) -> dict[str, float]:
    """Compute classifier metrics with PyTorch only.

    The returned keys match the training loop's established contract:
    ``acc``, macro ``f1``/``precision``/``recall``, and score-based macro
    ``auc``/``aupr`` when class scores are supplied.
    """

    truth = _as_cpu_tensor("y_true", y_true).reshape(-1).to(torch.long)
    if truth.numel() == 0:
        raise ValueError("y_true must not be empty.")

    scores: torch.Tensor | None = None
    if y_score is not None:
        scores = _validate_scores(y_score, n_samples=truth.numel())

    if y_pred is None:
        if scores is None:
            raise ValueError("Classification metrics require y_pred or y_score.")
        prediction = labels_from_score(scores)
    else:
        prediction = _as_cpu_tensor("y_pred", y_pred).reshape(-1).to(torch.long)

    if prediction.numel() != truth.numel():
        raise ValueError(
            "y_true and y_pred must have the same number of samples: "
            f"got {truth.numel()} and {prediction.numel()}."
        )

    labels = torch.unique(torch.cat((truth, prediction))).sort().values
    if labels.numel() == 0:
        raise ValueError("At least one class label is required.")

    precisions: list[float] = []
    recalls: list[float] = []
    f1_values: list[float] = []
    for label in labels:
        true_positive = int(((truth == label) & (prediction == label)).sum())
        predicted_positive = int((prediction == label).sum())
        actual_positive = int((truth == label).sum())
        precision = (
            true_positive / predicted_positive
            if predicted_positive > 0
            else 0.0
        )
        recall = (
            true_positive / actual_positive
            if actual_positive > 0
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append(f1)

    result = {
        "acc": float((truth == prediction).to(torch.float64).mean()),
        "f1_macro": _mean(f1_values),
        "precision_macro": _mean(precisions),
        "recall_macro": _mean(recalls),
    }

    if scores is not None:
        result.update(_classification_score_metrics(truth, scores))
    return result


def labels_from_score(y_score: Any) -> torch.Tensor:
    """Return argmax class indices from a two-dimensional score tensor."""

    scores = _validate_scores(y_score)
    return scores.argmax(dim=1).to(torch.long)


def _classification_score_metrics(
    truth: torch.Tensor,
    scores: torch.Tensor,
) -> dict[str, float]:
    n_classes = int(scores.shape[1])
    classes = torch.unique(truth)
    if classes.numel() < 2:
        return {"auc_macro": math.nan, "aupr_macro": math.nan}
    if bool((classes < 0).any()) or bool((classes >= n_classes).any()):
        raise ValueError(
            "Classification labels must be integer class indices compatible "
            f"with score columns. Got {classes.tolist()} with score shape "
            f"{tuple(scores.shape)}."
        )

    if n_classes == 2:
        binary = truth == 1
        return {
            "auc_macro": _binary_roc_auc(binary, scores[:, 1]),
            "aupr_macro": _binary_average_precision(binary, scores[:, 1]),
        }

    auc_values: list[float] = []
    aupr_values: list[float] = []
    for index in range(n_classes):
        binary = truth == index
        positives = int(binary.sum())
        if positives == 0 or positives == binary.numel():
            continue
        auc_values.append(_binary_roc_auc(binary, scores[:, index]))
        aupr_values.append(_binary_average_precision(binary, scores[:, index]))

    return {
        "auc_macro": _mean(auc_values) if auc_values else math.nan,
        "aupr_macro": _mean(aupr_values) if aupr_values else math.nan,
    }


def _binary_roc_auc(target: torch.Tensor, score: torch.Tensor) -> float:
    """Mann–Whitney ROC AUC with average ranks for tied scores."""

    target = target.to(torch.bool)
    n_positive = int(target.sum())
    n_negative = int(target.numel() - n_positive)
    if n_positive == 0 or n_negative == 0:
        return math.nan

    order = torch.argsort(score, stable=True)
    sorted_score = score[order]
    sorted_target = target[order]
    rank_sum_positive = 0.0
    start = 0
    count = int(score.numel())
    while start < count:
        end = start + 1
        while end < count and bool(sorted_score[end] == sorted_score[start]):
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        rank_sum_positive += average_rank * int(sorted_target[start:end].sum())
        start = end

    minimum_rank_sum = n_positive * (n_positive + 1) / 2.0
    return float(
        (rank_sum_positive - minimum_rank_sum) / (n_positive * n_negative)
    )


def _binary_average_precision(
    target: torch.Tensor,
    score: torch.Tensor,
) -> float:
    """Average precision using threshold groups, including score ties."""

    target = target.to(torch.bool)
    n_positive = int(target.sum())
    if n_positive == 0:
        return math.nan

    order = torch.argsort(score, descending=True, stable=True)
    sorted_score = score[order]
    sorted_target = target[order]
    true_positive = 0
    seen = 0
    average_precision = 0.0
    start = 0
    count = int(score.numel())
    while start < count:
        end = start + 1
        while end < count and bool(sorted_score[end] == sorted_score[start]):
            end += 1
        group_positive = int(sorted_target[start:end].sum())
        true_positive += group_positive
        seen += end - start
        if group_positive:
            precision = true_positive / seen
            recall_increment = group_positive / n_positive
            average_precision += precision * recall_increment
        start = end
    return float(average_precision)


def _validate_scores(
    y_score: Any,
    *,
    n_samples: int | None = None,
) -> torch.Tensor:
    scores = _as_cpu_tensor("y_score", y_score)
    if scores.ndim != 2:
        raise ValueError(
            "Classification y_score must be a 2D tensor with shape "
            f"(n_samples, n_classes). Got shape {tuple(scores.shape)}."
        )
    if scores.shape[1] < 2:
        raise ValueError(
            "Classification y_score must contain at least two class scores. "
            f"Got shape {tuple(scores.shape)}."
        )
    if n_samples is not None and scores.shape[0] != n_samples:
        raise ValueError(
            "y_true and y_score must have the same number of samples: "
            f"got {n_samples} and {scores.shape[0]}."
        )
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("y_score must contain only finite values.")
    return scores.to(torch.float64)


def _as_cpu_tensor(name: str, value: Any) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be tensor-like.") from exc
    return tensor.detach().cpu()


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


__all__ = ["classification_metrics", "labels_from_score"]
