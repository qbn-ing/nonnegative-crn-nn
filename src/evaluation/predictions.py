from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from training.loop import unpack_batch
from training.losses import (
    concentration_probs,
    output_to_concentrations,
)


@dataclass(frozen=True)
class ClassificationPredictions:
    """CPU tensors collected with the same concentration semantics as loss."""

    y_true: torch.Tensor
    y_pred: torch.Tensor
    y_score: torch.Tensor

    @property
    def n_samples(self) -> int:
        return int(self.y_true.shape[0])

    @property
    def n_classes(self) -> int:
        return (
            int(self.y_score.shape[1])
            if self.y_score.ndim == 2
            else 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
            "y_true": self.y_true.tolist(),
            "y_pred": self.y_pred.tolist(),
            "y_score": self.y_score.tolist(),
        }


@torch.no_grad()
def collect_classification_predictions(
    model: torch.nn.Module,
    dataloader: Any,
    *,
    device: torch.device | str | None = None,
    eps: float = 1e-8,
    check_nonnegative: bool = True,
) -> ClassificationPredictions:
    """Collect labels, predicted classes and normalized concentrations."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise TypeError("eps must be a number.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    if not isinstance(check_nonnegative, bool):
        raise TypeError("check_nonnegative must be a bool.")

    resolved_device = _resolve_device(model, device)
    was_training = model.training
    y_true_chunks: list[torch.Tensor] = []
    y_score_chunks: list[torch.Tensor] = []

    try:
        model.eval()
        if device is not None:
            model.to(resolved_device)
        for batch in dataloader:
            inputs, target = unpack_batch(batch)
            inputs = _move_to_device(inputs, resolved_device)
            target = _move_to_device(target, resolved_device)
            output = model(inputs)
            concentrations = output_to_concentrations(output)
            probabilities = concentration_probs(
                concentrations,
                eps=float(eps),
                check_nonnegative=check_nonnegative,
            )
            y_true_chunks.append(
                target.detach().cpu().reshape(-1)
            )
            y_score_chunks.append(
                probabilities.detach().cpu()
            )
    finally:
        model.train(was_training)

    if not y_true_chunks:
        raise ValueError(
            "Cannot collect predictions from an empty dataloader."
        )

    y_true = torch.cat(y_true_chunks, dim=0)
    y_score = torch.cat(y_score_chunks, dim=0)
    if y_score.ndim != 2:
        raise ValueError(
            "Class scores must have shape (n_samples, n_classes)."
        )
    return ClassificationPredictions(
        y_true=y_true,
        y_pred=torch.argmax(y_score, dim=1),
        y_score=y_score,
    )


def _resolve_device(
    model: torch.nn.Module,
    device: torch.device | str | None,
) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {
            key: _move_to_device(item, device)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _move_to_device(item, device) for item in value
        )
    if isinstance(value, list):
        return [
            _move_to_device(item, device) for item in value
        ]
    return value


__all__ = [
    "ClassificationPredictions",
    "collect_classification_predictions",
]
