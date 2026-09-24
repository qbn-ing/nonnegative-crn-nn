from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .protocol import ChebyshevTask, RunSpec


@dataclass(frozen=True)
class DatasetPack:
    """Fixed raw-coordinate splits shared by all four paired conditions."""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    x_grid: torch.Tensor
    y_grid: torch.Tensor
    grid_shape: tuple[int, int]
    dataset_seed: int
    fingerprint: str

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_seed": self.dataset_seed,
            "fingerprint": self.fingerprint,
            "train_samples": int(self.x_train.shape[0]),
            "val_samples": int(self.x_val.shape[0]),
            "test_samples": int(self.x_test.shape[0]),
            "grid_samples": int(self.x_grid.shape[0]),
            "grid_shape": self.grid_shape,
            "train_class_counts": _class_counts(self.y_train),
            "val_class_counts": _class_counts(self.y_val),
            "test_class_counts": _class_counts(self.y_test),
            "grid_class_counts": _class_counts(self.y_grid),
        }


def chebyshev_value(x01: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluate T_m after mapping a concentration from [0, 1] to [-1, 1]."""

    if degree < 0:
        raise ValueError("degree must be nonnegative.")
    z = 2.0 * x01 - 1.0
    if degree == 0:
        return torch.ones_like(z)
    if degree == 1:
        return z
    previous = torch.ones_like(z)
    current = z
    for _ in range(2, degree + 1):
        previous, current = current, 2.0 * z * current - previous
    return current


def score_from_raw(x_raw: torch.Tensor, degree: int) -> torch.Tensor:
    """Signed parity score; class 1 occupies opposite-sign Chebyshev cells."""

    _validate_raw_inputs(x_raw)
    first = chebyshev_value(x_raw[:, 0], degree)
    second = chebyshev_value(x_raw[:, 1], degree)
    return -(first * second)


def hard_labels(x_raw: torch.Tensor, degree: int) -> torch.Tensor:
    return (score_from_raw(x_raw, degree) >= 0.0).to(torch.long)


def soft_targets(
    x_raw: torch.Tensor,
    *,
    degree: int,
    beta: float,
) -> torch.Tensor:
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    probability_one = torch.sigmoid(beta * score_from_raw(x_raw, degree))
    return torch.stack((1.0 - probability_one, probability_one), dim=1)


def hard_targets(y: torch.Tensor, *, n_classes: int = 2) -> torch.Tensor:
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    return F.one_hot(y.to(torch.long), num_classes=n_classes).to(torch.float32)


def encode_inputs(x_raw: torch.Tensor, task: ChebyshevTask) -> torch.Tensor:
    """Return the two raw coordinates used by the formal experiment.

    Trainable affine offsets belong to every E/I evidence map and are model
    parameters.  They are not appended input coordinates.  This function
    therefore performs no complement expansion and adds no fixed-one feature.
    """

    _validate_raw_inputs(x_raw)
    if task.input_encoding == "raw":
        return x_raw
    raise ValueError(f"unsupported input encoding {task.input_encoding!r}.")


def encoded_input_dim(task: ChebyshevTask) -> int:
    if task.input_encoding == "raw":
        return 2
    raise ValueError(f"unsupported input encoding {task.input_encoding!r}.")


def make_dataset(spec: RunSpec) -> DatasetPack:
    """Generate task-fixed data independent of model seed and I/C condition."""

    rng = np.random.default_rng(spec.dataset_seed)
    train = rng.random((spec.train_samples, 2), dtype=np.float32)
    validation = rng.random((spec.val_samples, 2), dtype=np.float32)
    test = rng.random((spec.test_samples, 2), dtype=np.float32)
    axis = np.linspace(0.0, 1.0, spec.eval_grid_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
    grid = np.stack((grid_x.reshape(-1), grid_y.reshape(-1)), axis=1)

    x_train = torch.from_numpy(train)
    x_val = torch.from_numpy(validation)
    x_test = torch.from_numpy(test)
    x_grid = torch.from_numpy(grid.astype(np.float32, copy=False))
    y_train = hard_labels(x_train, spec.task.degree)
    y_val = hard_labels(x_val, spec.task.degree)
    y_test = hard_labels(x_test, spec.task.degree)
    y_grid = hard_labels(x_grid, spec.task.degree)
    fingerprint = _fingerprint(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    )
    return DatasetPack(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        x_grid=x_grid,
        y_grid=y_grid,
        grid_shape=(spec.eval_grid_size, spec.eval_grid_size),
        dataset_seed=spec.dataset_seed,
        fingerprint=fingerprint,
    )


def edge_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    grid_shape: tuple[int, int],
    tolerance_cells: int,
) -> dict[str, float | int]:
    """Compare predicted and target decision-boundary edges on a 2D grid."""

    if tolerance_cells < 0:
        raise ValueError("tolerance_cells must be nonnegative.")
    pred = np.asarray(predicted).reshape(grid_shape).astype(np.int64)
    true = np.asarray(target).reshape(grid_shape).astype(np.int64)
    predicted_edges: list[np.ndarray] = []
    target_edges: list[np.ndarray] = []
    for axis in (0, 1):
        left = [slice(None), slice(None)]
        right = [slice(None), slice(None)]
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        predicted_edges.append(pred[tuple(left)] != pred[tuple(right)])
        target_edges.append(true[tuple(left)] != true[tuple(right)])

    predicted_count = sum(int(mask.sum()) for mask in predicted_edges)
    target_count = sum(int(mask.sum()) for mask in target_edges)
    recall_hits = 0
    precision_hits = 0
    for predicted_mask, target_mask in zip(predicted_edges, target_edges):
        predicted_neighborhood = _dilate(predicted_mask, tolerance_cells)
        target_neighborhood = _dilate(target_mask, tolerance_cells)
        recall_hits += int((target_mask & predicted_neighborhood).sum())
        precision_hits += int((predicted_mask & target_neighborhood).sum())
    return {
        "target_switches": target_count,
        "predicted_switches": predicted_count,
        "edge_recall": recall_hits / max(1, target_count),
        "edge_precision": precision_hits / max(1, predicted_count),
        "edge_tolerance_cells": tolerance_cells,
    }


def _dilate(mask: np.ndarray, tolerance_cells: int) -> np.ndarray:
    if tolerance_cells == 0:
        return mask.astype(bool)
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    kernel = 2 * tolerance_cells + 1
    dilated = F.max_pool2d(
        tensor,
        kernel_size=kernel,
        stride=1,
        padding=tolerance_cells,
    )
    return dilated[0, 0].numpy() > 0.5


def _validate_raw_inputs(x_raw: torch.Tensor) -> None:
    if not isinstance(x_raw, torch.Tensor):
        raise TypeError("x_raw must be a torch.Tensor.")
    if x_raw.ndim != 2 or x_raw.shape[1] != 2:
        raise ValueError("x_raw must have shape (n_samples, 2).")
    if not torch.all(torch.isfinite(x_raw)).item():
        raise ValueError("x_raw must contain finite values.")
    if torch.any(x_raw < 0.0).item() or torch.any(x_raw > 1.0).item():
        raise ValueError("x_raw must lie in [0, 1].")


def _fingerprint(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _class_counts(y: torch.Tensor) -> dict[str, int]:
    counts = torch.bincount(y.to(torch.long), minlength=2)
    return {str(index): int(value) for index, value in enumerate(counts)}


__all__ = [
    "DatasetPack",
    "chebyshev_value",
    "edge_metrics",
    "encode_inputs",
    "encoded_input_dim",
    "hard_labels",
    "hard_targets",
    "make_dataset",
    "score_from_raw",
    "soft_targets",
]
