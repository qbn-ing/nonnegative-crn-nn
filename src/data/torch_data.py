from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from utils.seed import dataloader_seed_kwargs
from utils.validation import (
    check_nonnegative_int,
    check_positive_int,
)

from .dataset_spec import DatasetBundle


@dataclass(frozen=True)
class TorchBundle:
    X: torch.Tensor
    y: torch.Tensor
    name: str | None = None
    feature_names: list[str] | None = None
    target_names: list[str] | None = None
    metadata: dict[str, Any] | None = None

    def as_dataset(self) -> TensorDataset:
        return TensorDataset(self.X, self.y)


def to_torch_bundle(
    bundle: DatasetBundle,
    *,
    x_dtype: torch.dtype = torch.float32,
    y_dtype: torch.dtype = torch.long,
    check_nonnegative: bool = True,
) -> TorchBundle:
    if not isinstance(bundle, DatasetBundle):
        raise TypeError("bundle must be a DatasetBundle.")
    X = np.asarray(bundle.X)
    y = np.asarray(bundle.y).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError("Bundle must contain aligned 2D X and 1D y.")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or Inf.")
    if check_nonnegative and float(X.min()) < 0.0:
        raise ValueError("CRN input concentrations must be nonnegative.")
    return TorchBundle(
        X=torch.as_tensor(X, dtype=x_dtype),
        y=torch.as_tensor(y, dtype=y_dtype),
        name=bundle.name,
        feature_names=(
            None
            if bundle.feature_names is None
            else list(bundle.feature_names)
        ),
        target_names=(
            None
            if bundle.target_names is None
            else list(bundle.target_names)
        ),
        metadata=dict(bundle.metadata),
    )


def to_tensor_dataset(
    bundle: DatasetBundle,
    *,
    x_dtype: torch.dtype = torch.float32,
    y_dtype: torch.dtype = torch.long,
    check_nonnegative: bool = True,
) -> TensorDataset:
    return to_torch_bundle(
        bundle,
        x_dtype=x_dtype,
        y_dtype=y_dtype,
        check_nonnegative=check_nonnegative,
    ).as_dataset()


def make_dataloader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool = False,
    seed: int | None = None,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> DataLoader[Any]:
    """Build a reproducible PyTorch DataLoader without experiment policy."""

    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be a torch Dataset.")
    check_positive_int("batch_size", batch_size)
    check_nonnegative_int("num_workers", num_workers)
    for name, value in (
        ("shuffle", shuffle),
        ("drop_last", drop_last),
        ("pin_memory", pin_memory),
        ("persistent_workers", persistent_workers),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool.")
    if persistent_workers and num_workers == 0:
        raise ValueError(
            "persistent_workers requires num_workers > 0."
        )

    seed_kwargs = (
        {}
        if seed is None
        else dataloader_seed_kwargs(seed)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        **seed_kwargs,
    )


def make_bundle_dataloader(
    bundle: DatasetBundle,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int | None = None,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    x_dtype: torch.dtype = torch.float32,
    y_dtype: torch.dtype = torch.long,
    check_nonnegative: bool = True,
) -> DataLoader[Any]:
    return make_dataloader(
        to_tensor_dataset(
            bundle,
            x_dtype=x_dtype,
            y_dtype=y_dtype,
            check_nonnegative=check_nonnegative,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def make_tensor_dataloader(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool = False,
    seed: int | None = None,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
) -> DataLoader[Any]:
    """Convenience wrapper for the common tensor classification case."""

    if not isinstance(inputs, torch.Tensor):
        raise TypeError("inputs must be a torch.Tensor.")
    if not isinstance(targets, torch.Tensor):
        raise TypeError("targets must be a torch.Tensor.")
    if inputs.ndim == 0 or targets.ndim == 0:
        raise ValueError(
            "inputs and targets must have a sample dimension."
        )
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError(
            "inputs and targets must contain the same number of samples."
        )
    return make_dataloader(
        TensorDataset(inputs, targets),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )


__all__ = [
    "TorchBundle",
    "make_bundle_dataloader",
    "make_dataloader",
    "make_tensor_dataloader",
    "to_tensor_dataset",
    "to_torch_bundle",
]
