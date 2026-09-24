from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from data import make_tensor_dataloader


@dataclass(frozen=True)
class LoaderBundle:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    generator: torch.Generator


def make_classification_loaders(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_validation: torch.Tensor,
    y_validation: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    batch_size: int,
    batch_seed: int,
) -> LoaderBundle:
    """Build the shared deterministic train/validation/test loader triplet."""

    train = make_tensor_dataloader(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        seed=batch_seed,
        num_workers=0,
    )
    validation = make_tensor_dataloader(
        x_validation,
        y_validation,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test = make_tensor_dataloader(
        x_test,
        y_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    generator = train.generator
    if not isinstance(generator, torch.Generator):
        raise RuntimeError("seeded training DataLoader has no generator.")
    return LoaderBundle(train, validation, test, generator)


__all__ = ["LoaderBundle", "make_classification_loaders"]
