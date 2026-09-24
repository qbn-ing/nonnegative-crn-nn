from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .dataset_spec import DatasetBundle, dataset_fingerprint


@dataclass(frozen=True)
class DatasetSplits:
    train: DatasetBundle
    validation: DatasetBundle
    test: DatasetBundle
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    seed: int
    source_fingerprint: str

    @property
    def val(self) -> DatasetBundle:
        return self.validation

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "seed": self.seed,
            "source_fingerprint": self.source_fingerprint,
            "train_indices": self.train_indices.tolist(),
            "validation_indices": self.validation_indices.tolist(),
            "test_indices": self.test_indices.tolist(),
        }


@dataclass(frozen=True)
class DatasetFold:
    train: DatasetBundle
    validation: DatasetBundle
    repeat_id: int
    fold_id: int
    seed: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    source_fingerprint: str

    @property
    def val(self) -> DatasetBundle:
        return self.validation

    @property
    def val_indices(self) -> np.ndarray:
        return self.validation_indices

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "repeat_id": self.repeat_id,
            "fold_id": self.fold_id,
            "seed": self.seed,
            "source_fingerprint": self.source_fingerprint,
            "train_indices": self.train_indices.tolist(),
            "validation_indices": self.validation_indices.tolist(),
        }


def split_train_val_test(
    bundle: DatasetBundle,
    *,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 0,
    stratify: bool = True,
) -> DatasetSplits:
    """Create disjoint train/validation/test splits and retain exact indices."""

    _validate_bundle(bundle)
    _validate_sizes(train_size, val_size, test_size)
    seed = _check_seed(random_state)
    train_test_split = _sklearn_model_selection().train_test_split
    indices = np.arange(bundle.n_samples, dtype=np.int64)
    labels = np.asarray(bundle.y).reshape(-1) if stratify else None
    try:
        train_indices, temp_indices = train_test_split(
            indices,
            train_size=train_size,
            random_state=seed,
            shuffle=True,
            stratify=labels,
        )
        temp_labels = labels[temp_indices] if labels is not None else None
        validation_indices, test_indices = train_test_split(
            temp_indices,
            train_size=val_size / (val_size + test_size),
            random_state=seed + 1,
            shuffle=True,
            stratify=temp_labels,
        )
    except ValueError as exc:
        raise ValueError(
            "Could not create the requested stratified split. "
            "Check per-class sample counts or set stratify=False."
        ) from exc
    train_indices = _indices(train_indices)
    validation_indices = _indices(validation_indices)
    test_indices = _indices(test_indices)
    _assert_partition(
        bundle.n_samples,
        train_indices,
        validation_indices,
        test_indices,
    )
    fingerprint = dataset_fingerprint(bundle)
    return DatasetSplits(
        train=_subset(bundle, train_indices, "train"),
        validation=_subset(bundle, validation_indices, "validation"),
        test=_subset(bundle, test_indices, "test"),
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        seed=seed,
        source_fingerprint=fingerprint,
    )


def make_kfold_splits(
    bundle: DatasetBundle,
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 0,
    stratify: bool = True,
    repeat_id: int = 0,
) -> list[DatasetFold]:
    """Create one reproducible outer cross-validation repeat."""

    _validate_bundle(bundle)
    if isinstance(n_splits, bool) or not isinstance(n_splits, int):
        raise TypeError("n_splits must be an int.")
    if n_splits < 2 or n_splits > bundle.n_samples:
        raise ValueError("n_splits must lie in [2, n_samples].")
    if not isinstance(shuffle, bool) or not isinstance(stratify, bool):
        raise TypeError("shuffle and stratify must be bools.")
    seed = _check_seed(random_state)
    model_selection = _sklearn_model_selection()
    X = np.asarray(bundle.X)
    y = np.asarray(bundle.y).reshape(-1)
    if stratify:
        splitter = model_selection.StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=seed if shuffle else None,
        )
        iterator = splitter.split(X, y)
    else:
        splitter = model_selection.KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=seed if shuffle else None,
        )
        iterator = splitter.split(X)
    fingerprint = dataset_fingerprint(bundle)
    folds: list[DatasetFold] = []
    try:
        for fold_id, (train_indices, validation_indices) in enumerate(
            iterator
        ):
            train = _indices(train_indices)
            validation = _indices(validation_indices)
            _assert_disjoint(train, validation)
            folds.append(
                DatasetFold(
                    train=_subset(
                        bundle,
                        train,
                        f"repeat_{repeat_id}_fold_{fold_id}_train",
                    ),
                    validation=_subset(
                        bundle,
                        validation,
                        f"repeat_{repeat_id}_fold_{fold_id}_validation",
                    ),
                    repeat_id=int(repeat_id),
                    fold_id=fold_id,
                    seed=seed,
                    train_indices=train,
                    validation_indices=validation,
                    source_fingerprint=fingerprint,
                )
            )
    except ValueError as exc:
        raise ValueError(
            "Could not create stratified folds; reduce n_splits or inspect "
            "per-class sample counts."
        ) from exc
    return folds


def make_repeated_kfold_splits(
    bundle: DatasetBundle,
    *,
    n_splits: int = 5,
    n_repeats: int = 3,
    base_seed: int = 0,
    stratify: bool = True,
) -> list[DatasetFold]:
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, int):
        raise TypeError("n_repeats must be an int.")
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive.")
    seed = _check_seed(base_seed)
    result: list[DatasetFold] = []
    for repeat_id in range(n_repeats):
        result.extend(
            make_kfold_splits(
                bundle,
                n_splits=n_splits,
                random_state=seed + repeat_id,
                stratify=stratify,
                repeat_id=repeat_id,
            )
        )
    return result


def _sklearn_model_selection() -> Any:
    try:
        return importlib.import_module("sklearn.model_selection")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("sklearn"):
            raise ImportError(
                "Dataset splitting requires scikit-learn."
            ) from exc
        raise


def _validate_bundle(bundle: DatasetBundle) -> None:
    if not isinstance(bundle, DatasetBundle):
        raise TypeError("bundle must be a DatasetBundle.")
    X = np.asarray(bundle.X)
    y = np.asarray(bundle.y)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("Bundle must contain aligned 2D X and 1D y.")


def _validate_sizes(train: float, validation: float, test: float) -> None:
    values = (train, validation, test)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        raise TypeError("Split sizes must be numbers.")
    if any(value <= 0.0 or value >= 1.0 for value in values):
        raise ValueError("Each split size must lie strictly inside (0, 1).")
    if not np.isclose(sum(values), 1.0):
        raise ValueError("Split sizes must sum to 1.")


def _check_seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("random_state/base_seed must be an int.")
    if value < 0:
        raise ValueError("random_state/base_seed must be nonnegative.")
    return value


def _indices(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.int64).reshape(-1)


def _assert_disjoint(*parts: np.ndarray) -> None:
    concatenated = np.concatenate(parts)
    if np.unique(concatenated).size != concatenated.size:
        raise RuntimeError("Split indices overlap.")


def _assert_partition(n_samples: int, *parts: np.ndarray) -> None:
    _assert_disjoint(*parts)
    combined = np.sort(np.concatenate(parts))
    if not np.array_equal(combined, np.arange(n_samples)):
        raise RuntimeError("Split indices do not cover the source dataset.")


def _subset(
    bundle: DatasetBundle,
    indices: np.ndarray,
    split_name: str,
) -> DatasetBundle:
    metadata = dict(bundle.metadata)
    metadata.update(
        {
            "split": split_name,
            "split_indices": indices.tolist(),
        }
    )
    return DatasetBundle(
        X=np.asarray(bundle.X)[indices],
        y=np.asarray(bundle.y)[indices],
        task=bundle.task,
        name=bundle.name,
        feature_names=bundle.feature_names,
        target_names=bundle.target_names,
        metadata=metadata,
    )


__all__ = [
    "DatasetFold",
    "DatasetSplits",
    "make_kfold_splits",
    "make_repeated_kfold_splits",
    "split_train_val_test",
]
