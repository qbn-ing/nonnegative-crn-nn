from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np


TaskType = Literal["classification"]


def _check_task(task: str) -> TaskType:
    if task != "classification":
        raise ValueError(
            "The current Rational CRN paper protocol supports classification "
            f"only; got task={task!r}."
        )
    return "classification"


@dataclass(frozen=True)
class DatasetMeta:
    """Dataset identity independent of any model or training policy."""

    task: TaskType = "classification"
    name: str | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_task(self.task)
        if self.name is not None and not str(self.name).strip():
            raise ValueError("Dataset name must be non-empty when provided.")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass
class DatasetBundle:
    """Numeric features, encoded class labels and reproducibility metadata."""

    X: Any
    y: Any
    task: TaskType = "classification"
    name: str | None = None
    feature_names: Sequence[str] | None = None
    target_names: Sequence[str] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_task(self.task)
        self.feature_names = (
            None
            if self.feature_names is None
            else [str(value) for value in self.feature_names]
        )
        self.target_names = (
            None
            if self.target_names is None
            else [str(value) for value in self.target_names]
        )
        self.metadata = dict(self.metadata)

    @property
    def n_samples(self) -> int:
        array = np.asarray(self.X)
        if array.ndim == 0:
            raise ValueError("X must contain a sample dimension.")
        return int(array.shape[0])

    @property
    def n_features(self) -> int:
        array = np.asarray(self.X)
        if array.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features).")
        return int(array.shape[1])

    @property
    def n_classes(self) -> int:
        labels = np.asarray(self.y).reshape(-1)
        return int(np.unique(labels).size)


def dataset_fingerprint(bundle: DatasetBundle) -> str:
    """Return a stable SHA-256 fingerprint for an adapted dataset bundle."""

    if not isinstance(bundle, DatasetBundle):
        raise TypeError("bundle must be a DatasetBundle.")
    X = np.ascontiguousarray(np.asarray(bundle.X))
    y = np.ascontiguousarray(np.asarray(bundle.y))
    digest = hashlib.sha256()
    for name, array in (("X", X), ("y", y)):
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    identity = {
        "task": bundle.task,
        "name": bundle.name,
        "feature_names": bundle.feature_names,
        "target_names": bundle.target_names,
        "dataset_version": bundle.metadata.get("dataset_version"),
        "adapter_version": bundle.metadata.get("adapter_version"),
    }
    digest.update(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


__all__ = [
    "DatasetBundle",
    "DatasetMeta",
    "TaskType",
    "dataset_fingerprint",
]
