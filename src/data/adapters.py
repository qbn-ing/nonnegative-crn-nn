from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .dataset_spec import DatasetBundle, DatasetMeta, TaskType


LoadedDataset = DatasetBundle | tuple[Any, Any] | Mapping[str, Any]


def as_dataset_bundle(
    data: LoadedDataset,
    meta: DatasetMeta | None = None,
    *,
    task: TaskType | None = None,
    name: str | None = None,
    feature_names: Sequence[str] | None = None,
    target_names: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    encode_labels: bool = True,
) -> DatasetBundle:
    """Normalize supported dataset inputs without applying preprocessing."""

    if not isinstance(encode_labels, bool):
        raise TypeError("encode_labels must be a bool.")
    extracted = _extract_user_data(data)
    resolved_task = _resolve_task(
        meta=meta,
        explicit_task=task,
        extracted_task=extracted["task"],
    )
    resolved_name = (
        name
        if name is not None
        else meta.name
        if meta is not None and meta.name is not None
        else extracted["name"]
    )
    resolved_feature_names = (
        feature_names
        if feature_names is not None
        else extracted["feature_names"]
    )
    resolved_target_names = (
        target_names
        if target_names is not None
        else extracted["target_names"]
    )
    resolved_metadata: dict[str, Any] = dict(extracted["metadata"])
    if meta is not None:
        resolved_metadata.update(dict(meta.metadata))
        if meta.version is not None:
            resolved_metadata.setdefault("dataset_version", meta.version)
    if metadata is not None:
        resolved_metadata.update(dict(metadata))

    X = _normalize_X(extracted["X"])
    y, class_values = _normalize_labels(
        extracted["y"],
        encode_labels=encode_labels,
    )
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "X and y must have the same number of samples: "
            f"got {X.shape[0]} and {y.shape[0]}."
        )
    normalized_features = _normalize_names(
        resolved_feature_names,
        field_name="feature_names",
    )
    if (
        normalized_features is not None
        and len(normalized_features) != X.shape[1]
    ):
        raise ValueError(
            "feature_names length must match X.shape[1]."
        )
    normalized_targets = _normalize_names(
        resolved_target_names,
        field_name="target_names",
    )
    if (
        normalized_targets is not None
        and len(normalized_targets) != len(class_values)
    ):
        raise ValueError(
            "target_names length must match the number of classes."
        )
    resolved_metadata.update(
        {
            "adapter_version": 1,
            "class_values": [
                value.item() if isinstance(value, np.generic) else value
                for value in class_values
            ],
            "labels_encoded_contiguously": True,
        }
    )
    return DatasetBundle(
        X=X,
        y=y,
        task=resolved_task,
        name=resolved_name,
        feature_names=normalized_features,
        target_names=normalized_targets,
        metadata=resolved_metadata,
    )


def _extract_user_data(data: LoadedDataset) -> dict[str, Any]:
    if isinstance(data, DatasetBundle):
        return {
            "X": data.X,
            "y": data.y,
            "task": data.task,
            "name": data.name,
            "feature_names": data.feature_names,
            "target_names": data.target_names,
            "metadata": data.metadata,
        }
    if isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError("Tuple input must be exactly (X, y).")
        return {
            "X": data[0],
            "y": data[1],
            "task": None,
            "name": None,
            "feature_names": None,
            "target_names": None,
            "metadata": {},
        }
    if isinstance(data, Mapping):
        if "X" in data:
            X = data["X"]
        elif "x" in data:
            X = data["x"]
        else:
            raise ValueError("Mapping input must contain 'X' or 'x'.")
        if "y" not in data:
            raise ValueError("Mapping input must contain 'y'.")
        raw_metadata = data.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise TypeError("Dataset metadata must be a mapping.")
        return {
            "X": X,
            "y": data["y"],
            "task": data.get("task"),
            "name": data.get("name"),
            "feature_names": data.get("feature_names"),
            "target_names": data.get("target_names"),
            "metadata": raw_metadata,
        }
    raise TypeError(
        "Dataset input must be DatasetBundle, (X, y), or a mapping."
    )


def _resolve_task(
    *,
    meta: DatasetMeta | None,
    explicit_task: TaskType | None,
    extracted_task: Any | None,
) -> TaskType:
    candidates = [
        value
        for value in (
            meta.task if meta is not None else None,
            explicit_task,
            extracted_task,
        )
        if value is not None
    ]
    if not candidates:
        return "classification"
    if any(value != candidates[0] for value in candidates[1:]):
        raise ValueError(f"Conflicting dataset tasks: {candidates!r}.")
    if candidates[0] != "classification":
        raise ValueError("Only classification datasets are supported.")
    return "classification"


def _normalize_X(value: Any) -> np.ndarray:
    X = np.asarray(value, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(
            "X must have shape (n_samples, n_features); "
            f"got {X.shape}."
        )
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must not be empty.")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or Inf.")
    return X


def _normalize_labels(
    value: Any,
    *,
    encode_labels: bool,
) -> tuple[np.ndarray, tuple[Any, ...]]:
    y = np.asarray(value)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y must be a non-empty one-dimensional array.")
    class_values, encoded = np.unique(y, return_inverse=True)
    if class_values.size < 2:
        raise ValueError("Classification data must contain at least two classes.")
    if not encode_labels:
        try:
            raw = y.astype(np.int64, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "encode_labels=False requires integer labels."
            ) from exc
        expected = np.arange(class_values.size, dtype=np.int64)
        if not np.array_equal(np.unique(raw), expected):
            raise ValueError(
                "encode_labels=False requires contiguous labels 0..K-1."
            )
        encoded = raw
    return (
        np.asarray(encoded, dtype=np.int64),
        tuple(class_values.tolist()),
    )


def _normalize_names(
    values: Sequence[str] | None,
    *,
    field_name: str,
) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of strings.")
    return [str(value) for value in values]


__all__ = ["LoadedDataset", "as_dataset_bundle"]
