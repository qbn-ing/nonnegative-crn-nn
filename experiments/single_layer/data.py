from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
    make_circles,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from .protocol import RunSpec


@dataclass(frozen=True)
class TabularDataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        X = np.asarray(self.X, dtype=np.float64)
        y = np.asarray(self.y, dtype=np.int64).reshape(-1)
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError("X and y must be aligned 2D/1D arrays.")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("dataset must not be empty.")
        if np.isinf(X).any():
            raise ValueError("X contains infinite values.")
        classes = np.unique(y)
        if not np.array_equal(classes, np.arange(classes.size)):
            raise ValueError("labels must be contiguous integers 0..K-1.")
        if len(self.feature_names) != X.shape[1]:
            raise ValueError("feature_names length must match X columns.")
        if len(self.target_names) != classes.size:
            raise ValueError("target_names length must match class count.")
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_classes(self) -> int:
        return int(np.unique(self.y).size)


@dataclass(frozen=True)
class FoldIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def __post_init__(self) -> None:
        parts = [np.asarray(value, dtype=np.int64) for value in (
            self.train,
            self.validation,
            self.test,
        )]
        combined = np.concatenate(parts)
        if combined.size != np.unique(combined).size:
            raise ValueError("train/validation/test indices overlap.")
        object.__setattr__(self, "train", parts[0])
        object.__setattr__(self, "validation", parts[1])
        object.__setattr__(self, "test", parts[2])

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_indices": self.train.tolist(),
            "validation_indices": self.validation.tolist(),
            "test_indices": self.test.tolist(),
        }


@dataclass(frozen=True)
class PreprocessorState:
    method: str
    kept_feature_indices: np.ndarray
    imputation_values: np.ndarray
    offsets: np.ndarray
    scales: np.ndarray
    clip: bool
    source_feature_count: int
    metadata: dict[str, Any]

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.source_feature_count:
            raise ValueError("preprocessor input feature count mismatch.")
        selected = values[:, self.kept_feature_indices].copy()
        missing = np.isnan(selected)
        if missing.any():
            rows, cols = np.where(missing)
            selected[rows, cols] = self.imputation_values[cols]
        output = (selected - self.offsets) / self.scales
        if self.clip:
            output = np.clip(output, 0.0, 1.0)
        output = np.maximum(output, 0.0)
        if not np.isfinite(output).all():
            raise ValueError("preprocessing produced NaN or Inf.")
        return np.asarray(output, dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": self.method,
            "source_feature_count": self.source_feature_count,
            "output_feature_count": int(self.kept_feature_indices.size),
            "kept_feature_indices": self.kept_feature_indices.tolist(),
            "imputation_values": self.imputation_values.tolist(),
            "offsets": self.offsets.tolist(),
            "scales": self.scales.tolist(),
            "clip": self.clip,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PreparedFold:
    dataset: TabularDataset
    indices: FoldIndices
    preprocessor: PreprocessorState
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    fingerprint: str

    @property
    def n_features(self) -> int:
        return int(self.x_train.shape[1])

    @property
    def n_classes(self) -> int:
        return self.dataset.n_classes

    def summary(self, spec: RunSpec) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dataset": self.dataset.name,
            "source_metadata": self.dataset.metadata,
            "source_samples": self.dataset.n_samples,
            "source_features": self.dataset.n_features,
            "processed_features": self.n_features,
            "n_classes": self.n_classes,
            "class_counts": _class_counts(self.dataset.y),
            "split_class_counts": {
                "train": _class_counts(self.y_train),
                "validation": _class_counts(self.y_validation),
                "test": _class_counts(self.y_test),
            },
            "repeat": spec.repeat,
            "fold": spec.fold,
            "outer_folds": spec.outer_folds,
            "split_seed": spec.split_seed,
            "model_seed": spec.model_seed,
            "indices": self.indices.to_dict(),
            "preprocessor": self.preprocessor.to_dict(),
            "fingerprint": self.fingerprint,
        }


def load_dataset(
    spec: RunSpec,
    *,
    st003390_csv: str | Path,
) -> TabularDataset:
    if spec.dataset == "iris":
        raw = load_iris()
        return _from_sklearn(raw, name="iris", source="load_iris")
    if spec.dataset in {"wine_binary", "wine_full"}:
        raw = load_wine()
        if spec.dataset == "wine_full":
            return _from_sklearn(raw, name="wine_full", source="load_wine")
        mask = np.asarray(raw.target) < 2
        return TabularDataset(
            name="wine_0_vs_1",
            X=np.asarray(raw.data)[mask],
            y=np.asarray(raw.target)[mask],
            feature_names=tuple(map(str, raw.feature_names)),
            target_names=tuple(map(str, raw.target_names[:2])),
            metadata={
                "source": "sklearn.datasets.load_wine",
                "class_filter": [0, 1],
                "paper_default": True,
            },
        )
    if spec.dataset == "breast_cancer":
        raw = load_breast_cancer()
        return _from_sklearn(
            raw,
            name="breast_cancer",
            source="load_breast_cancer",
        )
    if spec.dataset == "circles":
        X, y = make_circles(
            n_samples=spec.circle.n_samples,
            factor=spec.circle.factor,
            noise=spec.circle.noise,
            random_state=spec.circle.generator_seed,
        )
        return TabularDataset(
            name="circles",
            X=X,
            y=y,
            feature_names=("x0", "x1"),
            target_names=("outer", "inner"),
            metadata={
                "source": "sklearn.datasets.make_circles",
                "generator": as_circle_dict(spec),
                "formal_parameters_frozen": spec.circle.parameters_frozen,
            },
        )
    if spec.dataset == "st003390_m1m2":
        return load_st003390_csv(st003390_csv)
    raise AssertionError("unhandled dataset.")


def as_circle_dict(spec: RunSpec) -> dict[str, Any]:
    return {
        "n_samples": spec.circle.n_samples,
        "factor": spec.circle.factor,
        "noise": spec.circle.noise,
        "generator_seed": spec.circle.generator_seed,
        "parameters_frozen": spec.circle.parameters_frozen,
    }


def load_st003390_csv(path: str | Path) -> TabularDataset:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Prepared ST003390 M1/M2 file is missing: {source}. "
            "Run experiments/single_layer/prepare_st003390.py first."
        )
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or header[:3] != ["sample_id", "label", "group"]:
            raise ValueError("unexpected ST003390 CSV header.")
        feature_names = tuple(header[3:])
        samples: list[str] = []
        groups: list[str] = []
        labels: list[int] = []
        rows: list[list[float]] = []
        for row in reader:
            if len(row) != len(header):
                raise ValueError("ST003390 CSV row width mismatch.")
            samples.append(row[0])
            labels.append(int(row[1]))
            groups.append(row[2])
            rows.append([_optional_float(value) for value in row[3:]])
    dataset = TabularDataset(
        name="st003390_m1m2",
        X=np.asarray(rows, dtype=np.float64),
        y=np.asarray(labels, dtype=np.int64),
        feature_names=feature_names,
        target_names=("healthy_control", "T2DM"),
        metadata={
            "source": "Metabolomics Workbench ST003390/PR002101",
            "analyses": ["AN005559", "AN005560"],
            "methods": ["M1", "M2"],
            "M3_excluded": True,
            "M4_excluded": True,
            "sample_ids_unique": len(samples) == len(set(samples)),
            "groups_unique": len(groups) == len(set(groups)),
            "csv_path": source.as_posix(),
        },
    )
    if dataset.X.shape != (300, 201):
        raise ValueError(
            "ST003390 M1/M2 must have shape (300, 201); "
            f"got {dataset.X.shape}."
        )
    if _class_counts(dataset.y) != {"0": 200, "1": 100}:
        raise ValueError("ST003390 class counts must be 200 healthy / 100 T2DM.")
    if not dataset.metadata["sample_ids_unique"]:
        raise ValueError("ST003390 sample IDs must be unique.")
    return dataset


def prepare_fold(spec: RunSpec, dataset: TabularDataset) -> PreparedFold:
    indices = make_fold_indices(spec, dataset.y)
    x_train_raw = dataset.X[indices.train]
    if spec.dataset == "st003390_m1m2":
        preprocessor = fit_st003390_preprocessor(
            x_train_raw,
            max_missing_fraction=spec.st_missing_fraction,
            scale_quantile=spec.st_scale_quantile,
        )
    else:
        preprocessor = fit_minmax_preprocessor(x_train_raw)
    x_train = preprocessor.transform(x_train_raw)
    x_validation = preprocessor.transform(dataset.X[indices.validation])
    x_test = preprocessor.transform(dataset.X[indices.test])
    y_train = dataset.y[indices.train].copy()
    y_validation = dataset.y[indices.validation].copy()
    y_test = dataset.y[indices.test].copy()
    fingerprint = fold_fingerprint(
        dataset,
        indices,
        preprocessor,
        spec=spec,
    )
    return PreparedFold(
        dataset=dataset,
        indices=indices,
        preprocessor=preprocessor,
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
        fingerprint=fingerprint,
    )


def make_fold_indices(spec: RunSpec, y: np.ndarray) -> FoldIndices:
    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    splitter = StratifiedKFold(
        n_splits=spec.outer_folds,
        shuffle=True,
        random_state=spec.split_seed,
    )
    dummy = np.zeros((labels.size, 1), dtype=np.float32)
    pairs = list(splitter.split(dummy, labels))
    outer_train, test = pairs[spec.fold]
    train, validation = train_test_split(
        outer_train,
        test_size=spec.validation_fraction,
        random_state=10_000 + 100 * spec.split_seed + spec.fold,
        shuffle=True,
        stratify=labels[outer_train],
    )
    return FoldIndices(
        train=np.sort(train),
        validation=np.sort(validation),
        test=np.sort(test),
    )


def fit_minmax_preprocessor(X_train: np.ndarray) -> PreprocessorState:
    values = np.asarray(X_train, dtype=np.float64)
    if np.isnan(values).any() or np.isinf(values).any():
        raise ValueError("non-ST datasets must contain finite features.")
    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    scale = maximum - minimum
    scale[scale <= 1e-12] = 1.0
    count = values.shape[1]
    return PreprocessorState(
        method="train_fold_minmax",
        kept_feature_indices=np.arange(count, dtype=np.int64),
        imputation_values=np.zeros(count, dtype=np.float64),
        offsets=minimum,
        scales=scale,
        clip=True,
        source_feature_count=count,
        metadata={"fit_scope": "training_fold_only", "feature_range": [0, 1]},
    )


def fit_st003390_preprocessor(
    X_train: np.ndarray,
    *,
    max_missing_fraction: float,
    scale_quantile: float,
) -> PreprocessorState:
    values = np.asarray(X_train, dtype=np.float64)
    if values.ndim != 2 or np.isinf(values).any():
        raise ValueError("invalid ST003390 training matrix.")
    missing_fraction = np.mean(np.isnan(values), axis=0)
    keep = np.flatnonzero(missing_fraction <= max_missing_fraction)
    if keep.size == 0:
        raise ValueError("missingness filtering removed every feature.")
    selected = values[:, keep]
    imputation = np.zeros(selected.shape[1], dtype=np.float64)
    for index in range(selected.shape[1]):
        observed = selected[:, index]
        positive = observed[np.isfinite(observed) & (observed > 0.0)]
        imputation[index] = 0.5 * float(positive.min()) if positive.size else 0.0
    filled = selected.copy()
    rows, cols = np.where(np.isnan(filled))
    if rows.size:
        filled[rows, cols] = imputation[cols]
    scale = np.quantile(filled, scale_quantile, axis=0)
    scale[scale <= 1e-12] = 1.0
    return PreprocessorState(
        method="st003390_half_min_q90",
        kept_feature_indices=keep,
        imputation_values=imputation,
        offsets=np.zeros(keep.size, dtype=np.float64),
        scales=scale,
        clip=False,
        source_feature_count=values.shape[1],
        metadata={
            "fit_scope": "training_fold_only",
            "max_missing_fraction": max_missing_fraction,
            "scale_quantile": scale_quantile,
            "imputation": "half_training_fold_minimum_positive",
            "removed_feature_count": int(values.shape[1] - keep.size),
        },
    )


def fold_fingerprint(
    dataset: TabularDataset,
    indices: FoldIndices,
    preprocessor: PreprocessorState,
    *,
    spec: RunSpec,
) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ("X", dataset.X),
        ("y", dataset.y),
        ("train", indices.train),
        ("validation", indices.validation),
        ("test", indices.test),
    ):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    protocol = {
        "dataset": spec.dataset,
        "repeat": spec.repeat,
        "fold": spec.fold,
        "split_seed": spec.split_seed,
        "circle": as_circle_dict(spec) if spec.dataset == "circles" else None,
        "preprocessor": preprocessor.to_dict(),
    }
    digest.update(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return digest.hexdigest()


def expected_basis_dim(n_inputs: int, representation: str) -> int:
    if representation == "raw":
        return int(n_inputs)
    if representation == "quadratic":
        return int(2 * n_inputs + n_inputs * (n_inputs - 1) // 2)
    raise ValueError("unsupported representation.")


def _from_sklearn(raw: Any, *, name: str, source: str) -> TabularDataset:
    return TabularDataset(
        name=name,
        X=np.asarray(raw.data),
        y=np.asarray(raw.target),
        feature_names=tuple(map(str, raw.feature_names)),
        target_names=tuple(map(str, raw.target_names)),
        metadata={"source": f"sklearn.datasets.{source}"},
    )


def _optional_float(value: str) -> float:
    stripped = value.strip()
    return float("nan") if stripped == "" else float(stripped)


def _class_counts(y: Sequence[int] | np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(np.asarray(y, dtype=np.int64), return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


__all__ = [
    "FoldIndices",
    "PreparedFold",
    "PreprocessorState",
    "TabularDataset",
    "as_circle_dict",
    "expected_basis_dim",
    "fit_minmax_preprocessor",
    "fit_st003390_preprocessor",
    "fold_fingerprint",
    "load_dataset",
    "load_st003390_csv",
    "make_fold_indices",
    "prepare_fold",
]
