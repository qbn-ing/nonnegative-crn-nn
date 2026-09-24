from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import KFold

from .data_prepare import processed_paths, validate_processed
from .protocol import RunSpec


@dataclass(frozen=True)
class PreparedFold:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    train_writers: np.ndarray
    validation_writers: np.ndarray
    test_writers: np.ndarray
    all_groups: np.ndarray
    dataset_fingerprint: str
    split_fingerprint: str
    input_dim: int
    n_classes: int

    def summary(self, spec: RunSpec) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dataset": "Handwritten Chinese Numbers (Chinese-MNIST)",
            "dataset_fingerprint": self.dataset_fingerprint,
            "split_fingerprint": self.split_fingerprint,
            "fold": spec.fold,
            "split_seed": spec.split_seed,
            "writer_disjoint": True,
            "train_samples": len(self.train_indices),
            "validation_samples": len(self.validation_indices),
            "test_samples": len(self.test_indices),
            "train_writer_count": len(self.train_writers),
            "validation_writer_count": len(self.validation_writers),
            "test_writer_count": len(self.test_writers),
            "train_writers": self.train_writers.tolist(),
            "validation_writers": self.validation_writers.tolist(),
            "test_writers": self.test_writers.tolist(),
            "input_dim": self.input_dim,
            "num_classes": self.n_classes,
            "class_counts": {
                "train": _class_counts(self.y_train, self.n_classes),
                "validation": _class_counts(self.y_validation, self.n_classes),
                "test": _class_counts(self.y_test, self.n_classes),
            },
        }


def _digest_array(array: np.ndarray) -> str:
    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _class_counts(labels: np.ndarray, n_classes: int) -> list[int]:
    return np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes).tolist()


def _split_fingerprint(
    arrays: dict[str, np.ndarray],
    dataset_fingerprint: str,
) -> str:
    digest = hashlib.sha256()
    for key in (
        "train",
        "validation",
        "test",
        "train_writers",
        "validation_writers",
        "test_writers",
    ):
        digest.update(_digest_array(arrays[key]).encode("ascii"))
    digest.update(dataset_fingerprint.encode("ascii"))
    return digest.hexdigest()


def split_root(data_root: str | Path, spec: RunSpec) -> Path:
    return Path(data_root).resolve() / "splits" / (
        f"writer{spec.n_splits}fold_seed_{spec.split_seed}_"
        f"valwriters_{spec.validation_writers}"
    )


def make_writer_split(data_root: str | Path, spec: RunSpec) -> dict[str, Any]:
    info = validate_processed(data_root, spec.image_size)
    paths = processed_paths(data_root, spec.image_size)
    labels = np.asarray(np.load(paths["labels"], mmap_mode="r"), dtype=np.int64)
    groups = np.asarray(np.load(paths["groups"], mmap_mode="r"), dtype=np.int64)
    writers = np.sort(np.unique(groups)).astype(np.int64)
    if int(info["num_classes"]) != spec.num_classes:
        raise RuntimeError("processed class count does not match the run protocol.")
    if len(writers) < spec.n_splits:
        raise RuntimeError("not enough writers for outer cross-validation.")
    outer = KFold(n_splits=spec.n_splits, shuffle=True, random_state=spec.split_seed)
    outer_train_positions, test_positions = list(outer.split(writers))[spec.fold]
    outer_train_writers = writers[np.asarray(outer_train_positions, dtype=np.int64)]
    test_writers = np.sort(writers[np.asarray(test_positions, dtype=np.int64)])
    if not 0 < spec.validation_writers < len(outer_train_writers):
        raise ValueError("validation_writers is invalid for this outer fold.")
    rng = np.random.default_rng(spec.split_seed + 10_000 + spec.fold)
    shuffled = outer_train_writers.copy()
    rng.shuffle(shuffled)
    validation_writers = np.sort(shuffled[: spec.validation_writers])
    train_writers = np.sort(shuffled[spec.validation_writers :])

    def indices_for(selected: np.ndarray) -> np.ndarray:
        return np.sort(np.flatnonzero(np.isin(groups, selected))).astype(np.int64)

    train = indices_for(train_writers)
    validation = indices_for(validation_writers)
    test = indices_for(test_writers)
    if np.intersect1d(train_writers, validation_writers).size:
        raise AssertionError("train and validation writers overlap.")
    if np.intersect1d(train_writers, test_writers).size:
        raise AssertionError("train and test writers overlap.")
    if np.intersect1d(validation_writers, test_writers).size:
        raise AssertionError("validation and test writers overlap.")
    if len(np.unique(np.concatenate((train, validation, test)))) != len(labels):
        raise AssertionError("split indices do not partition the dataset.")
    expected = {
        "train": len(train_writers) * 10,
        "validation": len(validation_writers) * 10,
        "test": len(test_writers) * 10,
    }
    for name, indices in (("train", train), ("validation", validation), ("test", test)):
        counts = np.bincount(labels[indices], minlength=spec.num_classes)
        if not np.all(counts == expected[name]):
            raise AssertionError(f"{name} class counts are not writer-balanced: {counts.tolist()}")

    split_arrays = {
        "train": train,
        "validation": validation,
        "test": test,
        "train_writers": train_writers,
        "validation_writers": validation_writers,
        "test_writers": test_writers,
    }
    payload = {
        "schema_version": 1,
        "protocol": "writer_disjoint_outer_5fold_with_writer_disjoint_validation",
        "fold": spec.fold,
        "split_seed": spec.split_seed,
        "n_splits": spec.n_splits,
        "validation_writers": spec.validation_writers,
        "dataset_fingerprint": info["dataset_fingerprint"],
        "split_fingerprint": _split_fingerprint(
            split_arrays,
            str(info["dataset_fingerprint"]),
        ),
        "train_samples": len(train),
        "validation_samples": len(validation),
        "test_samples": len(test),
        "train_writers": train_writers.tolist(),
        "validation_writers_list": validation_writers.tolist(),
        "test_writers": test_writers.tolist(),
        "train_class_counts": _class_counts(labels[train], spec.num_classes),
        "validation_class_counts": _class_counts(labels[validation], spec.num_classes),
        "test_class_counts": _class_counts(labels[test], spec.num_classes),
    }
    root = split_root(data_root, spec)
    root.mkdir(parents=True, exist_ok=True)
    npz_path = root / f"fold_{spec.fold}.npz"
    json_path = root / f"fold_{spec.fold}.json"
    temporary_npz = root / f"fold_{spec.fold}.{os.getpid()}.tmp.npz"
    np.savez_compressed(
        temporary_npz,
        train=train,
        validation=validation,
        test=test,
        train_writers=train_writers,
        validation_writers=validation_writers,
        test_writers=test_writers,
    )
    os.replace(temporary_npz, npz_path)
    temporary_json = root / f"fold_{spec.fold}.{os.getpid()}.tmp.json"
    temporary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_json, json_path)
    return payload


def load_split(data_root: str | Path, spec: RunSpec) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    info = validate_processed(data_root, spec.image_size)
    root = split_root(data_root, spec)
    json_path = root / f"fold_{spec.fold}.json"
    npz_path = root / f"fold_{spec.fold}.npz"
    if not json_path.is_file() or not npz_path.is_file():
        make_writer_split(data_root, spec)
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = make_writer_split(data_root, spec)
    expected = {
        "fold": spec.fold,
        "split_seed": spec.split_seed,
        "n_splits": spec.n_splits,
        "validation_writers": spec.validation_writers,
    }
    cache_matches = all(payload.get(key) == value for key, value in expected.items())
    cache_matches = cache_matches and (
        payload.get("dataset_fingerprint") == info["dataset_fingerprint"]
    )
    if not cache_matches:
        payload = make_writer_split(data_root, spec)
    try:
        with np.load(npz_path) as archive:
            arrays = {
                key: np.asarray(archive[key], dtype=np.int64)
                for key in archive.files
            }
    except (OSError, ValueError, KeyError):
        payload = make_writer_split(data_root, spec)
        with np.load(npz_path) as archive:
            arrays = {
                key: np.asarray(archive[key], dtype=np.int64)
                for key in archive.files
            }
    observed = _split_fingerprint(arrays, str(info["dataset_fingerprint"]))
    if payload.get("split_fingerprint") != observed:
        payload = make_writer_split(data_root, spec)
        with np.load(npz_path) as archive:
            arrays = {
                key: np.asarray(archive[key], dtype=np.int64)
                for key in archive.files
            }
        observed = _split_fingerprint(arrays, str(info["dataset_fingerprint"]))
        if payload.get("split_fingerprint") != observed:
            raise RuntimeError("regenerated split fingerprint does not match arrays.")
    return payload, arrays


def prepare_fold(data_root: str | Path, spec: RunSpec) -> PreparedFold:
    info = validate_processed(data_root, spec.image_size)
    paths = processed_paths(data_root, spec.image_size)
    split, indices = load_split(data_root, spec)
    images = np.load(paths["images"], mmap_mode="r")
    labels = np.asarray(np.load(paths["labels"], mmap_mode="r"), dtype=np.int64)
    groups = np.asarray(np.load(paths["groups"], mmap_mode="r"), dtype=np.int64)

    def take(name: str) -> tuple[np.ndarray, np.ndarray]:
        selected = indices[name]
        x = np.asarray(images[selected], dtype=np.float32).copy()
        y = np.asarray(labels[selected], dtype=np.int64).copy()
        return x, y

    x_train, y_train = take("train")
    x_validation, y_validation = take("validation")
    x_test, y_test = take("test")
    return PreparedFold(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
        train_indices=indices["train"],
        validation_indices=indices["validation"],
        test_indices=indices["test"],
        train_writers=indices["train_writers"],
        validation_writers=indices["validation_writers"],
        test_writers=indices["test_writers"],
        all_groups=groups,
        dataset_fingerprint=str(info["dataset_fingerprint"]),
        split_fingerprint=str(split["split_fingerprint"]),
        input_dim=int(info["input_dim"]),
        n_classes=int(info["num_classes"]),
    )


def validate_all_outer_folds(data_root: str | Path, specs: tuple[RunSpec, ...]) -> None:
    by_fold: dict[int, RunSpec] = {}
    for spec in specs:
        by_fold.setdefault(spec.fold, spec)
    test_writer_parts: list[np.ndarray] = []
    test_sample_parts: list[np.ndarray] = []
    for fold in sorted(by_fold):
        _, arrays = load_split(data_root, by_fold[fold])
        test_writer_parts.append(arrays["test_writers"])
        test_sample_parts.append(arrays["test"])
    if set(by_fold) == set(range(next(iter(by_fold.values())).n_splits)):
        all_writers = np.concatenate(test_writer_parts)
        all_samples = np.concatenate(test_sample_parts)
        if len(all_writers) != 100 or len(np.unique(all_writers)) != 100:
            raise AssertionError("every writer must appear in one outer test fold.")
        if len(all_samples) != 15_000 or len(np.unique(all_samples)) != 15_000:
            raise AssertionError("every sample must appear in one outer test fold.")


__all__ = [
    "PreparedFold",
    "load_split",
    "make_writer_split",
    "prepare_fold",
    "split_root",
    "validate_all_outer_folds",
]
