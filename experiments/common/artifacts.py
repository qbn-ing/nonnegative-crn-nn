from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from reporting.checkpoint import load_checkpoint


@dataclass(frozen=True)
class LoadedRun:
    kind: str
    run_dir: Path
    run_id: str
    protocol: dict[str, Any]
    metrics: dict[str, Any]
    model: torch.nn.Module
    network: torch.nn.Module
    x_test: torch.Tensor
    y_test: torch.Tensor
    sample_indices: np.ndarray
    dataset_fingerprint: str

    @property
    def depth(self) -> int:
        return int(getattr(self.network, "depth"))

    def network_inputs(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "single_layer":
            return self.model.features(x)
        if self.kind == "chebyshev":
            from experiments.chebyshev.data import encode_inputs

            return encode_inputs(x, self.model.task)
        if self.kind == "chinese_mnist":
            return x
        raise AssertionError(f"unsupported run kind: {self.kind}")

    def crn_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Return inputs expected by the exported CRN model specification."""

        if self.kind == "chebyshev":
            from experiments.chebyshev.data import encode_inputs

            return encode_inputs(x, self.model.task)
        if self.kind in {"single_layer", "chinese_mnist"}:
            return x
        raise AssertionError(f"unsupported run kind: {self.kind}")


RunLoader = Callable[..., LoadedRun]


def load_completed_run(
    run_dir: str | Path,
    *,
    st003390_csv: str | Path = Path("data/st003390/st003390_m1m2.csv"),
    chinese_mnist_data_root: str | Path = Path("data/chinese_mnist"),
    device: torch.device | str = "cpu",
    deterministic: bool = True,
) -> LoadedRun:
    """Rebuild any registered experiment run and restore its selected checkpoint."""

    source = Path(run_dir).resolve()
    protocol = read_json_object(source / "resolved_protocol.json")
    metrics = read_json_object(source / "metrics.json")
    status = read_json_object(source / "run_status.json")
    if status.get("status") != "completed":
        raise ValueError(f"run is not completed: {source}")
    checkpoint = source / "training" / "best_checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"selected checkpoint is missing: {checkpoint}")
    kind = infer_experiment_kind(protocol)
    loaders: dict[str, Callable[[], LoadedRun]] = {
        "single_layer": lambda: _load_single_layer(
            source,
            protocol,
            metrics,
            st003390_csv=st003390_csv,
        ),
        "chebyshev": lambda: _load_chebyshev(
            source,
            protocol,
            metrics,
            deterministic=deterministic,
        ),
        "chinese_mnist": lambda: _load_chinese_mnist(
            source,
            protocol,
            metrics,
            data_root=chinese_mnist_data_root,
        ),
    }
    loaded = loaders[kind]()
    loaded.model.to(device)
    load_checkpoint(checkpoint, loaded.model, map_location=device)
    loaded.model.eval()
    return loaded


def infer_experiment_kind(protocol: dict[str, Any]) -> str:
    explicit = protocol.get("experiment_kind")
    if explicit in {"single_layer", "chebyshev", "chinese_mnist"}:
        return str(explicit)
    if "architecture_name" in protocol and "image_size" in protocol:
        return "chinese_mnist"
    if "task" in protocol:
        return "chebyshev"
    if "dataset" in protocol:
        return "single_layer"
    raise ValueError(
        "run type cannot be inferred; protocol has no registered experiment_kind."
    )


def _load_single_layer(
    run_dir: Path,
    protocol: dict[str, Any],
    metrics: dict[str, Any],
    *,
    st003390_csv: str | Path,
) -> LoadedRun:
    from experiments.single_layer.data import load_dataset, prepare_fold
    from experiments.single_layer.model import build_model
    from experiments.single_layer.protocol import RunSpec

    spec = RunSpec.from_dict(protocol)
    dataset = load_dataset(spec, st003390_csv=st003390_csv)
    fold = prepare_fold(spec, dataset)
    _require_fingerprint(
        fold.fingerprint,
        metrics,
        kind="single-layer dataset",
    )
    build = build_model(spec, n_inputs=fold.n_features, n_classes=fold.n_classes)
    return LoadedRun(
        kind="single_layer",
        run_dir=run_dir,
        run_id=spec.run_id,
        protocol=protocol,
        metrics=metrics,
        model=build.model,
        network=build.model.network,
        x_test=torch.from_numpy(fold.x_test),
        y_test=torch.from_numpy(fold.y_test),
        sample_indices=fold.indices.test.copy(),
        dataset_fingerprint=fold.fingerprint,
    )


def _load_chebyshev(
    run_dir: Path,
    protocol: dict[str, Any],
    metrics: dict[str, Any],
    *,
    deterministic: bool,
) -> LoadedRun:
    from experiments.chebyshev.data import make_dataset
    from experiments.chebyshev.model import build_model
    from experiments.chebyshev.protocol import RunSpec

    spec = RunSpec.from_dict(protocol)
    dataset = make_dataset(spec)
    _require_fingerprint(dataset.fingerprint, metrics, kind="Chebyshev dataset")
    full_spec = replace(spec, condition="dense_full_start")
    build = build_model(
        full_spec,
        reference_raw=dataset.x_val[: min(64, len(dataset.x_val))],
        deterministic=deterministic,
    )
    return LoadedRun(
        kind="chebyshev",
        run_dir=run_dir,
        run_id=spec.run_id,
        protocol=protocol,
        metrics=metrics,
        model=build.model,
        network=build.model.network,
        x_test=dataset.x_test,
        y_test=dataset.y_test,
        sample_indices=np.arange(len(dataset.x_test), dtype=np.int64),
        dataset_fingerprint=dataset.fingerprint,
    )


def _load_chinese_mnist(
    run_dir: Path,
    protocol: dict[str, Any],
    metrics: dict[str, Any],
    *,
    data_root: str | Path,
) -> LoadedRun:
    from experiments.chinese_mnist.data import prepare_fold
    from experiments.chinese_mnist.model import build_model
    from experiments.chinese_mnist.protocol import RunSpec

    spec = RunSpec.from_dict(protocol)
    fold = prepare_fold(data_root, spec)
    _require_fingerprint(
        fold.dataset_fingerprint,
        metrics,
        kind="Chinese-MNIST dataset",
    )
    if str(metrics.get("split_fingerprint")) != fold.split_fingerprint:
        raise ValueError(
            "reconstructed Chinese-MNIST split fingerprint does not match run."
        )
    full_spec = replace(spec, condition="dense_full_start")
    reference = torch.as_tensor(
        fold.x_validation[: min(32, len(fold.x_validation))],
        dtype=torch.float32,
    )
    build = build_model(full_spec, input_dim=fold.input_dim, reference_h=reference)
    return LoadedRun(
        kind="chinese_mnist",
        run_dir=run_dir,
        run_id=spec.run_id,
        protocol=protocol,
        metrics=metrics,
        model=build.model,
        network=build.network,
        x_test=torch.from_numpy(fold.x_test),
        y_test=torch.from_numpy(fold.y_test),
        sample_indices=fold.test_indices.copy(),
        dataset_fingerprint=fold.dataset_fingerprint,
    )


def _require_fingerprint(
    observed: str,
    metrics: dict[str, Any],
    *,
    kind: str,
) -> None:
    if str(metrics.get("dataset_fingerprint")) != observed:
        raise ValueError(f"reconstructed {kind} fingerprint does not match run.")


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact must be an object: {path}")
    return value


__all__ = [
    "LoadedRun",
    "infer_experiment_kind",
    "load_completed_run",
    "read_json_object",
]
