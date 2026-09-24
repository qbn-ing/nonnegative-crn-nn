from __future__ import annotations

import importlib
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from utils.paths import normalize_path
from utils.serialization import to_jsonable


def save_checkpoint(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int | None = None,
    global_step: int | None = None,
    config: Mapping[str, Any] | None = None,
    model_spec: Any | None = None,
    data_state: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    generators: Mapping[str, torch.Generator] | None = None,
) -> Path:
    """Atomically save model, training and random-number state."""

    _check_model(model)
    _check_optional_nonnegative_int("epoch", epoch)
    _check_optional_nonnegative_int("global_step", global_step)
    _check_optional_mapping("config", config)
    _check_optional_mapping("data_state", data_state)
    _check_optional_mapping("extra", extra)
    normalized_generators = _normalize_generators(generators)

    payload: dict[str, Any] = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_state_dict": _cpu_state_dict(model),
        "optimizer_state_dict": _state_dict_or_none(optimizer),
        "scheduler_state_dict": _state_dict_or_none(scheduler),
        "scaler_state_dict": _state_dict_or_none(scaler),
        "epoch": epoch,
        "global_step": global_step,
        "config": None if config is None else to_jsonable(config),
        "model_spec": to_jsonable(
            _extract_model_spec(model)
            if model_spec is None
            else model_spec
        ),
        "model_metadata": _model_metadata(model),
        "data_state": (
            None if data_state is None else to_jsonable(data_state)
        ),
        "extra": None if extra is None else to_jsonable(extra),
        "rng_state": _capture_rng_state(normalized_generators),
    }

    output_path = normalize_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def load_checkpoint(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    generators: Mapping[str, torch.Generator] | None = None,
    map_location: torch.device | str = "cpu",
    strict_model: bool = True,
    restore_rng: bool = False,
    strict_rng: bool = True,
) -> dict[str, Any]:
    """Load a v10.2 checkpoint and optionally resume RNG streams."""

    _check_model(model)
    if not isinstance(strict_model, bool):
        raise TypeError("strict_model must be a bool.")
    if not isinstance(restore_rng, bool):
        raise TypeError("restore_rng must be a bool.")
    if not isinstance(strict_rng, bool):
        raise TypeError("strict_rng must be a bool.")
    normalized_generators = _normalize_generators(generators)

    checkpoint_path = normalize_path(path)
    payload = _safe_torch_load(
        checkpoint_path,
        map_location=map_location,
    )
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint payload must be a mapping.")
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format_version.")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint model_state_dict must be a mapping.")

    model_load_result = model.load_state_dict(
        state_dict,
        strict=strict_model,
    )
    loaded_components = {
        "model": True,
        "optimizer": _load_optional_state(
            optimizer,
            payload.get("optimizer_state_dict"),
        ),
        "scheduler": _load_optional_state(
            scheduler,
            payload.get("scheduler_state_dict"),
        ),
        "scaler": _load_optional_state(
            scaler,
            payload.get("scaler_state_dict"),
        ),
        "rng": False,
    }
    if restore_rng:
        rng_state = payload.get("rng_state")
        if not isinstance(rng_state, Mapping):
            raise TypeError("Checkpoint rng_state must be a mapping.")
        _restore_rng_state(
            rng_state,
            generators=normalized_generators,
            strict=strict_rng,
        )
        loaded_components["rng"] = True

    return {
        "format_version": int(payload["format_version"]),
        "created_at_utc": payload.get("created_at_utc"),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "config": payload.get("config"),
        "model_spec": payload.get("model_spec"),
        "model_metadata": payload.get("model_metadata"),
        "data_state": payload.get("data_state"),
        "extra": payload.get("extra"),
        "loaded_components": loaded_components,
        "missing_model_keys": list(model_load_result.missing_keys),
        "unexpected_model_keys": list(
            model_load_result.unexpected_keys
        ),
    }


def _capture_rng_state(
    generators: Mapping[str, torch.Generator],
) -> dict[str, Any]:
    numpy = _optional_numpy()
    numpy_state = None
    if numpy is not None:
        algorithm, keys, position, has_gauss, cached = (
            numpy.random.get_state()
        )
        numpy_state = {
            "algorithm": str(algorithm),
            "keys": [int(value) for value in keys.tolist()],
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached),
        }

    cuda_states: list[torch.Tensor] = []
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_states = [
            state.detach().cpu().clone()
            for state in torch.cuda.get_rng_state_all()
        ]

    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().detach().cpu().clone(),
        "torch_cuda": cuda_states,
        "numpy": numpy_state,
        "generators": {
            name: generator.get_state().detach().cpu().clone()
            for name, generator in generators.items()
        },
    }


def _restore_rng_state(
    payload: Mapping[str, Any],
    *,
    generators: Mapping[str, torch.Generator],
    strict: bool,
) -> None:
    expected_generators = payload.get("generators", {})
    if not isinstance(expected_generators, Mapping):
        raise TypeError("rng_state.generators must be a mapping.")
    expected_names = {str(name) for name in expected_generators}
    actual_names = set(generators)
    if strict and expected_names != actual_names:
        raise ValueError(
            "Named generator mismatch: "
            f"expected {sorted(expected_names)}, "
            f"got {sorted(actual_names)}."
        )

    python_state = payload.get("python")
    torch_cpu_state = payload.get("torch_cpu")
    if not isinstance(python_state, tuple):
        raise TypeError("rng_state.python must be a tuple.")
    if not isinstance(torch_cpu_state, torch.Tensor):
        raise TypeError("rng_state.torch_cpu must be a tensor.")

    random.setstate(python_state)
    torch.set_rng_state(torch_cpu_state.detach().cpu())

    raw_cuda_states = payload.get("torch_cuda", [])
    if not isinstance(raw_cuda_states, (list, tuple)):
        raise TypeError("rng_state.torch_cuda must be a list.")
    current_cuda_count = (
        torch.cuda.device_count() if torch.cuda.is_available() else 0
    )
    if strict and raw_cuda_states and (
        len(raw_cuda_states) != current_cuda_count
    ):
        raise RuntimeError(
            "CUDA device count does not match the checkpoint RNG state."
        )
    if raw_cuda_states and current_cuda_count:
        states = [
            state.detach().cpu()
            for state in raw_cuda_states[:current_cuda_count]
        ]
        if len(states) == current_cuda_count:
            torch.cuda.set_rng_state_all(states)
        else:
            for index, state in enumerate(states):
                torch.cuda.set_rng_state(state, device=index)

    raw_numpy = payload.get("numpy")
    if raw_numpy is not None:
        numpy = _optional_numpy()
        if numpy is None:
            if strict:
                raise RuntimeError(
                    "Checkpoint contains NumPy RNG state but NumPy is absent."
                )
        elif not isinstance(raw_numpy, Mapping):
            raise TypeError("rng_state.numpy must be a mapping or None.")
        else:
            numpy.random.set_state(
                (
                    str(raw_numpy["algorithm"]),
                    numpy.asarray(
                        raw_numpy["keys"],
                        dtype=numpy.uint32,
                    ),
                    int(raw_numpy["position"]),
                    int(raw_numpy["has_gauss"]),
                    float(raw_numpy["cached_gaussian"]),
                )
            )

    for name in expected_names & actual_names:
        state = expected_generators[name]
        if not isinstance(state, torch.Tensor):
            raise TypeError(
                f"rng_state.generators[{name!r}] must be a tensor."
            )
        generators[name].set_state(state.detach().cpu())


def _safe_torch_load(
    path: Path,
    *,
    map_location: torch.device | str,
) -> Any:
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def _state_dict_or_none(value: Any | None) -> Any | None:
    if value is None:
        return None
    method = getattr(value, "state_dict", None)
    if not callable(method):
        raise TypeError("Checkpoint components must expose state_dict().")
    return method()


def _load_optional_state(
    target: Any | None,
    state: Any | None,
) -> bool:
    if target is None or state is None:
        return False
    method = getattr(target, "load_state_dict", None)
    if not callable(method):
        raise TypeError(
            "Checkpoint targets must expose load_state_dict()."
        )
    method(state)
    return True


def _cpu_state_dict(
    model: torch.nn.Module,
) -> dict[str, Any]:
    return {
        str(name): (
            value.detach().cpu().clone()
            if isinstance(value, torch.Tensor)
            else value
        )
        for name, value in model.state_dict().items()
    }


def _extract_model_spec(model: torch.nn.Module) -> Any | None:
    for method_name in (
        "export_model_spec",
        "compile_spec",
        "structure_info",
    ):
        method = getattr(model, method_name, None)
        if callable(method):
            return method()
    return None


def _model_metadata(model: torch.nn.Module) -> dict[str, Any]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
    return {
        "class": type(model).__name__,
        "module": type(model).__module__,
        "total_parameter_scalars": total,
        "trainable_parameter_scalars": trainable,
        "state_dict_keys": sorted(
            str(key) for key in model.state_dict()
        ),
    }


def _normalize_generators(
    generators: Mapping[str, torch.Generator] | None,
) -> dict[str, torch.Generator]:
    if generators is None:
        return {}
    if not isinstance(generators, Mapping):
        raise TypeError("generators must be a mapping or None.")
    normalized: dict[str, torch.Generator] = {}
    for name, generator in generators.items():
        if not isinstance(name, str) or not name.strip():
            raise TypeError("Generator names must be non-empty strings.")
        if not isinstance(generator, torch.Generator):
            raise TypeError(
                f"generators[{name!r}] must be a torch.Generator."
            )
        normalized[name] = generator
    return dict(sorted(normalized.items()))


def _optional_numpy() -> Any | None:
    try:
        return importlib.import_module("numpy")
    except ModuleNotFoundError as exc:
        if exc.name != "numpy":
            raise
        return None


def _check_model(model: Any) -> None:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module.")


def _check_optional_nonnegative_int(
    name: str,
    value: int | None,
) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int or None.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")


def _check_optional_mapping(
    name: str,
    value: Mapping[str, Any] | None,
) -> None:
    if value is not None and not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or None.")


__all__ = [
    "load_checkpoint",
    "save_checkpoint",
]
