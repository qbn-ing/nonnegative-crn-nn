from __future__ import annotations

import importlib
import os
import random
from typing import Any

import torch


_MAX_PORTABLE_SEED = 2**32 - 1
_DEFAULT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def set_global_seed(
    seed: int,
    *,
    deterministic: bool = True,
    warn_only: bool = False,
    seed_numpy: bool = True,
    strict: bool = False,
) -> dict[str, Any]:
    """Seed the current Python/PyTorch process with one integer.

    The same ``seed`` is sent to Python, optional NumPy, PyTorch CPU/CUDA and
    child-process defaults.  Formal experiment launchers should additionally
    set ``PYTHONHASHSEED`` and ``CUBLAS_WORKSPACE_CONFIG`` before Python starts.

    ``strict=True`` verifies those two launch-time settings instead of silently
    claiming that a late in-process assignment changed already-initialized
    behavior.
    """

    _check_seed(seed)
    _check_bool("deterministic", deterministic)
    _check_bool("warn_only", warn_only)
    _check_bool("seed_numpy", seed_numpy)
    _check_bool("strict", strict)
    if warn_only and not deterministic:
        raise ValueError("warn_only requires deterministic=True.")

    seed_text = str(seed)
    hash_preconfigured = os.environ.get("PYTHONHASHSEED") == seed_text
    cuda_initialized = bool(torch.cuda.is_initialized())
    cublas_value = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    cublas_preconfigured = cublas_value in {
        ":16:8",
        ":4096:8",
    }

    if strict and not hash_preconfigured:
        raise RuntimeError(
            "Strict reproducibility requires PYTHONHASHSEED to equal the seed "
            "before Python starts."
        )
    if (
        strict
        and deterministic
        and torch.cuda.is_available()
        and cuda_initialized
        and not cublas_preconfigured
    ):
        raise RuntimeError(
            "Strict CUDA reproducibility requires CUBLAS_WORKSPACE_CONFIG to "
            "be configured before CUDA initialization."
        )

    os.environ["PYTHONHASHSEED"] = seed_text
    if deterministic and not cublas_preconfigured:
        os.environ[
            "CUBLAS_WORKSPACE_CONFIG"
        ] = _DEFAULT_CUBLAS_WORKSPACE_CONFIG

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    numpy = _optional_numpy() if seed_numpy else None
    if numpy is not None:
        numpy.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(
        deterministic,
        warn_only=warn_only,
    )

    notes: list[str] = []
    if not hash_preconfigured:
        notes.append(
            "PYTHONHASHSEED now applies to child processes; restart Python "
            "with it preconfigured for strict hash reproducibility."
        )
    if deterministic and cuda_initialized and not cublas_preconfigured:
        notes.append(
            "CUDA was initialized before cuBLAS determinism was configured; "
            "restart Python for strict CUDA reproducibility."
        )
    if seed_numpy and numpy is None:
        notes.append("NumPy is not installed, so its RNG was not seeded.")

    return {
        "seed": seed,
        "deterministic": deterministic,
        "warn_only": warn_only,
        "numpy_seeded": numpy is not None,
        "python_hash_seed_preconfigured": hash_preconfigured,
        "cublas_workspace_preconfigured": cublas_preconfigured,
        "strict_ready": (
            hash_preconfigured
            and (
                not deterministic
                or not torch.cuda.is_available()
                or not cuda_initialized
                or cublas_preconfigured
            )
        ),
        "notes": notes,
    }


def make_torch_generator(
    seed: int | None,
    *,
    device: torch.device | str = "cpu",
) -> torch.Generator | None:
    """Create an independent generator for a DataLoader or sampler."""

    if seed is None:
        return None
    _check_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """Seed Python and optional NumPy inside one DataLoader worker."""

    if not isinstance(worker_id, int) or isinstance(worker_id, bool):
        raise TypeError("worker_id must be an int.")
    if worker_id < 0:
        raise ValueError("worker_id must be nonnegative.")

    worker_seed = int(torch.initial_seed() % 2**32)
    random.seed(worker_seed)
    numpy = _optional_numpy()
    if numpy is not None:
        numpy.random.seed(worker_seed)


def dataloader_seed_kwargs(seed: int) -> dict[str, Any]:
    """Return the two reproducibility arguments used by ``DataLoader``."""

    return {
        "generator": make_torch_generator(seed),
        "worker_init_fn": seed_worker,
    }


def _optional_numpy() -> Any | None:
    try:
        return importlib.import_module("numpy")
    except ModuleNotFoundError as exc:
        if exc.name != "numpy":
            raise
        return None


def _check_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an int.")
    if seed < 0 or seed > _MAX_PORTABLE_SEED:
        raise ValueError(
            "seed must be in the portable range "
            f"[0, {_MAX_PORTABLE_SEED}]."
        )


def _check_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")


__all__ = [
    "dataloader_seed_kwargs",
    "make_torch_generator",
    "seed_worker",
    "set_global_seed",
]
