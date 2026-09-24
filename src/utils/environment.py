from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from .paths import normalize_path
from .serialization import to_jsonable, write_json


def collect_environment_info(
    *,
    project_root: str | os.PathLike[str] | None = None,
    include_packages: bool = True,
    include_git: bool = True,
) -> dict[str, Any]:
    """Collect a JSON-safe hardware, software and source-code snapshot."""

    if not isinstance(include_packages, bool):
        raise TypeError("include_packages must be a bool.")
    if not isinstance(include_git, bool):
        raise TypeError("include_git must be a bool.")

    root = None if project_root is None else normalize_path(project_root)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": _hardware_info(),
        "software": _software_info(include_packages=include_packages),
        "runtime": _runtime_info(),
    }
    if include_git:
        payload["git"] = _git_info(root)
    return payload


def save_environment_info(
    path: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str] | None = None,
    include_packages: bool = True,
    include_git: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write the environment snapshot before the first training update."""

    if extra is not None and not isinstance(extra, Mapping):
        raise TypeError("extra must be a mapping or None.")
    payload = collect_environment_info(
        project_root=project_root,
        include_packages=include_packages,
        include_git=include_git,
    )
    if extra is not None:
        payload["extra"] = to_jsonable(extra)
    return write_json(payload, path)


def _hardware_info() -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": (
                        f"{properties.major}.{properties.minor}"
                    ),
                    "multiprocessor_count": int(
                        getattr(properties, "multi_processor_count", 0)
                    ),
                }
            )

    return {
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "processor": (
            platform.processor()
            or os.environ.get("PROCESSOR_IDENTIFIER")
            or None
        ),
        "logical_cpu_count": os.cpu_count(),
        "total_memory_bytes": _total_memory_bytes(),
        "cuda_available": cuda_available,
        "cuda_devices": cuda_devices,
        "nvidia_driver_version": _nvidia_driver_version(),
        "mps_available": bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ),
    }


def _software_info(*, include_packages: bool) -> dict[str, Any]:
    numpy = _optional_import("numpy")
    tqdm = _optional_import("tqdm")
    payload: dict[str, Any] = {
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "numpy_version": (
            None if numpy is None else str(numpy.__version__)
        ),
        "tqdm_version": (
            None if tqdm is None else str(tqdm.__version__)
        ),
        "cuda_runtime_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if include_packages:
        payload["installed_packages"] = _installed_packages()
    return payload


def _runtime_info() -> dict[str, Any]:
    warn_only_getter = getattr(
        torch,
        "is_deterministic_algorithms_warn_only_enabled",
        None,
    )
    return {
        "argv": list(sys.argv),
        "working_directory": Path.cwd().as_posix(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only": (
            bool(warn_only_getter())
            if callable(warn_only_getter)
            else None
        ),
        "cudnn_deterministic": bool(
            torch.backends.cudnn.deterministic
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def _installed_packages() -> list[dict[str, str]]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        packages[str(name)] = str(distribution.version)
    return [
        {"name": name, "version": packages[name]}
        for name in sorted(packages, key=str.casefold)
    ]


def _git_info(project_root: Path | None) -> dict[str, Any]:
    working_root = Path.cwd() if project_root is None else project_root
    git = shutil.which("git")
    if git is None:
        return {"available": False, "reason": "git executable not found"}

    probe = _run_command(
        [git, "-C", str(working_root), "rev-parse", "--show-toplevel"]
    )
    if probe["returncode"] != 0:
        return {
            "available": False,
            "reason": "project root is not inside a Git work tree",
        }

    root = Path(str(probe["stdout"])).resolve()
    commit = _command_stdout(
        [git, "-C", str(root), "rev-parse", "HEAD"]
    )
    branch = _command_stdout(
        [git, "-C", str(root), "symbolic-ref", "--short", "HEAD"]
    )
    describe = _command_stdout(
        [git, "-C", str(root), "describe", "--always", "--dirty", "--tags"]
    )
    status = _command_stdout(
        [git, "-C", str(root), "status", "--porcelain"]
    )
    status_lines = [
        line for line in status.splitlines() if line.strip()
    ]
    return {
        "available": True,
        "root": root.as_posix(),
        "commit": commit or None,
        "branch": branch or None,
        "describe": describe or None,
        "dirty": bool(status_lines),
        "changed_path_count": len(status_lines),
        "untracked_path_count": sum(
            line.startswith("??") for line in status_lines
        ),
    }


def _nvidia_driver_version() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    result = _run_command(
        [
            executable,
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]
    )
    if result["returncode"] != 0:
        return None
    versions = sorted(
        {
            line.strip()
            for line in str(result["stdout"]).splitlines()
            if line.strip()
        }
    )
    return ",".join(versions) if versions else None


def _total_memory_bytes() -> int | None:
    psutil = _optional_import("psutil")
    if psutil is not None:
        return int(psutil.virtual_memory().total)
    if hasattr(os, "sysconf"):
        try:
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return pages * page_size
        except (OSError, TypeError, ValueError):
            return None
    return None


def _optional_import(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        return None


def _command_stdout(command: list[str]) -> str:
    result = _run_command(command)
    if result["returncode"] != 0:
        return ""
    return str(result["stdout"]).strip()


def _run_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "returncode": int(completed.returncode),
        "stdout": (completed.stdout or "").strip(),
        "stderr": (completed.stderr or "").strip(),
    }


__all__ = [
    "collect_environment_info",
    "save_environment_info",
]
