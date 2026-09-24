from __future__ import annotations

import json
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, TypeVar

import torch

from utils.serialization import write_json

from .lifecycle import mark_experiment_failed


class RunSpecLike(Protocol):
    @property
    def run_id(self) -> str: ...

    def to_dict(self) -> dict[str, Any]: ...


SpecT = TypeVar("SpecT", bound=RunSpecLike)
Worker = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class BatchExecutionResult:
    completed: int
    failures: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed": self.completed,
            "failed": len(self.failures),
            "failures": [dict(item) for item in self.failures],
        }


def execute_run_plan(
    pending: Sequence[tuple[int, SpecT]],
    *,
    worker: Worker,
    worker_kwargs: Mapping[str, Any],
    workers: int,
    fail_fast: bool,
) -> BatchExecutionResult:
    """Execute a run plan through the one shared sequential/parallel scheduler."""

    if workers <= 0:
        raise ValueError("workers must be positive.")
    if not callable(worker):
        raise TypeError("worker must be callable.")
    if workers == 1:
        return _execute_sequential(
            pending,
            worker=worker,
            worker_kwargs=worker_kwargs,
            fail_fast=fail_fast,
        )
    return _execute_parallel(
        pending,
        worker=worker,
        worker_kwargs=worker_kwargs,
        workers=workers,
        fail_fast=fail_fast,
    )


def select_runs(
    plan: Sequence[SpecT],
    *,
    indices: Iterable[int] | None,
    max_runs: int | None,
) -> tuple[tuple[int, SpecT], ...]:
    if indices is None:
        selected = list(enumerate(plan))
    else:
        selected = []
        for index in indices:
            if index < 0 or index >= len(plan):
                raise ValueError(f"run index {index} is out of range.")
            selected.append((index, plan[index]))
    if max_runs is not None:
        if max_runs <= 0:
            raise ValueError("max_runs must be positive.")
        selected = selected[:max_runs]
    return tuple(selected)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def parse_csv_strings(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("comma-separated option must not be empty.")
    return items


def parse_csv_ints(value: str | None) -> tuple[int, ...] | None:
    items = parse_csv_strings(value)
    return None if items is None else tuple(int(item) for item in items)


def is_completed(
    run_dir: str | Path,
    *,
    required_artifacts: Iterable[str] = (
        "resolved_protocol.json",
        "metrics.json",
    ),
) -> bool:
    source = Path(run_dir)
    status_path = source / "run_status.json"
    if not status_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return status.get("status") == "completed" and all(
        (source / relative).is_file() for relative in required_artifacts
    )


def planned_run_directories(
    output_root: str | Path,
) -> tuple[tuple[Path, ...], tuple[dict[str, str], ...]]:
    """Resolve only selected run-plan directories, ignoring stale neighbors."""

    root = Path(output_root).resolve()
    plan_path = root / "run_plan.json"
    if not plan_path.is_file():
        return (
            tuple(sorted(path.parent for path in root.glob("*/metrics.json"))),
            (),
        )
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (), (
            {
                "run_id": "run_plan",
                "issue": "invalid_run_plan",
                "detail": str(exc),
            },
        )
    selected = plan.get("selected_run_ids")
    if not isinstance(selected, list) or not all(
        isinstance(item, str) and item for item in selected
    ):
        runs = plan.get("runs", [])
        selected = [
            item["run_id"]
            for item in runs
            if isinstance(item, dict) and isinstance(item.get("run_id"), str)
        ]
    selected_ids = tuple(dict.fromkeys(selected))
    selected_set = set(selected_ids)
    issues: list[dict[str, str]] = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        if metrics_path.parent.name not in selected_set:
            issues.append(
                {
                    "run_id": metrics_path.parent.name,
                    "issue": "unplanned_run_ignored",
                    "detail": "not present in run_plan.selected_run_ids",
                }
            )
    return tuple(root / run_id for run_id in selected_ids), tuple(issues)


def write_execution_summary(
    output_root: str | Path,
    *,
    selected: int,
    skipped: int,
    result: BatchExecutionResult,
) -> Path:
    return write_json(
        {
            "selected": int(selected),
            "completed_this_launch": result.completed,
            "skipped_completed": int(skipped),
            "failed_this_launch": len(result.failures),
            "failures": [dict(item) for item in result.failures],
        },
        Path(output_root) / "execution_summary.json",
    )


def _execute_sequential(
    pending: Sequence[tuple[int, SpecT]],
    *,
    worker: Worker,
    worker_kwargs: Mapping[str, Any],
    fail_fast: bool,
) -> BatchExecutionResult:
    completed = 0
    failures: list[dict[str, str]] = []
    for index, spec in pending:
        print(f"RUN  {index:04d} {spec.run_id}", flush=True)
        try:
            payload = worker(spec.to_dict(), **dict(worker_kwargs))
        except Exception as exc:
            failure = _failure(spec, exc)
            failures.append(failure)
            _record_parent_failure(spec, worker_kwargs, exc)
            print(
                f"FAIL {spec.run_id}: {failure['error_type']}: {failure['error']}",
                flush=True,
            )
            if fail_fast:
                break
        else:
            completed += 1
            _print_completed(spec.run_id, payload)
    return BatchExecutionResult(completed, tuple(failures))


def _execute_parallel(
    pending: Sequence[tuple[int, SpecT]],
    *,
    worker: Worker,
    worker_kwargs: Mapping[str, Any],
    workers: int,
    fail_fast: bool,
) -> BatchExecutionResult:
    completed = 0
    failures: list[dict[str, str]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
    ) as executor:
        future_to_spec = {
            executor.submit(worker, spec.to_dict(), **dict(worker_kwargs)): spec
            for _, spec in pending
        }
        remaining = set(future_to_spec)
        while remaining:
            done, remaining = wait(remaining, return_when=FIRST_COMPLETED)
            for future in done:
                spec = future_to_spec[future]
                try:
                    payload = future.result()
                except Exception as exc:
                    failure = _failure(spec, exc)
                    failures.append(failure)
                    _record_parent_failure(spec, worker_kwargs, exc)
                    print(
                        f"FAIL {spec.run_id}: {failure['error_type']}: "
                        f"{failure['error']}",
                        flush=True,
                    )
                    if fail_fast:
                        for item in remaining:
                            item.cancel()
                        remaining.clear()
                        break
                else:
                    completed += 1
                    _print_completed(spec.run_id, payload)
    return BatchExecutionResult(completed, tuple(failures))


def _failure(spec: RunSpecLike, exc: Exception) -> dict[str, str]:
    return {
        "run_id": spec.run_id,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _record_parent_failure(
    spec: RunSpecLike,
    worker_kwargs: Mapping[str, Any],
    exc: Exception,
) -> None:
    output_root = worker_kwargs.get("output_root")
    if output_root is None:
        return
    run_dir = Path(output_root) / spec.run_id
    mark_experiment_failed(
        run_dir,
        exc,
        run_id=spec.run_id,
        traceback_text=traceback.format_exc(),
    )


def _print_completed(run_id: str, payload: Mapping[str, Any]) -> None:
    fragments = [f"DONE {run_id}"]
    if "test_bacc" in payload:
        fragments.append(f"test_bacc={float(payload['test_bacc']):.6f}")
    if "elapsed_seconds" in payload:
        fragments.append(f"seconds={float(payload['elapsed_seconds']):.1f}")
    print(" ".join(fragments), flush=True)


__all__ = [
    "BatchExecutionResult",
    "RunSpecLike",
    "execute_run_plan",
    "is_completed",
    "parse_csv_ints",
    "parse_csv_strings",
    "planned_run_directories",
    "resolve_device",
    "select_runs",
    "write_execution_summary",
]
