from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from utils.serialization import to_jsonable, write_json


@dataclass
class ExperimentRunSession:
    """One authoritative lifecycle for training and all post-training artifacts."""

    run_dir: Path
    protocol: dict[str, Any]
    resumed: bool = False
    started_at: float = field(default_factory=time.perf_counter)
    _completed: bool = field(default=False, init=False, repr=False)

    @property
    def training_dir(self) -> Path:
        return self.run_dir / "training"

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started_at

    def __enter__(self) -> ExperimentRunSession:
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: Any,
    ) -> bool:
        if error is not None:
            self.fail(error)
        elif not self._completed:
            self.fail(RuntimeError("run session exited without completion."))
        return False

    def complete(self, payload: Mapping[str, Any]) -> None:
        if self._completed:
            raise RuntimeError("run session is already completed.")
        status = {
            **to_jsonable(dict(payload)),
            "status": "completed",
            "run_id": self.protocol["run_id"],
            "run_signature": self.protocol.get("run_signature"),
            "resumed": self.resumed,
            "elapsed_seconds": self.elapsed_seconds,
        }
        write_json(status, self.run_dir / "run_status.json")
        self._completed = True

    def fail(self, error: BaseException) -> None:
        if self._completed:
            return
        write_json(
            {
                "status": "failed",
                "run_id": self.protocol["run_id"],
                "run_signature": self.protocol.get("run_signature"),
                "resumed": self.resumed,
                "elapsed_seconds": self.elapsed_seconds,
                "error_type": type(error).__name__,
                "error": str(error),
            },
            self.run_dir / "run_status.json",
        )


def start_experiment_run(
    run_dir: str | Path,
    *,
    protocol: Mapping[str, Any],
    overwrite: bool = False,
    resume: bool = False,
) -> ExperimentRunSession:
    """Prepare an isolated run directory and reject stale configuration reuse."""

    if overwrite and resume:
        raise ValueError("overwrite and resume are mutually exclusive.")
    source = Path(run_dir).resolve()
    normalized = to_jsonable(dict(protocol))
    if not isinstance(normalized, dict):
        raise TypeError("protocol must serialize to a JSON object.")
    if not isinstance(normalized.get("run_id"), str):
        raise ValueError("protocol must contain run_id.")
    if source.exists() and any(source.iterdir()):
        if overwrite:
            shutil.rmtree(source)
        elif resume:
            _validate_resume_protocol(source, normalized)
        else:
            raise FileExistsError(
                f"run directory already contains artifacts: {source}. "
                "Use resume=True or overwrite=True explicitly."
            )
    source.mkdir(parents=True, exist_ok=True)
    write_json(normalized, source / "resolved_protocol.json")
    write_json(
        {
            "status": "running",
            "run_id": normalized["run_id"],
            "run_signature": normalized.get("run_signature"),
            "resumed": bool(resume),
        },
        source / "run_status.json",
    )
    return ExperimentRunSession(source, normalized, resumed=resume)


def _validate_resume_protocol(
    run_dir: Path,
    protocol: dict[str, Any],
) -> None:
    path = run_dir / "resolved_protocol.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"resume requires an existing resolved protocol: {path}"
        )
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read existing protocol: {exc}") from exc
    if existing != protocol:
        raise ValueError(
            "resume protocol does not exactly match the existing run."
        )
    status_path = run_dir / "run_status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "completed":
            raise ValueError("completed runs cannot be resumed.")


def mark_experiment_failed(
    run_dir: str | Path,
    error: BaseException,
    *,
    run_id: str | None = None,
    traceback_text: str | None = None,
) -> None:
    """Record a top-level failure without replacing a completed result."""

    source = Path(run_dir).resolve()
    source.mkdir(parents=True, exist_ok=True)
    status_path = source / "run_status.json"
    if status_path.is_file():
        try:
            existing = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if existing.get("status") == "completed":
            return
    protocol_path = source / "resolved_protocol.json"
    protocol: dict[str, Any] = {}
    if protocol_path.is_file():
        try:
            loaded = json.loads(protocol_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                protocol = loaded
        except (OSError, json.JSONDecodeError):
            pass
    payload = {
        "status": "failed",
        "run_id": run_id or protocol.get("run_id") or source.name,
        "run_signature": protocol.get("run_signature"),
        "error_type": type(error).__name__,
        "error": str(error),
    }
    if traceback_text:
        payload["traceback"] = traceback_text
    write_json(payload, status_path)


__all__ = [
    "ExperimentRunSession",
    "mark_experiment_failed",
    "start_experiment_run",
]
