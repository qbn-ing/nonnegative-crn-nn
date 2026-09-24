from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from training.loop import EpochResult, FitHistory
from utils.environment import save_environment_info
from utils.paths import normalize_path
from utils.serialization import to_jsonable, write_json

from .checkpoint import save_checkpoint
from .history import (
    HistoryPaths,
    epoch_result_to_dict,
    save_fit_history,
)


@dataclass
class RunRecorder:
    """Small experiment recorder with callback-compatible methods."""

    output_dir: Path
    events_path: Path
    status_path: Path | None
    _finished: bool = field(default=False, init=False, repr=False)

    def __enter__(self) -> RunRecorder:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: Any,
    ) -> bool:
        if exception is None:
            self.mark_completed()
        else:
            self.mark_failed(exception)
        return False

    def log_event(
        self,
        event: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(event, str) or not event.strip():
            raise ValueError("event must be a non-empty string.")
        if payload is not None and not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping or None.")
        row = {
            "recorded_at_utc": _now(),
            "event": event,
            "payload": (
                None if payload is None else to_jsonable(payload)
            ),
        }
        with self.events_path.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as file:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            file.write("\n")

    def record_epoch(
        self,
        epoch: int,
        train: EpochResult,
        validation: EpochResult | None,
    ) -> None:
        """Callback accepted directly by ``training.fit``."""

        self.log_event(
            "epoch_completed",
            {
                "epoch": int(epoch),
                "train": epoch_result_to_dict(train),
                "validation": (
                    None
                    if validation is None
                    else epoch_result_to_dict(validation)
                ),
            },
        )

    def record_stage(
        self,
        stage_result: Any,
        model: torch.nn.Module,
    ) -> None:
        """Callback accepted directly by depth continuation."""

        to_dict = getattr(stage_result, "to_dict", None)
        if not callable(to_dict):
            raise TypeError("stage_result must expose to_dict().")
        self.log_event(
            "continuation_stage_completed",
            {
                "stage": to_dict(),
                "model_class": type(model).__name__,
            },
        )

    def save_history(
        self,
        history: FitHistory,
    ) -> HistoryPaths:
        paths = save_fit_history(history, self.output_dir)
        self.log_event("history_saved", paths.to_dict())
        return paths

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        *,
        filename: str = "checkpoint.pt",
        **kwargs: Any,
    ) -> Path:
        path = save_checkpoint(
            self.output_dir / filename,
            model,
            **kwargs,
        )
        self.log_event(
            "checkpoint_saved",
            {"path": path.name},
        )
        return path

    def save_summary(
        self,
        summary: Mapping[str, Any],
        *,
        filename: str = "summary.json",
    ) -> Path:
        if not isinstance(summary, Mapping):
            raise TypeError("summary must be a mapping.")
        path = write_json(summary, self.output_dir / filename)
        self.log_event("summary_saved", {"path": path.name})
        return path

    def mark_completed(
        self,
        summary: Mapping[str, Any] | None = None,
    ) -> None:
        if self._finished:
            return
        self.log_event("run_completed", summary)
        self._write_status("completed", summary=summary)
        self._finished = True

    def mark_failed(self, error: BaseException) -> None:
        if self._finished:
            return
        payload = {
            "error_type": type(error).__name__,
            "message": str(error),
        }
        self.log_event("run_failed", payload)
        self._write_status("failed", summary=payload)
        self._finished = True

    def _write_status(
        self,
        status: str,
        *,
        summary: Mapping[str, Any] | None = None,
    ) -> None:
        if self.status_path is None:
            return
        write_json(
            {
                "status": status,
                "updated_at_utc": _now(),
                "summary": (
                    None
                    if summary is None
                    else to_jsonable(summary)
                ),
            },
            self.status_path,
        )


def start_run(
    output_dir: str | os.PathLike[str],
    *,
    config: Mapping[str, Any],
    project_root: str | os.PathLike[str] | None = None,
    seed_report: Mapping[str, Any] | None = None,
    include_packages: bool = True,
    overwrite: bool = False,
    resume: bool = False,
    manage_status: bool = True,
) -> RunRecorder:
    """Create a run directory and save environment/config before training."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping.")
    if seed_report is not None and not isinstance(
        seed_report,
        Mapping,
    ):
        raise TypeError("seed_report must be a mapping or None.")
    if not isinstance(include_packages, bool):
        raise TypeError("include_packages must be a bool.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool.")
    if not isinstance(resume, bool):
        raise TypeError("resume must be a bool.")
    if not isinstance(manage_status, bool):
        raise TypeError("manage_status must be a bool.")
    if overwrite and resume:
        raise ValueError("overwrite and resume are mutually exclusive.")

    output_path = normalize_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    managed_paths: dict[str, Path] = {
        "environment": output_path / "environment.json",
        "config": output_path / "config.json",
        "events": output_path / "events.jsonl",
    }
    if manage_status:
        managed_paths["status"] = output_path / "status.json"
    existing = [
        path.name for path in managed_paths.values() if path.exists()
    ]
    if existing and not overwrite and not resume:
        raise FileExistsError(
            "Run directory already contains managed files: "
            f"{sorted(existing)}."
        )

    if resume:
        if not managed_paths["config"].is_file():
            raise FileNotFoundError("resume requires config.json.")
        existing_config = json.loads(
            managed_paths["config"].read_text(encoding="utf-8")
        )
        if existing_config != to_jsonable(config):
            raise ValueError("resume config does not match the existing run.")
        if not managed_paths["events"].is_file():
            raise FileNotFoundError("resume requires events.jsonl.")
    else:
        save_environment_info(
            managed_paths["environment"],
            project_root=project_root,
            include_packages=include_packages,
            extra=(
                None
                if seed_report is None
                else {"seed_report": seed_report}
            ),
        )
        write_json(config, managed_paths["config"])
        managed_paths["events"].write_text("", encoding="utf-8")

    recorder = RunRecorder(
        output_dir=output_path,
        events_path=managed_paths["events"],
        status_path=managed_paths.get("status"),
    )
    if manage_status:
        recorder._write_status("running")
    recorder.log_event(
        "run_resumed" if resume else "run_started",
        {
            "environment": managed_paths["environment"].name,
            "config": managed_paths["config"].name,
        },
    )
    return recorder


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "RunRecorder",
    "start_run",
]
