from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Configuration for validation-driven early stopping.

    The monitored value is selected from an epoch result metric dictionary.
    For classification experiments, ``monitor='classification'`` with
    ``mode='min'`` is usually the safest default because it tracks the
    supervised loss on the validation split.
    """

    enabled: bool = False
    monitor: str = "classification"
    mode: str = "min"
    patience: int = 20
    min_delta: float = 0.0
    restore_best: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(
                "enabled must be a bool. "
                f"Got {type(self.enabled).__name__}."
            )

        if not isinstance(self.monitor, str) or not self.monitor:
            raise ValueError("monitor must be a non-empty string.")

        if self.mode not in {"min", "max"}:
            raise ValueError(
                f"mode must be 'min' or 'max'. Got {self.mode!r}."
            )

        if not isinstance(self.patience, int):
            raise TypeError(
                "patience must be an int. "
                f"Got {type(self.patience).__name__}."
            )
        if self.patience < 0:
            raise ValueError(f"patience must be nonnegative, got {self.patience}.")

        if self.min_delta < 0.0:
            raise ValueError(f"min_delta must be nonnegative, got {self.min_delta}.")

        if not isinstance(self.restore_best, bool):
            raise TypeError(
                "restore_best must be a bool. "
                f"Got {type(self.restore_best).__name__}."
            )


@dataclass(frozen=True)
class EarlyStoppingResult:
    """Final early-stopping state serialized into FitHistory."""

    enabled: bool
    monitor: str
    mode: str
    patience: int
    min_delta: float
    restore_best: bool
    stopped_early: bool = False
    best_epoch: int | None = None
    best_score: float | None = None
    stop_epoch: int | None = None
    num_bad_epochs: int = 0
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "restore_best": self.restore_best,
            "stopped_early": self.stopped_early,
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "stop_epoch": self.stop_epoch,
            "num_bad_epochs": self.num_bad_epochs,
            "reason": self.reason,
        }


class EarlyStoppingTracker:
    """Stateful monitor used by training loops."""

    def __init__(self, config: EarlyStoppingConfig) -> None:
        if not isinstance(config, EarlyStoppingConfig):
            raise TypeError(
                "config must be an EarlyStoppingConfig. "
                f"Got {type(config).__name__}."
            )
        self.config = config
        self.best_epoch: int | None = None
        self.best_score: float | None = None
        self.num_bad_epochs = 0
        self.stopped_early = False
        self.stop_epoch: int | None = None
        self.reason: str | None = None

    def update(self, metrics: Mapping[str, Any], *, epoch: int) -> bool:
        """Update the tracker and return True when training should stop."""

        if not self.config.enabled:
            return False

        if epoch <= 0:
            raise ValueError(f"epoch must be 1-based and positive, got {epoch}.")

        if self.config.monitor not in metrics:
            raise KeyError(
                f"Early-stopping metric {self.config.monitor!r} was not found. "
                f"Available metrics: {sorted(str(key) for key in metrics.keys())}."
            )

        raw_score = metrics[self.config.monitor]
        if raw_score is None:
            raise ValueError(
                f"Early-stopping metric {self.config.monitor!r} is None at epoch {epoch}."
            )

        score = float(raw_score)

        if self._is_improvement(score):
            self.best_score = score
            self.best_epoch = int(epoch)
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        if self.num_bad_epochs > self.config.patience:
            self.stopped_early = True
            self.stop_epoch = int(epoch)
            self.reason = (
                f"No improvement in {self.config.monitor!r} for "
                f"{self.num_bad_epochs} consecutive epoch(s)."
            )
            return True

        return False

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True

        if self.config.mode == "min":
            return score < self.best_score - self.config.min_delta

        return score > self.best_score + self.config.min_delta

    def result(self) -> EarlyStoppingResult:
        return EarlyStoppingResult(
            enabled=self.config.enabled,
            monitor=self.config.monitor,
            mode=self.config.mode,
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            restore_best=self.config.restore_best,
            stopped_early=self.stopped_early,
            best_epoch=self.best_epoch,
            best_score=self.best_score,
            stop_epoch=self.stop_epoch,
            num_bad_epochs=self.num_bad_epochs,
            reason=self.reason,
        )
