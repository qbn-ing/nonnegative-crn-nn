from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ExperimentProtocol:
    """Checkpoint-selection rules actually executed by the shared runner."""

    minimum_selection_step: int = 0
    require_final_depth: bool = True
    test_evaluation_limit: int = 1

    def __post_init__(self) -> None:
        if self.minimum_selection_step < 0:
            raise ValueError("minimum_selection_step must be nonnegative.")
        if self.test_evaluation_limit != 1:
            raise ValueError(
                "The paper protocol evaluates the test set exactly once."
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointDecision:
    selected: bool
    reason: str
    best_loss: float | None
    best_step: int | None


class ValidationCheckpointSelector:
    """State-only minimum-validation-loss selector.

    Model/optimizer/RNG snapshots remain the responsibility of the checkpoint
    module; this class owns only the experiment selection rule.
    """

    def __init__(self, protocol: ExperimentProtocol) -> None:
        if not isinstance(protocol, ExperimentProtocol):
            raise TypeError("protocol must be an ExperimentProtocol.")
        self.protocol = protocol
        self.best_loss: float | None = None
        self.best_step: int | None = None
        self.test_evaluations = 0

    def consider(
        self,
        *,
        global_step: int,
        validation_loss: float,
        at_final_depth: bool,
    ) -> CheckpointDecision:
        if global_step < 0:
            raise ValueError("global_step must be nonnegative.")
        loss = float(validation_loss)
        if not math.isfinite(loss):
            raise ValueError("validation_loss must be finite.")
        if global_step < self.protocol.minimum_selection_step:
            return self._decision(False, "before_selection_window")
        if self.protocol.require_final_depth and not at_final_depth:
            return self._decision(False, "not_final_depth")
        if self.best_loss is not None and loss >= self.best_loss:
            return self._decision(False, "not_improved")
        self.best_loss = loss
        self.best_step = global_step
        return self._decision(True, "improved")

    def register_test_evaluation(self) -> None:
        if self.test_evaluations >= self.protocol.test_evaluation_limit:
            raise RuntimeError(
                "The test set has already been evaluated for this run."
            )
        self.test_evaluations += 1

    def _decision(self, selected: bool, reason: str) -> CheckpointDecision:
        return CheckpointDecision(
            selected=selected,
            reason=reason,
            best_loss=self.best_loss,
            best_step=self.best_step,
        )


def select_binary_threshold_by_balanced_accuracy(
    y_true: Sequence[int] | np.ndarray,
    positive_scores: Sequence[float] | np.ndarray,
    *,
    candidates: Sequence[float] | np.ndarray | None = None,
) -> tuple[float, float]:
    """Select on validation data only; ties prefer the threshold nearest 0.5."""

    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    scores = np.asarray(positive_scores, dtype=float).reshape(-1)
    if truth.size == 0 or truth.size != scores.size:
        raise ValueError("y_true and positive_scores must be aligned and non-empty.")
    if set(np.unique(truth).tolist()) != {0, 1}:
        raise ValueError("Binary threshold selection requires labels {0, 1}.")
    if not np.isfinite(scores).all():
        raise ValueError("positive_scores must be finite.")
    if candidates is None:
        thresholds = np.unique(
            np.concatenate(([0.0, 0.5, 1.0], scores))
        )
    else:
        thresholds = np.asarray(candidates, dtype=float).reshape(-1)
        if thresholds.size == 0 or not np.isfinite(thresholds).all():
            raise ValueError("candidates must contain finite values.")
    best: tuple[float, float, float] | None = None
    for threshold in thresholds:
        prediction = (scores >= threshold).astype(np.int64)
        true_negative = int(((truth == 0) & (prediction == 0)).sum())
        false_positive = int(((truth == 0) & (prediction == 1)).sum())
        true_positive = int(((truth == 1) & (prediction == 1)).sum())
        false_negative = int(((truth == 1) & (prediction == 0)).sum())
        specificity = true_negative / (true_negative + false_positive)
        sensitivity = true_positive / (true_positive + false_negative)
        bacc = 0.5 * (specificity + sensitivity)
        candidate = (bacc, -abs(float(threshold) - 0.5), -float(threshold))
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    selected_threshold = -best[2]
    return selected_threshold, best[0]


__all__ = [
    "CheckpointDecision",
    "ExperimentProtocol",
    "ValidationCheckpointSelector",
    "select_binary_threshold_by_balanced_accuracy",
]
