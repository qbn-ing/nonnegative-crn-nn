from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal


TargetMode = Literal["soft", "hard"]


@dataclass(frozen=True)
class StepProtocol:
    """Global optimizer-step protocol shared by direct and continuation runs."""

    total_steps: int = 8_000
    peak_lr: float = 3e-3
    warmup_steps: int = 400
    min_lr_ratio: float = 0.01
    soft_fraction: float = 0.35
    beta_min: float = 1.0
    beta_max: float = 32.0

    def __post_init__(self) -> None:
        if isinstance(self.total_steps, bool) or self.total_steps <= 1:
            raise ValueError("total_steps must be an integer greater than 1.")
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("warmup_steps must lie inside total_steps.")
        if self.peak_lr <= 0.0:
            raise ValueError("peak_lr must be positive.")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must lie in [0, 1].")
        if not 0.0 < self.soft_fraction < 1.0:
            raise ValueError("soft_fraction must lie inside (0, 1).")
        if self.beta_min <= 0.0 or self.beta_max < self.beta_min:
            raise ValueError("Invalid curriculum beta range.")
        if self.soft_steps >= self.total_steps:
            raise ValueError("The soft curriculum must end before training.")

    @property
    def soft_steps(self) -> int:
        return max(1, int(round(self.total_steps * self.soft_fraction)))

    def learning_rate(self, step: int) -> float:
        return learning_rate_at_step(
            step,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            peak_lr=self.peak_lr,
            min_lr_ratio=self.min_lr_ratio,
        )

    def curriculum(self, step: int) -> tuple[TargetMode, float | None]:
        return curriculum_at_step(
            step,
            soft_steps=self.soft_steps,
            beta_min=self.beta_min,
            beta_max=self.beta_max,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["soft_steps"] = self.soft_steps
        return payload


def learning_rate_at_step(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr_ratio: float,
) -> float:
    """Linear warm-up followed by one global cosine decay."""

    if not 0 <= step < total_steps:
        raise ValueError("step must lie inside the optimizer budget.")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps is invalid.")
    if peak_lr <= 0.0 or not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("Learning-rate parameters are invalid.")
    if warmup_steps and step < warmup_steps:
        return float(peak_lr * (step + 1) / warmup_steps)
    decay_steps = total_steps - warmup_steps
    decay_index = step - warmup_steps
    progress = (
        1.0
        if decay_steps <= 1
        else decay_index / (decay_steps - 1)
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(
        peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)
    )


def curriculum_at_step(
    step: int,
    *,
    soft_steps: int,
    beta_min: float,
    beta_max: float,
) -> tuple[TargetMode, float | None]:
    """Return soft/hard target mode for a zero-based optimizer step."""

    if step < 0:
        raise ValueError("step must be nonnegative.")
    if soft_steps <= 0:
        raise ValueError("soft_steps must be positive.")
    if beta_min <= 0.0 or beta_max < beta_min:
        raise ValueError("Invalid curriculum beta range.")
    if step >= soft_steps:
        return "hard", None
    progress = 1.0 if soft_steps == 1 else step / (soft_steps - 1)
    weight = 0.5 * (1.0 - math.cos(math.pi * progress))
    beta = beta_min + (beta_max - beta_min) * weight
    return "soft", float(beta)


def continuation_depths(depth: int) -> tuple[int, ...]:
    """Resolve the D2 anchor and intermediate effective-depth stages."""

    return resolve_continuation_ladder(depth, midpoint=True)


def resolve_continuation_ladder(
    depth: int,
    *,
    anchors: Iterable[int] = (2,),
    midpoint: bool = False,
    single_depth_repeats: int = 1,
) -> tuple[int, ...]:
    """Resolve an increasing depth ladder from reusable anchors or a midpoint."""

    if isinstance(depth, bool) or not isinstance(depth, int):
        raise TypeError("depth must be an int.")
    if depth <= 0:
        raise ValueError("depth must be positive.")
    if single_depth_repeats <= 0:
        raise ValueError("single_depth_repeats must be positive.")
    if depth == 1:
        return (1,) * single_depth_repeats
    resolved = sorted(
        {
            int(anchor)
            for anchor in anchors
            if 0 < int(anchor) < depth
        }
    )
    if midpoint and depth > 3:
        resolved.append(max(3, depth // 2))
    resolved.append(depth)
    return tuple(sorted(set(resolved)))


def continuation_step_budgets(
    *,
    total_steps: int,
    soft_steps: int,
    n_stages: int | None = None,
    ladder: tuple[int, ...] | None = None,
    prefinal_weights: Iterable[int] | None = None,
) -> tuple[int, ...]:
    """Allocate one global budget and reach final depth at soft→hard switch."""

    if total_steps <= 0 or not 0 < soft_steps < total_steps:
        raise ValueError("Invalid total_steps/soft_steps.")
    if ladder is not None:
        if not ladder:
            raise ValueError("ladder must not be empty.")
        if n_stages is not None and n_stages != len(ladder):
            raise ValueError("n_stages and ladder length disagree.")
        n_stages = len(ladder)
    if n_stages is None or n_stages <= 0:
        raise ValueError("n_stages must be positive.")
    if n_stages == 1:
        return (total_steps,)
    prefinal_count = n_stages - 1
    if prefinal_weights is None:
        weights = (1,) * prefinal_count
    else:
        weights = tuple(int(value) for value in prefinal_weights)
        if len(weights) != prefinal_count or any(value <= 0 for value in weights):
            raise ValueError(
                "prefinal_weights must contain one positive value per "
                "pre-final stage."
            )
    if soft_steps < prefinal_count:
        raise ValueError("soft_steps is too small for the continuation ladder.")
    weight_sum = sum(weights)
    raw = [soft_steps * weight / weight_sum for weight in weights]
    allocated = [max(1, int(math.floor(value))) for value in raw]
    remainder = soft_steps - sum(allocated)
    fractional_order = sorted(
        range(prefinal_count),
        key=lambda index: (raw[index] - math.floor(raw[index]), -index),
        reverse=True,
    )
    if remainder > 0:
        for index in range(remainder):
            allocated[fractional_order[index % prefinal_count]] += 1
    elif remainder < 0:
        removable = [index for index in reversed(fractional_order)]
        for _ in range(-remainder):
            candidate = next(
                (index for index in removable if allocated[index] > 1),
                None,
            )
            if candidate is None:
                raise ValueError("cannot allocate positive stage budgets.")
            allocated[candidate] -= 1
    return tuple(allocated) + (total_steps - soft_steps,)


def optimizer_reset_steps(
    stage_steps: tuple[int, ...],
) -> tuple[int, ...]:
    if not stage_steps or any(step <= 0 for step in stage_steps):
        raise ValueError("stage_steps must contain positive integers.")
    resets: list[int] = []
    completed = 0
    for steps in stage_steps[:-1]:
        completed += steps
        resets.append(completed)
    return tuple(resets)


__all__ = [
    "StepProtocol",
    "TargetMode",
    "continuation_depths",
    "continuation_step_budgets",
    "curriculum_at_step",
    "learning_rate_at_step",
    "optimizer_reset_steps",
    "resolve_continuation_ladder",
]
