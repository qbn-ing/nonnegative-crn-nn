from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class MetricAggregate:
    n: int
    mean: float
    standard_deviation: float
    standard_error: float
    median: float
    q1: float
    q3: float
    iqr: float
    ci95_low: float
    ci95_high: float
    ci95_method: str
    minimum: float
    maximum: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "n": self.n,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "standard_error": self.standard_error,
            "median": self.median,
            "q1": self.q1,
            "q3": self.q3,
            "iqr": self.iqr,
            "ci95_low": self.ci95_low,
            "ci95_high": self.ci95_high,
            "ci95_method": self.ci95_method,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass(frozen=True)
class PairedComparison:
    metric: str
    n_pairs: int
    n_wins: int
    n_losses: int
    n_ties: int
    win_rate_excluding_ties: float | None
    mean_difference: float
    standard_deviation: float
    sign_test_pvalue: float | None
    differences: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "n_pairs": self.n_pairs,
            "n_wins": self.n_wins,
            "n_losses": self.n_losses,
            "n_ties": self.n_ties,
            "win_rate_excluding_ties": self.win_rate_excluding_ties,
            "mean_difference": self.mean_difference,
            "standard_deviation": self.standard_deviation,
            "sign_test_pvalue": self.sign_test_pvalue,
            "differences": list(self.differences),
        }


def aggregate_metric(values: Sequence[float]) -> MetricAggregate:
    normalized = _finite_values(values)
    n = len(normalized)
    mean = sum(normalized) / n
    variance = (
        sum((value - mean) ** 2 for value in normalized) / (n - 1)
        if n > 1
        else 0.0
    )
    standard_deviation = math.sqrt(variance)
    standard_error = standard_deviation / math.sqrt(n)
    ordered = tuple(sorted(normalized))
    q1 = _linear_quantile(ordered, 0.25)
    median = _linear_quantile(ordered, 0.50)
    q3 = _linear_quantile(ordered, 0.75)
    critical = _student_t_975(n - 1) if n > 1 else 0.0
    half_width = critical * standard_error
    return MetricAggregate(
        n=n,
        mean=float(mean),
        standard_deviation=float(standard_deviation),
        standard_error=float(standard_error),
        median=float(median),
        q1=float(q1),
        q3=float(q3),
        iqr=float(q3 - q1),
        ci95_low=float(mean - half_width),
        ci95_high=float(mean + half_width),
        ci95_method=("undefined_singleton" if n == 1 else "student_t"),
        minimum=float(ordered[0]),
        maximum=float(ordered[-1]),
    )


def _linear_quantile(ordered: Sequence[float], probability: float) -> float:
    """Return the type-7/NumPy-default sample quantile."""

    if not ordered:
        raise ValueError("ordered values must not be empty.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1].")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(
        (1.0 - fraction) * ordered[lower] + fraction * ordered[upper]
    )


def _student_t_975(degrees_of_freedom: int) -> float:
    """Two-sided 95% critical value without a SciPy runtime dependency."""

    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive.")
    table = (
        12.706,
        4.303,
        3.182,
        2.776,
        2.571,
        2.447,
        2.365,
        2.306,
        2.262,
        2.228,
        2.201,
        2.179,
        2.160,
        2.145,
        2.131,
        2.120,
        2.110,
        2.101,
        2.093,
        2.086,
        2.080,
        2.074,
        2.069,
        2.064,
        2.060,
        2.056,
        2.052,
        2.048,
        2.045,
        2.042,
    )
    if degrees_of_freedom <= len(table):
        return table[degrees_of_freedom - 1]
    if degrees_of_freedom <= 40:
        return 2.021
    if degrees_of_freedom <= 60:
        return 2.000
    if degrees_of_freedom <= 120:
        return 1.980
    return 1.960


def compare_paired_metric(
    baseline: Mapping[str, float],
    treatment: Mapping[str, float],
    *,
    metric: str,
    tie_tolerance: float = 0.0,
) -> PairedComparison:
    """Compare matched cells identified by the same run key."""

    if not isinstance(metric, str) or not metric:
        raise ValueError("metric must be a non-empty string.")
    if tie_tolerance < 0.0 or not math.isfinite(tie_tolerance):
        raise ValueError("tie_tolerance must be finite and nonnegative.")
    baseline_keys = set(baseline)
    treatment_keys = set(treatment)
    if baseline_keys != treatment_keys:
        raise ValueError(
            "Paired comparison requires identical run keys; "
            f"baseline_only={sorted(baseline_keys - treatment_keys)}, "
            f"treatment_only={sorted(treatment_keys - baseline_keys)}."
        )
    if not baseline_keys:
        raise ValueError("Paired comparison requires at least one run.")
    differences: list[float] = []
    for key in sorted(baseline_keys):
        before = float(baseline[key])
        after = float(treatment[key])
        if not math.isfinite(before) or not math.isfinite(after):
            raise ValueError("Paired metric values must be finite.")
        differences.append(after - before)
    wins = sum(value > tie_tolerance for value in differences)
    losses = sum(value < -tie_tolerance for value in differences)
    ties = len(differences) - wins - losses
    non_ties = wins + losses
    aggregate = aggregate_metric(differences)
    return PairedComparison(
        metric=metric,
        n_pairs=len(differences),
        n_wins=wins,
        n_losses=losses,
        n_ties=ties,
        win_rate_excluding_ties=(
            None if non_ties == 0 else wins / non_ties
        ),
        mean_difference=aggregate.mean,
        standard_deviation=aggregate.standard_deviation,
        sign_test_pvalue=(
            None
            if non_ties == 0
            else exact_sign_test_pvalue(wins, losses)
        ),
        differences=tuple(differences),
    )


def exact_sign_test_pvalue(n_wins: int, n_losses: int) -> float:
    """Two-sided exact binomial sign-test p-value, excluding ties."""

    for name, value in (("n_wins", n_wins), ("n_losses", n_losses)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int.")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative.")
    n = n_wins + n_losses
    if n == 0:
        raise ValueError("At least one non-tied pair is required.")
    tail = min(n_wins, n_losses)
    one_sided = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return float(min(1.0, 2.0 * one_sided))


def _finite_values(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("values must be a numeric sequence.")
    normalized = tuple(float(value) for value in values)
    if not normalized:
        raise ValueError("values must not be empty.")
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("values must be finite.")
    return normalized


__all__ = [
    "MetricAggregate",
    "PairedComparison",
    "aggregate_metric",
    "compare_paired_metric",
    "exact_sign_test_pvalue",
]
