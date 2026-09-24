from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Iterable, Literal

from experiments.common.signatures import (
    scientific_signature,
    signed_run_id,
    validate_run_identity,
)


DatasetName = Literal[
    "iris",
    "wine_binary",
    "wine_full",
    "breast_cancer",
    "circles",
    "st003390_m1m2",
]
Representation = Literal["raw", "quadratic"]

DATASETS: tuple[DatasetName, ...] = (
    "iris",
    "wine_binary",
    "wine_full",
    "breast_cancer",
    "circles",
    "st003390_m1m2",
)
DEFAULT_DATASETS: tuple[DatasetName, ...] = (
    "iris",
    "wine_binary",
    "breast_cancer",
    "circles",
    "st003390_m1m2",
)
REPRESENTATIONS: tuple[Representation, ...] = ("raw", "quadratic")


@dataclass(frozen=True)
class CircleConfig:
    """Circle generator values plus an explicit preregistration flag."""

    n_samples: int = 2_000
    factor: float = 0.5
    noise: float = 0.08
    generator_seed: int = 0
    parameters_frozen: bool = False

    def __post_init__(self) -> None:
        if self.n_samples < 20:
            raise ValueError("circle n_samples must be at least 20.")
        if not 0.0 < self.factor < 1.0:
            raise ValueError("circle factor must lie inside (0, 1).")
        if self.noise < 0.0:
            raise ValueError("circle noise must be nonnegative.")
        if self.generator_seed < 0:
            raise ValueError("circle generator_seed must be nonnegative.")
        if not isinstance(self.parameters_frozen, bool):
            raise TypeError("parameters_frozen must be a bool.")


@dataclass(frozen=True)
class ExperimentProfile:
    """Dataset grid and shared D1 optimization protocol."""

    name: str
    datasets: tuple[DatasetName, ...]
    representations: tuple[Representation, ...] = REPRESENTATIONS
    repeats: int = 1
    outer_folds: int = 5
    validation_fraction: float = 0.20
    base_seed: int = 0
    total_steps: int = 1_000
    batch_size: int = 64
    learning_rate: float = 1.5e-3
    warmup_steps: int = 50
    min_lr_ratio: float = 0.01
    soft_fraction: float = 0.35
    label_smoothing_start: float = 0.10
    eval_interval: int = 50
    grad_clip: float = 10.0
    xavier_gain: float = 1.0
    st_missing_fraction: float = 0.20
    st_scale_quantile: float = 0.90
    circle: CircleConfig = CircleConfig()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("profile name must not be empty.")
        _unique_members("datasets", self.datasets, DATASETS)
        _unique_members(
            "representations", self.representations, REPRESENTATIONS
        )
        for name in ("repeats", "outer_folds", "total_steps", "batch_size"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.outer_folds < 2:
            raise ValueError("outer_folds must be at least 2.")
        if self.base_seed < 0:
            raise ValueError("base_seed must be nonnegative.")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must lie inside (0, 0.5).")
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("warmup_steps must lie inside total_steps.")
        if not 0.0 < self.soft_fraction < 1.0:
            raise ValueError("soft_fraction must lie inside (0, 1).")
        if self.soft_steps >= self.total_steps:
            raise ValueError("soft curriculum must end before training.")
        for name in (
            "learning_rate",
            "min_lr_ratio",
            "grad_clip",
            "xavier_gain",
            "st_scale_quantile",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if not 0.0 <= self.label_smoothing_start < 1.0:
            raise ValueError("label_smoothing_start must lie in [0, 1).")
        if not 0.0 <= self.st_missing_fraction < 1.0:
            raise ValueError("st_missing_fraction must lie in [0, 1).")
        if not 0.0 < self.st_scale_quantile <= 1.0:
            raise ValueError("st_scale_quantile must lie inside (0, 1].")
        if not isinstance(self.circle, CircleConfig):
            raise TypeError("circle must be a CircleConfig.")

    @property
    def soft_steps(self) -> int:
        return max(1, int(round(self.total_steps * self.soft_fraction)))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update({"schema_version": 1, "soft_steps": self.soft_steps})
        return payload


@dataclass(frozen=True)
class RunSpec:
    """One paired dataset/representation/fold D1 run."""

    profile_name: str
    dataset: DatasetName
    representation: Representation
    repeat: int
    fold: int
    outer_folds: int
    validation_fraction: float
    split_seed: int
    model_seed: int
    total_steps: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    min_lr_ratio: float
    soft_fraction: float
    label_smoothing_start: float
    eval_interval: int
    grad_clip: float
    xavier_gain: float
    st_missing_fraction: float
    st_scale_quantile: float
    circle: CircleConfig

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError("unsupported dataset.")
        if self.representation not in REPRESENTATIONS:
            raise ValueError("unsupported representation.")
        if self.repeat < 0 or not 0 <= self.fold < self.outer_folds:
            raise ValueError("invalid repeat/fold indices.")
        if self.split_seed < 0 or self.model_seed < 0:
            raise ValueError("seeds must be nonnegative.")
        if self.total_steps <= 1:
            raise ValueError("total_steps must be greater than one.")

    @property
    def depth(self) -> int:
        return 1

    @property
    def soft_steps(self) -> int:
        return max(1, int(round(self.total_steps * self.soft_fraction)))

    @property
    def batch_seed(self) -> int:
        return 100_000 + self.model_seed

    @property
    def run_id(self) -> str:
        base = (
            f"{self.dataset}_{self.representation}_d1_"
            f"r{self.repeat:02d}f{self.fold:02d}s{self.model_seed:03d}"
        )
        return signed_run_id(base, asdict(self))

    @property
    def run_signature(self) -> str:
        return scientific_signature(asdict(self))

    @property
    def paired_cell(self) -> tuple[str, int, int]:
        return (self.dataset, self.repeat, self.fold)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": 1,
                "experiment_kind": "single_layer",
                "run_id": self.run_id,
                "run_signature": self.run_signature,
                "depth": 1,
                "batch_seed": self.batch_seed,
                "basis_definition": (
                    "raw=[x_i]; quadratic=[x_i,x_i^2,x_i*x_j(i<j)]; "
                    "neither includes a constant basis term"
                ),
                "bias_definition": (
                    "each terminal E/I-to-r unit has trainable nonnegative "
                    "affine E and I biases"
                ),
                "depth_definition": (
                    "D counts E/I-to-r layers including terminal class "
                    "evidence; competitive O is excluded"
                ),
            }
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunSpec:
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported single-layer run schema.")
        names = {item.name for item in fields(cls)}
        missing = sorted(names - set(payload))
        if missing:
            raise ValueError("run specification is missing: " + ", ".join(missing))
        values = {name: payload[name] for name in names}
        values["circle"] = CircleConfig(**values["circle"])
        instance = cls(**values)
        validate_run_identity(instance, payload)
        return instance


def _unique_members(name: str, values: Iterable[str], valid: Iterable[str]) -> None:
    items = tuple(values)
    allowed = set(valid)
    if not items or len(items) != len(set(items)):
        raise ValueError(f"{name} must be non-empty and unique.")
    unknown = sorted(set(items) - allowed)
    if unknown:
        raise ValueError(f"{name} contains unsupported values: {unknown}.")


PROFILES: dict[str, ExperimentProfile] = {
    "smoke": ExperimentProfile(
        name="smoke",
        datasets=("iris", "circles"),
        repeats=1,
        outer_folds=2,
        total_steps=12,
        batch_size=32,
        warmup_steps=2,
        eval_interval=4,
        circle=CircleConfig(n_samples=200, factor=0.5, noise=0.08),
    ),
    "pilot": ExperimentProfile(
        name="pilot",
        datasets=DEFAULT_DATASETS,
        repeats=1,
        outer_folds=5,
        total_steps=1_000,
        batch_size=64,
        warmup_steps=50,
        eval_interval=50,
    ),
    "circles_main": ExperimentProfile(
        name="circles_main",
        datasets=("circles",),
        repeats=2,
        outer_folds=5,
        total_steps=8_000,
        batch_size=64,
        warmup_steps=400,
        eval_interval=100,
        circle=CircleConfig(
            n_samples=4_000,
            factor=0.5,
            noise=0.08,
            generator_seed=0,
            parameters_frozen=True,
        ),
    ),
    "st003390_main": ExperimentProfile(
        name="st003390_main",
        datasets=("st003390_m1m2",),
        repeats=2,
        outer_folds=5,
        total_steps=8_000,
        batch_size=64,
        warmup_steps=400,
        eval_interval=100,
    ),
    "supplement_main": ExperimentProfile(
        name="supplement_main",
        datasets=("iris", "wine_binary", "breast_cancer"),
        repeats=2,
        outer_folds=5,
        total_steps=8_000,
        batch_size=64,
        warmup_steps=400,
        eval_interval=100,
    ),
}


def get_profile(name: str) -> ExperimentProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown profile: {name!r}.") from exc


def override_profile(
    profile: ExperimentProfile,
    *,
    datasets: tuple[str, ...] | None = None,
    representations: tuple[str, ...] | None = None,
    repeats: int | None = None,
    outer_folds: int | None = None,
    total_steps: int | None = None,
    circle_n_samples: int | None = None,
    circle_factor: float | None = None,
    circle_noise: float | None = None,
    circle_seed: int | None = None,
) -> ExperimentProfile:
    updates: dict[str, Any] = {}
    if datasets is not None:
        updates["datasets"] = tuple(datasets)
    if representations is not None:
        updates["representations"] = tuple(representations)
    for name, value in (
        ("repeats", repeats),
        ("outer_folds", outer_folds),
    ):
        if value is not None:
            updates[name] = value
    if total_steps is not None:
        updates["total_steps"] = total_steps
        if profile.warmup_steps >= total_steps:
            updates["warmup_steps"] = max(0, min(total_steps - 1, total_steps // 20))
    circle_updates = {
        "n_samples": circle_n_samples,
        "factor": circle_factor,
        "noise": circle_noise,
        "generator_seed": circle_seed,
    }
    if any(value is not None for value in circle_updates.values()):
        updates["circle"] = replace(
            profile.circle,
            parameters_frozen=False,
            **{
                name: value
                for name, value in circle_updates.items()
                if value is not None
            },
        )
    return replace(profile, **updates)


def build_run_plan(profile: ExperimentProfile) -> tuple[RunSpec, ...]:
    runs: list[RunSpec] = []
    for dataset in profile.datasets:
        for representation in profile.representations:
            for repeat in range(profile.repeats):
                split_seed = profile.base_seed + repeat
                for fold in range(profile.outer_folds):
                    runs.append(
                        RunSpec(
                            profile_name=profile.name,
                            dataset=dataset,
                            representation=representation,
                            repeat=repeat,
                            fold=fold,
                            outer_folds=profile.outer_folds,
                            validation_fraction=profile.validation_fraction,
                            split_seed=split_seed,
                            model_seed=profile.base_seed + fold,
                            total_steps=profile.total_steps,
                            batch_size=profile.batch_size,
                            learning_rate=profile.learning_rate,
                            warmup_steps=profile.warmup_steps,
                            min_lr_ratio=profile.min_lr_ratio,
                            soft_fraction=profile.soft_fraction,
                            label_smoothing_start=(
                                profile.label_smoothing_start
                            ),
                            eval_interval=profile.eval_interval,
                            grad_clip=profile.grad_clip,
                            xavier_gain=profile.xavier_gain,
                            st_missing_fraction=profile.st_missing_fraction,
                            st_scale_quantile=profile.st_scale_quantile,
                            circle=profile.circle,
                        )
                    )
    return tuple(runs)


def parse_csv_values(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("comma-separated override must not be empty.")
    return values


def parse_int_values(value: str | None) -> tuple[int, ...] | None:
    raw = parse_csv_values(value)
    return None if raw is None else tuple(int(item) for item in raw)


__all__ = [
    "CircleConfig",
    "DATASETS",
    "DEFAULT_DATASETS",
    "DatasetName",
    "ExperimentProfile",
    "PROFILES",
    "REPRESENTATIONS",
    "Representation",
    "RunSpec",
    "build_run_plan",
    "get_profile",
    "override_profile",
    "parse_csv_values",
    "parse_int_values",
]
