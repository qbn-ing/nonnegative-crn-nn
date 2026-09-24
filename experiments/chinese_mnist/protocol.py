from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Iterable, Literal

from experiments.common.signatures import (
    scientific_signature,
    signed_run_id,
    validate_run_identity,
)
from training.schedule import (
    continuation_step_budgets as _shared_step_budgets,
    resolve_continuation_ladder,
)


Condition = Literal[
    "dense_full_start",
    "identity_full_start",
    "dense_continuation",
    "continuation",
]

CONDITIONS: tuple[Condition, ...] = (
    "dense_full_start",
    "identity_full_start",
    "dense_continuation",
    "continuation",
)
CONDITION_FACTORS: dict[Condition, dict[str, bool]] = {
    "dense_full_start": {
        "identity_initialization": False,
        "depth_continuation": False,
    },
    "identity_full_start": {
        "identity_initialization": True,
        "depth_continuation": False,
    },
    "dense_continuation": {
        "identity_initialization": False,
        "depth_continuation": True,
    },
    "continuation": {
        "identity_initialization": True,
        "depth_continuation": True,
    },
}
FACTOR_CODES: dict[Condition, str] = {
    condition: (
        f"I{int(factors['identity_initialization'])}_"
        f"C{int(factors['depth_continuation'])}"
    )
    for condition, factors in CONDITION_FACTORS.items()
}


@dataclass(frozen=True)
class ArchitectureCondition:
    """One final topology and one I-by-C training cell."""

    name: str
    width: int
    depth: int
    condition: Condition
    roles: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("architecture condition name must not be empty.")
        if self.width <= 0 or self.depth <= 0:
            raise ValueError("width and depth must be positive.")
        if self.condition not in CONDITIONS:
            raise ValueError("unsupported I-by-C condition.")
        if not self.roles or any(not role for role in self.roles):
            raise ValueError("roles must be non-empty strings.")
        if self.depth == 1 and self.condition != "dense_full_start":
            raise ValueError("D1 defines only the I0_C0 shallow control.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEPTH_SCALING_CONDITIONS: tuple[ArchitectureCondition, ...] = (
    ArchitectureCondition(
        "d1_k15_i0_c0", 15, 1, "dense_full_start", ("depth_scaling",)
    ),
    ArchitectureCondition(
        "d2_w24_i0_c0", 24, 2, "dense_full_start", ("depth_scaling",)
    ),
    ArchitectureCondition(
        "d3_w24_i1_c0", 24, 3, "identity_full_start", ("depth_scaling",)
    ),
    ArchitectureCondition(
        "d5_w24_i1_c0",
        24,
        5,
        "identity_full_start",
        ("depth_scaling", "optimization_rescue"),
    ),
)


def _rescue_conditions() -> tuple[ArchitectureCondition, ...]:
    output: list[ArchitectureCondition] = []
    for depth in (5, 7, 10):
        for condition in CONDITIONS:
            if depth == 5 and condition == "identity_full_start":
                output.append(DEPTH_SCALING_CONDITIONS[-1])
                continue
            output.append(
                ArchitectureCondition(
                    name=f"d{depth}_w24_{FACTOR_CODES[condition].lower()}",
                    width=24,
                    depth=depth,
                    condition=condition,
                    roles=("optimization_rescue",),
                )
            )
    return tuple(output)


OPTIMIZATION_RESCUE_CONDITIONS = _rescue_conditions()
FULL_CONDITIONS = tuple(
    dict.fromkeys(DEPTH_SCALING_CONDITIONS + OPTIMIZATION_RESCUE_CONDITIONS)
)


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    architectures: tuple[ArchitectureCondition, ...]
    folds: tuple[int, ...]
    base_model_seed: int = 0
    split_seed: int = 9_483_557
    n_splits: int = 5
    validation_writers: int = 16
    image_size: int = 28
    num_classes: int = 15
    batch_size: int = 512
    nominal_epochs: int | None = 240
    total_steps: int = 4_560
    learning_rate: float = 3e-3
    warmup_steps: int = 57
    min_lr_ratio: float = 0.05
    soft_fraction: float = 0.35
    label_smoothing_start: float = 0.10
    eval_interval: int = 95
    grad_clip: float = 5.0
    xavier_gain: float = 1.0
    bootstrap_replicates: int = 20_000
    bootstrap_seed: int = 20_260_731

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("profile name must not be empty.")
        if not self.architectures or len(set(self.architectures)) != len(
            self.architectures
        ):
            raise ValueError("architectures must be non-empty and unique.")
        if not self.folds or len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be non-empty and unique.")
        if any(not 0 <= fold < self.n_splits for fold in self.folds):
            raise ValueError("fold index is outside the configured CV range.")
        if self.base_model_seed < 0 or self.split_seed < 0:
            raise ValueError("seeds must be nonnegative.")
        for name in (
            "n_splits",
            "validation_writers",
            "image_size",
            "num_classes",
            "batch_size",
            "total_steps",
            "eval_interval",
            "bootstrap_replicates",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.nominal_epochs is not None and self.nominal_epochs <= 0:
            raise ValueError("nominal_epochs must be positive or None.")
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("warmup_steps must lie inside total_steps.")
        if not 0.0 < self.soft_fraction < 1.0:
            raise ValueError("soft_fraction must lie inside (0, 1).")
        if self.soft_steps >= self.total_steps:
            raise ValueError("soft curriculum must end before training.")
        if not 0.0 <= self.label_smoothing_start < 1.0:
            raise ValueError("label smoothing must lie in [0, 1).")
        for name in (
            "learning_rate",
            "min_lr_ratio",
            "grad_clip",
            "xavier_gain",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive.")

    @property
    def soft_steps(self) -> int:
        return max(1, int(round(self.total_steps * self.soft_fraction)))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": 1,
                "soft_steps": self.soft_steps,
                "depth_definition": (
                    "D counts every E/I-to-r layer including the terminal "
                    "K=15 class-evidence layer; O is excluded"
                ),
            }
        )
        return payload


@dataclass(frozen=True)
class RunSpec:
    profile_name: str
    architecture_name: str
    roles: tuple[str, ...]
    width: int
    depth: int
    condition: Condition
    fold: int
    model_seed: int
    split_seed: int
    n_splits: int
    validation_writers: int
    image_size: int
    num_classes: int
    batch_size: int
    nominal_epochs: int | None
    total_steps: int
    learning_rate: float
    warmup_steps: int
    min_lr_ratio: float
    soft_fraction: float
    label_smoothing_start: float
    eval_interval: int
    grad_clip: float
    xavier_gain: float
    bootstrap_replicates: int
    bootstrap_seed: int

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError("unsupported condition.")
        if self.width <= 0 or self.depth <= 0:
            raise ValueError("width and depth must be positive.")
        if self.depth >= 2 and self.width < self.num_classes:
            raise ValueError("hidden width must be at least num_classes.")
        if not 0 <= self.fold < self.n_splits:
            raise ValueError("fold is outside the configured CV range.")
        if self.model_seed < 0 or self.split_seed < 0:
            raise ValueError("seeds must be nonnegative.")
        if self.total_steps <= 1:
            raise ValueError("total_steps must be greater than one.")
        if self.depth == 1 and self.condition != "dense_full_start":
            raise ValueError("D1 defines only the I0_C0 shallow control.")

    @property
    def factors(self) -> dict[str, bool]:
        return dict(CONDITION_FACTORS[self.condition])

    @property
    def factor_code(self) -> str:
        return FACTOR_CODES[self.condition]

    @property
    def uses_identity(self) -> bool:
        return self.factors["identity_initialization"]

    @property
    def uses_continuation(self) -> bool:
        return self.factors["depth_continuation"]

    @property
    def soft_steps(self) -> int:
        return max(1, int(round(self.total_steps * self.soft_fraction)))

    @property
    def continuation_depths(self) -> tuple[int, ...]:
        return continuation_ladder(self.depth)

    @property
    def stage_steps(self) -> tuple[int, ...]:
        return continuation_step_budgets(
            total_steps=self.total_steps,
            soft_steps=self.soft_steps,
            ladder=self.continuation_depths,
        )

    @property
    def initial_depth(self) -> int:
        return self.continuation_depths[0] if self.uses_continuation else self.depth

    @property
    def optimizer_stage_depths(self) -> tuple[int, ...]:
        if self.uses_continuation:
            return self.continuation_depths
        return (self.depth,) * len(self.stage_steps)

    @property
    def batch_seed(self) -> int:
        return 100_000 + self.model_seed

    @property
    def run_id(self) -> str:
        base = f"{self.architecture_name}_f{self.fold:02d}s{self.model_seed:03d}"
        return signed_run_id(base, asdict(self))

    @property
    def run_signature(self) -> str:
        return scientific_signature(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": 1,
                "experiment_kind": "chinese_mnist",
                "run_id": self.run_id,
                "run_signature": self.run_signature,
                "factor_code": self.factor_code,
                "factors": self.factors,
                "soft_steps": self.soft_steps,
                "continuation_depths": self.continuation_depths,
                "stage_steps": self.stage_steps,
                "initial_depth": self.initial_depth,
                "optimizer_stage_depths": self.optimizer_stage_depths,
                "batch_seed": self.batch_seed,
                "depth_definition": (
                    "all E/I-to-r layers including terminal K=15 class "
                    "evidence; competitive O excluded"
                ),
            }
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunSpec:
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported Chinese-MNIST run schema.")
        names = {item.name for item in fields(cls)}
        missing = sorted(names - set(payload))
        if missing:
            raise ValueError("run specification is missing: " + ", ".join(missing))
        values = {name: payload[name] for name in names}
        values["roles"] = tuple(values["roles"])
        instance = cls(**values)
        validate_run_identity(instance, payload)
        return instance


def continuation_ladder(depth: int) -> tuple[int, ...]:
    """Historical D2/D3/D5 anchors under the corrected total-depth rule."""

    return resolve_continuation_ladder(
        depth,
        anchors=(2, 3, 5),
    )


def continuation_step_budgets(
    *,
    total_steps: int,
    soft_steps: int,
    ladder: tuple[int, ...],
) -> tuple[int, ...]:
    """Reach the final depth exactly at the soft-to-hard boundary."""

    prefinal_count = max(0, len(ladder) - 1)
    historical_weights = {
        1: (1,),
        2: (42, 48),
        3: (42, 48, 60),
    }.get(prefinal_count, tuple(range(1, prefinal_count + 1)))
    return _shared_step_budgets(
        total_steps=total_steps,
        soft_steps=soft_steps,
        ladder=ladder,
        prefinal_weights=historical_weights,
    )


def _profile(
    name: str,
    architectures: tuple[ArchitectureCondition, ...],
    folds: tuple[int, ...],
    **changes: Any,
) -> ExperimentProfile:
    return ExperimentProfile(
        name=name,
        architectures=architectures,
        folds=folds,
        **changes,
    )


SMOKE_ARCHITECTURES = (
    DEPTH_SCALING_CONDITIONS[0],
    DEPTH_SCALING_CONDITIONS[1],
    next(item for item in OPTIMIZATION_RESCUE_CONDITIONS if item.depth == 5 and item.condition == "dense_full_start"),
    DEPTH_SCALING_CONDITIONS[-1],
    next(item for item in OPTIMIZATION_RESCUE_CONDITIONS if item.depth == 5 and item.condition == "dense_continuation"),
    next(item for item in OPTIMIZATION_RESCUE_CONDITIONS if item.depth == 5 and item.condition == "continuation"),
)

PROFILES: dict[str, ExperimentProfile] = {
    "smoke": _profile(
        "smoke",
        SMOKE_ARCHITECTURES,
        (0,),
        nominal_epochs=None,
        total_steps=12,
        batch_size=128,
        warmup_steps=2,
        eval_interval=4,
        bootstrap_replicates=200,
    ),
    "pilot": _profile("pilot", FULL_CONDITIONS, (0,), bootstrap_replicates=2_000),
    "depth_scaling": _profile(
        "depth_scaling", DEPTH_SCALING_CONDITIONS, (0, 1, 2, 3, 4)
    ),
    "optimization_rescue": _profile(
        "optimization_rescue",
        OPTIMIZATION_RESCUE_CONDITIONS,
        (0, 1, 2, 3, 4),
    ),
    "full": _profile("full", FULL_CONDITIONS, (0, 1, 2, 3, 4)),
}


def get_profile(name: str) -> ExperimentProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown profile {name!r}; choose from {tuple(PROFILES)}.") from exc


def override_profile(
    profile: ExperimentProfile,
    *,
    folds: Iterable[int] | None = None,
    depths: Iterable[int] | None = None,
    conditions: Iterable[str] | None = None,
    total_steps: int | None = None,
) -> ExperimentProfile:
    changes: dict[str, Any] = {}
    if folds is not None:
        changes["folds"] = tuple(folds)
    architectures = profile.architectures
    if depths is not None:
        selected_depths = set(depths)
        architectures = tuple(item for item in architectures if item.depth in selected_depths)
    if conditions is not None:
        selected_conditions = set(conditions)
        unknown = selected_conditions - set(CONDITIONS)
        if unknown:
            raise ValueError(f"unsupported conditions: {sorted(unknown)}")
        architectures = tuple(item for item in architectures if item.condition in selected_conditions)
    if architectures != profile.architectures:
        changes["architectures"] = architectures
    if total_steps is not None:
        changes["total_steps"] = total_steps
        changes["nominal_epochs"] = None
        changes["warmup_steps"] = min(profile.warmup_steps, max(0, total_steps // 20))
        changes["eval_interval"] = min(profile.eval_interval, max(1, total_steps // 4))
    return replace(profile, **changes)


def build_run_plan(profile: ExperimentProfile) -> tuple[RunSpec, ...]:
    runs: list[RunSpec] = []
    for architecture in profile.architectures:
        for fold in profile.folds:
            runs.append(
                RunSpec(
                    profile_name=profile.name,
                    architecture_name=architecture.name,
                    roles=architecture.roles,
                    width=architecture.width,
                    depth=architecture.depth,
                    condition=architecture.condition,
                    fold=fold,
                    model_seed=profile.base_model_seed + fold,
                    split_seed=profile.split_seed,
                    n_splits=profile.n_splits,
                    validation_writers=profile.validation_writers,
                    image_size=profile.image_size,
                    num_classes=profile.num_classes,
                    batch_size=profile.batch_size,
                    nominal_epochs=profile.nominal_epochs,
                    total_steps=profile.total_steps,
                    learning_rate=profile.learning_rate,
                    warmup_steps=profile.warmup_steps,
                    min_lr_ratio=profile.min_lr_ratio,
                    soft_fraction=profile.soft_fraction,
                    label_smoothing_start=profile.label_smoothing_start,
                    eval_interval=profile.eval_interval,
                    grad_clip=profile.grad_clip,
                    xavier_gain=profile.xavier_gain,
                    bootstrap_replicates=profile.bootstrap_replicates,
                    bootstrap_seed=profile.bootstrap_seed,
                )
            )
    return tuple(runs)


__all__ = [
    "CONDITIONS",
    "CONDITION_FACTORS",
    "DEPTH_SCALING_CONDITIONS",
    "FACTOR_CODES",
    "FULL_CONDITIONS",
    "OPTIMIZATION_RESCUE_CONDITIONS",
    "PROFILES",
    "ArchitectureCondition",
    "Condition",
    "ExperimentProfile",
    "RunSpec",
    "build_run_plan",
    "continuation_ladder",
    "continuation_step_budgets",
    "get_profile",
    "override_profile",
]
