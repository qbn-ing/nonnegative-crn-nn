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
InputEncoding = Literal["raw"]

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
class ChebyshevTask:
    """One two-dimensional tensor-Chebyshev parity task."""

    name: str
    degree: int
    input_encoding: InputEncoding

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("task name must not be empty.")
        if self.degree < 1:
            raise ValueError("degree must be positive.")
        if self.input_encoding != "raw":
            raise ValueError("unsupported Chebyshev input encoding.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


M3_TASK = ChebyshevTask("m3", 3, "raw")
M4_TASK = ChebyshevTask("m4", 4, "raw")
TASKS: dict[str, ChebyshevTask] = {
    task.name: task
    for task in (M3_TASK, M4_TASK)
}


@dataclass(frozen=True)
class ExperimentProfile:
    """Resolved grid plus the shared fixed-budget optimization protocol."""

    name: str
    tasks: tuple[ChebyshevTask, ...]
    widths: tuple[int, ...]
    depths: tuple[int, ...]
    seeds: tuple[int, ...]
    conditions: tuple[Condition, ...] = CONDITIONS
    dataset_seed: int = 0
    total_steps: int = 8_000
    batch_size: int = 512
    learning_rate: float = 3e-3
    min_lr_ratio: float = 0.01
    warmup_steps: int = 400
    soft_fraction: float = 0.35
    beta_min: float = 1.0
    beta_max: float = 32.0
    eval_interval: int = 100
    train_samples: int = 16_384
    val_samples: int = 4_096
    test_samples: int = 16_384
    eval_grid_size: int = 257
    edge_tolerance_cells: int = 2
    grad_clip: float = 1.0
    xavier_gain: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("profile name must not be empty.")
        if not self.tasks or len({task.name for task in self.tasks}) != len(
            self.tasks
        ):
            raise ValueError("tasks must be non-empty and unique.")
        _positive_unique("widths", self.widths)
        _positive_unique("depths", self.depths)
        _nonnegative_unique("seeds", self.seeds)
        if self.dataset_seed < 0:
            raise ValueError("dataset_seed must be nonnegative.")
        if not self.conditions or len(set(self.conditions)) != len(
            self.conditions
        ):
            raise ValueError("conditions must be non-empty and unique.")
        if any(condition not in CONDITIONS for condition in self.conditions):
            raise ValueError("profile contains an unsupported condition.")
        for name in (
            "total_steps",
            "batch_size",
            "eval_interval",
            "train_samples",
            "val_samples",
            "test_samples",
            "eval_grid_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("warmup_steps must lie inside the step budget.")
        if not 0.0 < self.soft_fraction < 1.0:
            raise ValueError("soft_fraction must lie strictly inside (0, 1).")
        if self.soft_steps >= self.total_steps:
            raise ValueError("soft curriculum must end before training.")
        for name in (
            "learning_rate",
            "min_lr_ratio",
            "beta_min",
            "beta_max",
            "grad_clip",
            "xavier_gain",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.beta_max < self.beta_min:
            raise ValueError("beta_max must be at least beta_min.")
        if self.edge_tolerance_cells < 0:
            raise ValueError("edge_tolerance_cells must be nonnegative.")

    @property
    def soft_steps(self) -> int:
        return max(1, int(round(self.total_steps * self.soft_fraction)))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["soft_steps"] = self.soft_steps
        payload["schema_version"] = 1
        return payload


@dataclass(frozen=True)
class RunSpec:
    """One paired I-by-C cell with a fixed total optimizer budget."""

    profile_name: str
    task: ChebyshevTask
    width: int
    depth: int
    seed: int
    condition: Condition
    dataset_seed: int
    total_steps: int
    batch_size: int
    learning_rate: float
    min_lr_ratio: float
    warmup_steps: int
    soft_steps: int
    beta_min: float
    beta_max: float
    eval_interval: int
    train_samples: int
    val_samples: int
    test_samples: int
    eval_grid_size: int
    edge_tolerance_cells: int
    grad_clip: float
    xavier_gain: float
    continuation_depths: tuple[int, ...]
    stage_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError("unsupported condition.")
        if self.width <= 0 or self.depth <= 0:
            raise ValueError("width and depth must be positive.")
        if self.seed < 0 or self.dataset_seed < 0:
            raise ValueError("seeds must be nonnegative.")
        if len(self.continuation_depths) != len(self.stage_steps):
            raise ValueError("continuation depths and stage steps must align.")
        if not self.continuation_depths:
            raise ValueError("optimizer stages must not be empty.")
        if self.continuation_depths[-1] != self.depth:
            raise ValueError("continuation schedule must end at final depth.")
        if any(value <= 0 for value in self.stage_steps):
            raise ValueError("every stage step budget must be positive.")
        if sum(self.stage_steps) != self.total_steps:
            raise ValueError("stage steps must sum to total_steps.")
        if not 0 < self.soft_steps < self.total_steps:
            raise ValueError("soft_steps must lie inside total_steps.")
        if self.depth == 1 and self.condition != "dense_full_start":
            raise ValueError("D1 defines only the I0_C0 shallow control.")

    @property
    def factors(self) -> dict[str, bool]:
        return dict(CONDITION_FACTORS[self.condition])

    @property
    def factor_code(self) -> str:
        return FACTOR_CODES[self.condition]

    @property
    def uses_continuation(self) -> bool:
        return self.factors["depth_continuation"]

    @property
    def uses_identity(self) -> bool:
        return self.factors["identity_initialization"]

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
        return 100_000 + self.seed

    @property
    def run_id(self) -> str:
        base = (
            f"{self.task.name}_w{self.width}_d{self.depth}_"
            f"{self.factor_code}_{self.condition}_s{self.seed:03d}"
        )
        return signed_run_id(base, asdict(self))

    @property
    def run_signature(self) -> str:
        return scientific_signature(asdict(self))

    @property
    def paired_cell(self) -> tuple[str, int, int, int]:
        return (self.task.name, self.width, self.depth, self.seed)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": 1,
                "experiment_kind": "chebyshev",
                "run_id": self.run_id,
                "run_signature": self.run_signature,
                "factor_code": self.factor_code,
                "factors": self.factors,
                "initial_depth": self.initial_depth,
                "optimizer_stage_depths": self.optimizer_stage_depths,
                "batch_seed": self.batch_seed,
                "depth_definition": (
                    "all E/I-to-r layers including terminal class evidence; "
                    "competitive O layer excluded"
                ),
            }
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunSpec:
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported Chebyshev run schema.")
        names = {item.name for item in fields(cls)}
        missing = sorted(names - set(payload))
        if missing:
            raise ValueError("run specification is missing: " + ", ".join(missing))
        values = {name: payload[name] for name in names}
        values["task"] = ChebyshevTask(**values["task"])
        values["continuation_depths"] = tuple(values["continuation_depths"])
        values["stage_steps"] = tuple(values["stage_steps"])
        instance = cls(**values)
        validate_run_identity(instance, payload)
        return instance


def _positive_unique(name: str, values: tuple[int, ...]) -> None:
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates.")


def _nonnegative_unique(name: str, values: tuple[int, ...]) -> None:
    if not values or any(value < 0 for value in values):
        raise ValueError(f"{name} must contain nonnegative integers.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates.")


PROFILES: dict[str, ExperimentProfile] = {
    "smoke": ExperimentProfile(
        name="smoke",
        tasks=(M3_TASK, M4_TASK),
        widths=(4,),
        depths=(4,),
        seeds=(0,),
        total_steps=12,
        batch_size=32,
        learning_rate=2e-3,
        warmup_steps=2,
        soft_fraction=1.0 / 3.0,
        eval_interval=2,
        train_samples=96,
        val_samples=64,
        test_samples=64,
        eval_grid_size=17,
        edge_tolerance_cells=1,
    ),
    "pilot": ExperimentProfile(
        name="pilot",
        tasks=(M3_TASK, M4_TASK),
        widths=(16,),
        depths=(9,),
        seeds=(0,),
    ),
    "main": ExperimentProfile(
        name="main",
        tasks=(M3_TASK, M4_TASK),
        widths=(8, 16, 32),
        depths=(4, 9),
        seeds=(0, 1, 2, 3, 4),
    ),
    "d1": ExperimentProfile(
        name="d1",
        tasks=(M3_TASK, M4_TASK),
        widths=(2,),
        depths=(1,),
        seeds=(0, 1, 2, 3, 4),
        conditions=("dense_full_start",),
    ),
}


def get_profile(name: str) -> ExperimentProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown profile {name!r}; choose from {tuple(PROFILES)}.") from exc


def override_profile(
    profile: ExperimentProfile,
    *,
    task_names: Iterable[str] | None = None,
    widths: Iterable[int] | None = None,
    depths: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
    conditions: Iterable[str] | None = None,
    total_steps: int | None = None,
) -> ExperimentProfile:
    changes: dict[str, Any] = {}
    if task_names is not None:
        try:
            changes["tasks"] = tuple(TASKS[name] for name in task_names)
        except KeyError as exc:
            raise ValueError(f"unknown task {exc.args[0]!r}.") from exc
    if widths is not None:
        changes["widths"] = tuple(widths)
    if depths is not None:
        changes["depths"] = tuple(depths)
    if seeds is not None:
        changes["seeds"] = tuple(seeds)
    if conditions is not None:
        changes["conditions"] = tuple(conditions)
    if total_steps is not None:
        changes["total_steps"] = total_steps
        changes["warmup_steps"] = min(
            profile.warmup_steps,
            max(0, total_steps // 20),
        )
    return replace(profile, **changes)


def build_run_plan(profile: ExperimentProfile) -> tuple[RunSpec, ...]:
    runs: list[RunSpec] = []
    for task in profile.tasks:
        for width in profile.widths:
            for depth in profile.depths:
                ladder = continuation_ladder(depth)
                budgets = continuation_step_budgets(
                    total_steps=profile.total_steps,
                    soft_steps=profile.soft_steps,
                    n_stages=len(ladder),
                )
                for seed in profile.seeds:
                    for condition in profile.conditions:
                        runs.append(
                            RunSpec(
                                profile_name=profile.name,
                                task=task,
                                width=width,
                                depth=depth,
                                seed=seed,
                                condition=condition,
                                dataset_seed=profile.dataset_seed,
                                total_steps=profile.total_steps,
                                batch_size=profile.batch_size,
                                learning_rate=profile.learning_rate,
                                min_lr_ratio=profile.min_lr_ratio,
                                warmup_steps=profile.warmup_steps,
                                soft_steps=profile.soft_steps,
                                beta_min=profile.beta_min,
                                beta_max=profile.beta_max,
                                eval_interval=profile.eval_interval,
                                train_samples=profile.train_samples,
                                val_samples=profile.val_samples,
                                test_samples=profile.test_samples,
                                eval_grid_size=profile.eval_grid_size,
                                edge_tolerance_cells=profile.edge_tolerance_cells,
                                grad_clip=profile.grad_clip,
                                xavier_gain=profile.xavier_gain,
                                continuation_depths=ladder,
                                stage_steps=budgets,
                            )
                        )
    return tuple(runs)


def continuation_ladder(depth: int) -> tuple[int, ...]:
    """D2 anchor, one midpoint, and final depth under the total-depth rule."""

    return resolve_continuation_ladder(
        depth,
        anchors=(2,),
        midpoint=True,
        single_depth_repeats=3,
    )


def continuation_step_budgets(
    *,
    total_steps: int,
    soft_steps: int,
    n_stages: int,
) -> tuple[int, ...]:
    """Open the final depth exactly at the soft-to-hard boundary."""

    return _shared_step_budgets(
        total_steps=total_steps,
        soft_steps=soft_steps,
        n_stages=n_stages,
    )


__all__ = [
    "CONDITION_FACTORS",
    "CONDITIONS",
    "FACTOR_CODES",
    "M3_TASK",
    "M4_TASK",
    "PROFILES",
    "TASKS",
    "ChebyshevTask",
    "Condition",
    "ExperimentProfile",
    "InputEncoding",
    "RunSpec",
    "build_run_plan",
    "continuation_ladder",
    "continuation_step_budgets",
    "get_profile",
    "override_profile",
]
