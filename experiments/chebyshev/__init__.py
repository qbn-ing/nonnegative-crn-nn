"""Controlled Chebyshev depth experiments for rational CRN classifiers."""

from .protocol import (
    CONDITION_FACTORS,
    CONDITIONS,
    M3_TASK,
    M4_TASK,
    PROFILES,
    TASKS,
    ChebyshevTask,
    ExperimentProfile,
    RunSpec,
    build_run_plan,
    get_profile,
)

__all__ = [
    "CONDITION_FACTORS",
    "CONDITIONS",
    "M3_TASK",
    "M4_TASK",
    "PROFILES",
    "TASKS",
    "ChebyshevTask",
    "ExperimentProfile",
    "RunSpec",
    "build_run_plan",
    "get_profile",
]
