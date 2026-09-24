from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from export.crn_instance import (
    CRNInstance,
    CRNInstanceError,
    CRNInstanceOptions,
    build_crn_instance,
)

class PySBUnavailableError(ImportError):
    """Raised when PySB is required but is not installed."""


class PySBValidationError(RuntimeError):
    """Raised when PySB model construction or simulation fails."""

@dataclass(frozen=True)
class PySBValidationConfig:
    """Configuration for PySB-based CRN validation.

    The validation treats ``Z``, ``Z_free``, and active-pool-normalized class
    scores as distinct quantities.  It can compare PySB and digital-model raw
    concentrations, check output-pool conservation, and compare normalized
    class scores without interpreting ``Z_free`` as a class.
    """

    t_end: float = 20.0
    n_timepoints: int = 201
    atol: float = 1e-4
    rtol: float = 1e-4
    eps: float = 1e-12
    integrator: str = "vode"
    cleanup: bool = True
    verbose: bool = False
    integrator_options: Mapping[str, Any] = field(default_factory=dict)
    tail_window: int = 5
    tail_atol: float = 1e-6
    tail_rtol: float = 1e-6


    def tspan(self) -> np.ndarray:
        if not math.isfinite(self.t_end) or self.t_end <= 0.0:
            raise ValueError("t_end must be a positive finite number.")
        if self.n_timepoints < 2:
            raise ValueError("n_timepoints must be at least 2.")
        return np.linspace(0.0, float(self.t_end), int(self.n_timepoints), dtype=float)
    
    def validate_tail_config(self) -> None:
        if self.tail_window < 2:
            raise ValueError(f"tail_window must be at least 2. Got {self.tail_window}.")
        if self.tail_atol < 0.0:
            raise ValueError(f"tail_atol must be nonnegative. Got {self.tail_atol}.")
        if self.tail_rtol < 0.0:
            raise ValueError(f"tail_rtol must be nonnegative. Got {self.tail_rtol}.")
    
@dataclass(frozen=True)
class PySBModelBundle:
    """A PySB model plus observables for the complete terminal output pool."""

    model: Any
    output_species: tuple[str, ...]
    output_observables: Mapping[str, str]
    inactive_output_species: str | None = None
    inactive_output_observable: str | None = None
    response_species: tuple[str, ...] = ()
    response_observables: Mapping[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class PySBValidationResult:
    """Validation result for one input sample."""

    sample_index: int | None
    target: int | None
    reference_pred: int | None
    pysb_pred: int
    reference_probabilities: tuple[float, ...] | None
    pysb_probabilities: tuple[float, ...]
    pysb_concentrations: tuple[float, ...]
    max_abs_error: float | None
    max_rel_error: float | None
    class_match: bool | None
    target_match: bool | None
    within_tolerance: bool | None
    output_tail: PySBOutputTailDiagnostic | None = None
    reference_r: tuple[float, ...] | None = None
    reference_Z: tuple[float, ...] | None = None
    reference_Z_free: float | None = None
    pysb_Z_free: float | None = None
    reference_output_pool_total: float | None = None
    pysb_output_pool_total: float | None = None
    output_pool_conservation_error: float | None = None
    reference_active_output_fraction: float | None = None
    pysb_active_output_fraction: float | None = None
    max_abs_Z_error: float | None = None
    max_rel_Z_error: float | None = None
    abs_Z_free_error: float | None = None
    rel_Z_free_error: float | None = None
    raw_concentrations_within_tolerance: bool | None = None
    normalized_scores_within_tolerance: bool | None = None
    output_pool_conserved: bool | None = None
    pysb_r: tuple[float, ...] | None = None
    max_abs_r_error: float | None = None
    max_rel_r_error: float | None = None
    response_within_tolerance: bool | None = None
    response_tail: PySBOutputTailDiagnostic | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "target": self.target,
            "reference_pred": self.reference_pred,
            "pysb_pred": self.pysb_pred,
            "reference_r": (
                list(self.reference_r)
                if self.reference_r is not None
                else None
            ),
            "pysb_r": (
                None if self.pysb_r is None else list(self.pysb_r)
            ),
            "max_abs_r_error": self.max_abs_r_error,
            "max_rel_r_error": self.max_rel_r_error,
            "response_within_tolerance": self.response_within_tolerance,
            "response_tail": (
                None
                if self.response_tail is None
                else self.response_tail.to_dict()
            ),
            "reference_Z": (
                list(self.reference_Z)
                if self.reference_Z is not None
                else None
            ),
            "reference_Z_free": self.reference_Z_free,
            "reference_output_pool_total": self.reference_output_pool_total,
            "reference_active_output_fraction": (
                self.reference_active_output_fraction
            ),
            "reference_normalized_class_scores": (
                list(self.reference_probabilities)
                if self.reference_probabilities is not None
                else None
            ),
            "pysb_Z": list(self.pysb_concentrations),
            "pysb_Z_free": self.pysb_Z_free,
            "pysb_output_pool_total": self.pysb_output_pool_total,
            "pysb_active_output_fraction": self.pysb_active_output_fraction,
            "pysb_normalized_class_scores": list(self.pysb_probabilities),
            "output_pool_conservation_error": (
                self.output_pool_conservation_error
            ),
            "max_abs_Z_error": self.max_abs_Z_error,
            "max_rel_Z_error": self.max_rel_Z_error,
            "abs_Z_free_error": self.abs_Z_free_error,
            "rel_Z_free_error": self.rel_Z_free_error,
            "max_abs_normalized_score_error": self.max_abs_error,
            "max_rel_normalized_score_error": self.max_rel_error,
            "raw_concentrations_within_tolerance": (
                self.raw_concentrations_within_tolerance
            ),
            "normalized_scores_within_tolerance": (
                self.normalized_scores_within_tolerance
            ),
            "output_pool_conserved": self.output_pool_conserved,
            "overall_within_tolerance": self.within_tolerance,
            "class_match": self.class_match,
            "target_match": self.target_match,
            "output_pool_tail": (
                None if self.output_tail is None else self.output_tail.to_dict()
            ),
            "legacy_normalized_score_aliases": {
                "reference_probabilities": (
                    list(self.reference_probabilities)
                    if self.reference_probabilities is not None
                    else None
                ),
                "pysb_probabilities": list(self.pysb_probabilities),
                "pysb_concentrations": list(self.pysb_concentrations),
                "max_abs_error": self.max_abs_error,
                "max_rel_error": self.max_rel_error,
                "within_tolerance": self.within_tolerance,
                "output_tail": (
                    None
                    if self.output_tail is None
                    else self.output_tail.to_dict()
                ),
            },
            "reference_probabilities": list(self.reference_probabilities)
            if self.reference_probabilities is not None
            else None,
            "pysb_probabilities": list(self.pysb_probabilities),
            "pysb_concentrations": list(self.pysb_concentrations),
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "class_match": self.class_match,
            "target_match": self.target_match,
            "within_tolerance": self.within_tolerance,
            "output_tail": (
                None if self.output_tail is None else self.output_tail.to_dict()
            ),
        }
    
@dataclass(frozen=True)
class PySBValidationSummary:
    """Aggregate statistics for a PySB validation run."""

    n_samples: int
    n_with_reference: int
    n_class_matches: int
    n_target_matches: int
    n_within_tolerance: int
    class_match_rate: float | None
    target_match_rate: float | None
    tolerance_pass_rate: float | None
    max_abs_error: float | None
    max_rel_error: float | None

    n_output_tail_checked: int = 0
    n_output_tail_steady_state_like: int = 0
    output_tail_steady_state_rate: float | None = None
    max_output_tail_abs_delta: float | None = None
    max_output_tail_rel_delta: float | None = None
    mean_output_tail_abs_delta: float | None = None
    mean_output_tail_rel_delta: float | None = None
    n_with_raw_reference: int = 0
    n_raw_concentrations_within_tolerance: int = 0
    raw_concentration_tolerance_pass_rate: float | None = None
    n_normalized_scores_within_tolerance: int = 0
    normalized_score_tolerance_pass_rate: float | None = None
    n_output_pool_conservation_checked: int = 0
    n_output_pool_conserved: int = 0
    output_pool_conservation_pass_rate: float | None = None
    max_abs_Z_error: float | None = None
    max_rel_Z_error: float | None = None
    max_abs_Z_free_error: float | None = None
    max_rel_Z_free_error: float | None = None
    max_output_pool_conservation_error: float | None = None
    n_with_response_reference: int = 0
    n_responses_within_tolerance: int = 0
    response_tolerance_pass_rate: float | None = None
    max_abs_r_error: float | None = None
    max_rel_r_error: float | None = None
    n_response_tail_checked: int = 0
    n_response_tail_steady_state_like: int = 0
    response_tail_steady_state_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_with_reference": self.n_with_reference,
            "n_class_matches": self.n_class_matches,
            "n_target_matches": self.n_target_matches,
            "n_within_tolerance": self.n_within_tolerance,
            "class_match_rate": self.class_match_rate,
            "target_match_rate": self.target_match_rate,
            "tolerance_pass_rate": self.tolerance_pass_rate,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "n_output_tail_checked": self.n_output_tail_checked,
            "n_output_tail_steady_state_like": self.n_output_tail_steady_state_like,
            "output_tail_steady_state_rate": self.output_tail_steady_state_rate,
            "max_output_tail_abs_delta": self.max_output_tail_abs_delta,
            "max_output_tail_rel_delta": self.max_output_tail_rel_delta,
            "mean_output_tail_abs_delta": self.mean_output_tail_abs_delta,
            "mean_output_tail_rel_delta": self.mean_output_tail_rel_delta,
            "n_with_raw_reference": self.n_with_raw_reference,
            "n_raw_concentrations_within_tolerance": (
                self.n_raw_concentrations_within_tolerance
            ),
            "raw_concentration_tolerance_pass_rate": (
                self.raw_concentration_tolerance_pass_rate
            ),
            "n_normalized_scores_within_tolerance": (
                self.n_normalized_scores_within_tolerance
            ),
            "normalized_score_tolerance_pass_rate": (
                self.normalized_score_tolerance_pass_rate
            ),
            "n_output_pool_conservation_checked": (
                self.n_output_pool_conservation_checked
            ),
            "n_output_pool_conserved": self.n_output_pool_conserved,
            "output_pool_conservation_pass_rate": (
                self.output_pool_conservation_pass_rate
            ),
            "max_abs_Z_error": self.max_abs_Z_error,
            "max_rel_Z_error": self.max_rel_Z_error,
            "max_abs_Z_free_error": self.max_abs_Z_free_error,
            "max_rel_Z_free_error": self.max_rel_Z_free_error,
            "max_output_pool_conservation_error": (
                self.max_output_pool_conservation_error
            ),
            "n_with_response_reference": self.n_with_response_reference,
            "n_responses_within_tolerance": (
                self.n_responses_within_tolerance
            ),
            "response_tolerance_pass_rate": (
                self.response_tolerance_pass_rate
            ),
            "max_abs_r_error": self.max_abs_r_error,
            "max_rel_r_error": self.max_rel_r_error,
            "n_response_tail_checked": self.n_response_tail_checked,
            "n_response_tail_steady_state_like": (
                self.n_response_tail_steady_state_like
            ),
            "response_tail_steady_state_rate": (
                self.response_tail_steady_state_rate
            ),
        }

@dataclass(frozen=True)
class PySBBatchValidationResult:
    """Validation result for many input samples."""

    results: tuple[PySBValidationResult, ...]
    summary: PySBValidationSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "results": [result.to_dict() for result in self.results],
        }

@dataclass(frozen=True)
class PySBSimulationResult:
    """Final PySB state and tail diagnostic for the terminal output pool."""

    final_concentrations: tuple[float, ...]
    output_tail: PySBOutputTailDiagnostic
    final_Z_free: float | None = None
    initial_output_pool_total: float | None = None
    final_output_pool_total: float | None = None
    active_output_fraction: float | None = None
    output_pool_conservation_error: float | None = None
    final_response_concentrations: tuple[float, ...] = ()
    response_tail: PySBOutputTailDiagnostic | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_concentrations": list(self.final_concentrations),
            "final_Z_free": self.final_Z_free,
            "initial_output_pool_total": self.initial_output_pool_total,
            "final_output_pool_total": self.final_output_pool_total,
            "active_output_fraction": self.active_output_fraction,
            "output_pool_conservation_error": (
                self.output_pool_conservation_error
            ),
            "final_response_concentrations": list(
                self.final_response_concentrations
            ),
            "response_tail": (
                None
                if self.response_tail is None
                else self.response_tail.to_dict()
            ),
            "output_pool_tail": self.output_tail.to_dict(),
            "output_tail": self.output_tail.to_dict(),
        }

@dataclass(frozen=True)
class PySBOutputTailDiagnostic:
    """Tail-convergence diagnostic for the complete terminal output pool."""

    window: int
    max_abs_delta: float
    max_rel_delta: float
    steady_state_like: bool
    species: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "max_abs_delta": self.max_abs_delta,
            "max_rel_delta": self.max_rel_delta,
            "steady_state_like": self.steady_state_like,
            "species": list(self.species),
        }


def diagnose_output_tail_convergence(
    output_trajectory: np.ndarray,
    *,
    window: int = 5,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    eps: float = 1e-12,
    species: Sequence[str] | None = None,
) -> PySBOutputTailDiagnostic:
    """Check whether output concentrations are nearly stable at trajectory tail.

    Parameters
    ----------
    output_trajectory:
        Array with shape ``(n_timepoints, n_outputs)``.

    window:
        Number of final timepoints used for the tail comparison.

    atol, rtol:
        Absolute and relative tolerances for checking tail stability.

    eps:
        Numerical floor for relative-error normalization.

    species:
        Species represented by the trajectory columns.  Terminal competitive
        output validation supplies active ``Z_k`` species followed by
        ``Z_free``.
    """

    trajectory = np.asarray(output_trajectory, dtype=float)

    if trajectory.ndim != 2:
        raise ValueError(
            "output_trajectory must be a 2D array with shape "
            "(n_timepoints, n_outputs). "
            f"Got shape {trajectory.shape}."
        )

    if trajectory.shape[0] < 2:
        raise ValueError(
            "output_trajectory must contain at least two timepoints. "
            f"Got {trajectory.shape[0]}."
        )

    if trajectory.shape[1] < 1:
        raise ValueError(
            "output_trajectory must contain at least one output species."
        )

    if not np.all(np.isfinite(trajectory)):
        raise ValueError("output_trajectory must contain finite values.")

    if window < 2:
        raise ValueError(f"window must be at least 2. Got {window}.")

    if atol < 0.0:
        raise ValueError(f"atol must be nonnegative. Got {atol}.")

    if rtol < 0.0:
        raise ValueError(f"rtol must be nonnegative. Got {rtol}.")

    if eps <= 0.0:
        raise ValueError(f"eps must be positive. Got {eps}.")

    species_tuple = tuple(str(value) for value in (species or ()))
    if species_tuple and len(species_tuple) != trajectory.shape[1]:
        raise ValueError(
            "species must contain one name per output_trajectory column."
        )

    effective_window = min(int(window), int(trajectory.shape[0]))
    tail = trajectory[-effective_window:]
    final = tail[-1]

    abs_delta = np.abs(tail - final)
    max_abs_delta = float(np.max(abs_delta))

    scale = np.maximum(np.abs(final), float(eps))
    rel_delta = abs_delta / scale.reshape(1, -1)
    max_rel_delta = float(np.max(rel_delta))

    steady_state_like = bool(
        max_abs_delta <= float(atol) + float(rtol) * float(np.max(np.abs(final)))
    )

    return PySBOutputTailDiagnostic(
        window=effective_window,
        max_abs_delta=max_abs_delta,
        max_rel_delta=max_rel_delta,
        steady_state_like=steady_state_like,
        species=species_tuple,
    )

def normalized_class_scores_from_concentrations(
    values: Sequence[float],
    *,
    eps: float = 1e-12,
) -> tuple[float, ...]:
    """Normalize active-class concentrations while excluding ``Z_free``."""

    array = _nonnegative_concentration_array(values, eps=eps)
    denominator = max(float(array.sum()), float(eps))
    return tuple(float(value / denominator) for value in array)


def concentration_probabilities(
    values: Sequence[float],
    *,
    eps: float = 1e-12,
) -> tuple[float, ...]:
    """Compatibility normalization with additive epsilon smoothing."""

    array = _nonnegative_concentration_array(values, eps=eps)
    denominator = float(array.sum()) + float(eps) * float(array.size)
    if denominator <= 0.0:
        return tuple(float(1.0 / array.size) for _ in range(array.size))
    return tuple(float((value + float(eps)) / denominator) for value in array)


def _nonnegative_concentration_array(
    values: Sequence[float],
    *,
    eps: float,
) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError("values must contain at least one concentration.")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite.")
    if np.any(array < 0.0):
        if np.min(array) < -float(eps):
            raise ValueError("values must be non-negative concentrations.")
        array = np.maximum(array, 0.0)
    return array

def _reactant_symmetry_factor(reactants: Sequence[str]) -> float:
    """Return the correction factor needed for repeated PySB reactant patterns.

    The CRN convention used by this project interprets a reaction such as

        X + X -> X + X + Phi

    as having deterministic mass-action rate k * X^2.

    PySB/BioNetGen rules with identical reactant patterns may introduce
    symmetry handling for indistinguishable reactant matches. To preserve the
    project's CRN rate-law convention, we multiply the PySB rule parameter by
    prod_s factorial(m_s), where m_s is the multiplicity of species s on the
    reactant side.
    """

    factor = 1
    for count in Counter(reactants).values():
        if count > 1:
            factor *= math.factorial(int(count))
    return float(factor)


def _reaction_base_rate_value(instance: CRNInstance, reaction: Any) -> float:
    """Return the numeric base rate value for one CRN reaction."""

    if reaction.rate_key in instance.parameters:
        return float(instance.parameters[reaction.rate_key])
    return float(reaction.rate_value)


def _inactive_output_species(instance: CRNInstance) -> str | None:
    """Resolve the inactive species of the terminal conserved output pool."""

    explicit = getattr(instance, "inactive_output_species", None)
    if explicit is None and isinstance(instance.metadata, Mapping):
        metadata_value = instance.metadata.get("inactive_output_species")
        if metadata_value is not None:
            explicit = str(metadata_value)
    if explicit is not None:
        if explicit not in instance.species:
            raise CRNInstanceError(
                "inactive_output_species is not present in the CRN instance."
            )
        return explicit

    active = set(instance.final_output_species)
    candidates = [
        species
        for species in instance.species
        if species not in active and species.endswith("Z_free")
    ]
    if len(candidates) > 1:
        raise CRNInstanceError(
            "The inactive output-pool species is ambiguous."
        )
    return candidates[0] if candidates else None


def build_pysb_model(instance: CRNInstance) -> PySBModelBundle:
    """Build an in-memory PySB model from one concrete CRN instance."""

    _validate_instance(instance)

    try:
        from pysb import Initial, Model, Monomer, Observable, Parameter, Rule
    except ImportError as exc:  # pragma: no cover
        raise PySBUnavailableError(
            "PySB is required for PySB validation. Install pysb in the runtime environment."
        ) from exc

    used_names: set[str] = set()
    model_name = _unique_component_name(instance.model_name or "crn_model", used_names)
    model = Model(model_name, _export=False)

    # 创建单体物质
    monomers: dict[str, Any] = {}
    for species_name in instance.species:
        component_name = _unique_component_name(species_name, used_names)
        monomer = Monomer(component_name, _export=False)
        model.add_component(monomer)
        monomers[species_name] = monomer
    
    
    rate_parameters: dict[str, Any] = {}
    for rate_key, value in instance.parameters.items():
        parameter_name = _unique_component_name(f"k_{rate_key}", used_names)
        parameter = Parameter(parameter_name, float(value), _export=False)
        model.add_component(parameter)
        rate_parameters[rate_key] = parameter
    # 创建速率参数
    for reaction in instance.reactions:
        if reaction.rate_key not in rate_parameters:
            parameter_name = _unique_component_name(f"k_{reaction.rate_key}", used_names)
            parameter = Parameter(parameter_name, float(reaction.rate_value), _export=False)
            model.add_component(parameter)
            rate_parameters[reaction.rate_key] = parameter

    # 设置初始浓度
    for species_name in instance.species:
        parameter_name = _unique_component_name(f"init_{species_name}", used_names)
        parameter = Parameter(parameter_name, instance.initial_value(species_name), _export=False)
        model.add_component(parameter)
        initial = Initial(monomers[species_name](), parameter, _export=False)
        model.add_initial(initial)

    # 添加反应规则
    for reaction_index, reaction in enumerate(instance.reactions):
        reactants = _reaction_side_pattern(reaction.reactants, monomers)
        products = _reaction_side_pattern(reaction.products, monomers)
        rule_name = _unique_component_name(
            reaction.name or f"r_{reaction_index}",
            used_names,
        )

        rule_rate = rate_parameters[reaction.rate_key]
        symmetry_factor = _reactant_symmetry_factor(reaction.reactants)

        if symmetry_factor != 1.0:
            parameter_name = _unique_component_name(
                f"k_eff_{reaction.rate_key}_{reaction_index}",
                used_names,
            )
            parameter_value = (
                _reaction_base_rate_value(instance, reaction)
                * symmetry_factor
            )
            rule_rate = Parameter(
                parameter_name,
                parameter_value,
                _export=False,
            )
            model.add_component(rule_rate)

        rule = Rule(
            rule_name,
            reactants >> products,
            rule_rate,
            _export=False,
        )
        model.add_component(rule)
    
    # 添加观察物质
    output_observables: dict[str, str] = {}
    for species_name in instance.final_output_species:
        observable_name = _unique_component_name(f"obs_{species_name}", used_names)
        observable = Observable(observable_name, monomers[species_name](), _export=False)
        model.add_component(observable)
        output_observables[species_name] = observable_name

    response_observables: dict[str, str] = {}
    for species_name in instance.final_response_species:
        observable_name = _unique_component_name(
            f"obs_{species_name}",
            used_names,
        )
        observable = Observable(
            observable_name,
            monomers[species_name](),
            _export=False,
        )
        model.add_component(observable)
        response_observables[species_name] = observable_name

    inactive_output_species = _inactive_output_species(instance)
    inactive_output_observable: str | None = None
    if inactive_output_species is not None:
        inactive_output_observable = _unique_component_name(
            f"obs_{inactive_output_species}",
            used_names,
        )
        observable = Observable(
            inactive_output_observable,
            monomers[inactive_output_species](),
            _export=False,
        )
        model.add_component(observable)

    return PySBModelBundle(
        model=model,
        output_species=tuple(instance.final_output_species),
        output_observables=output_observables,
        inactive_output_species=inactive_output_species,
        inactive_output_observable=inactive_output_observable,
        response_species=tuple(instance.final_response_species),
        response_observables=response_observables,
    )

def simulate_crn_instance_with_pysb_diagnostics(
    instance: CRNInstance,
    *,
    config: PySBValidationConfig | None = None,
) -> PySBSimulationResult:
    """Simulate one CRN instance with PySB and return final outputs plus tail diagnostics."""

    config = config or PySBValidationConfig()
    config.validate_tail_config()

    bundle = build_pysb_model(instance)

    try:
        from pysb.integrate import odesolve
    except ImportError as exc:  # pragma: no cover
        raise PySBUnavailableError(
            "PySB is required for PySB validation. Install pysb in the runtime environment."
        ) from exc

    try:
        trajectories = odesolve(
            bundle.model,
            config.tspan(),
            integrator=config.integrator,
            cleanup=config.cleanup,
            verbose=config.verbose,
            **dict(config.integrator_options),
        )
    except Exception as exc:  # pragma: no cover
        raise PySBValidationError(
            "PySB simulation failed. Check that PySB and BioNetGen are correctly installed, "
            "and consider increasing t_end if the CRN has not reached steady state."
        ) from exc

    output_columns: list[np.ndarray] = []
    final_concentrations: list[float] = []

    for species_name in bundle.output_species:
        observable_name = bundle.output_observables[species_name]
        if observable_name not in trajectories.dtype.names:
            raise PySBValidationError(f"Missing PySB observable: {observable_name!r}.")

        observable_values = np.asarray(
            trajectories[observable_name],
            dtype=float,
        ).reshape(-1)

        if observable_values.size == 0:
            raise PySBValidationError(
                f"Empty PySB trajectory for observable: {observable_name!r}."
            )

        output_columns.append(observable_values)
        final_concentrations.append(float(observable_values[-1]))

    tail_species = list(bundle.output_species)
    final_Z_free: float | None = None
    inactive_column: np.ndarray | None = None
    if (
        bundle.inactive_output_species is not None
        and bundle.inactive_output_observable is not None
    ):
        observable_name = bundle.inactive_output_observable
        if observable_name not in trajectories.dtype.names:
            raise PySBValidationError(
                f"Missing PySB observable: {observable_name!r}."
            )
        inactive_column = np.asarray(
            trajectories[observable_name],
            dtype=float,
        ).reshape(-1)
        if inactive_column.size == 0:
            raise PySBValidationError(
                f"Empty PySB trajectory for observable: {observable_name!r}."
            )
        output_columns.append(inactive_column)
        tail_species.append(bundle.inactive_output_species)
        final_Z_free = float(inactive_column[-1])

    output_trajectory = np.stack(output_columns, axis=1)

    output_tail = diagnose_output_tail_convergence(
        output_trajectory,
        window=config.tail_window,
        atol=config.tail_atol,
        rtol=config.tail_rtol,
        eps=config.eps,
        species=tail_species,
    )

    response_columns: list[np.ndarray] = []
    final_response_concentrations: list[float] = []
    for species_name in bundle.response_species:
        observable_name = bundle.response_observables[species_name]
        if observable_name not in trajectories.dtype.names:
            raise PySBValidationError(
                f"Missing PySB response observable: {observable_name!r}."
            )
        values = np.asarray(
            trajectories[observable_name],
            dtype=float,
        ).reshape(-1)
        if values.size == 0:
            raise PySBValidationError(
                f"Empty PySB response trajectory: {observable_name!r}."
            )
        response_columns.append(values)
        final_response_concentrations.append(float(values[-1]))
    response_tail = (
        None
        if not response_columns
        else diagnose_output_tail_convergence(
            np.stack(response_columns, axis=1),
            window=config.tail_window,
            atol=config.tail_atol,
            rtol=config.tail_rtol,
            eps=config.eps,
            species=bundle.response_species,
        )
    )

    initial_output_pool_total: float | None = None
    final_output_pool_total: float | None = None
    active_output_fraction: float | None = None
    output_pool_conservation_error: float | None = None
    if bundle.inactive_output_species is not None:
        initial_output_pool_total = float(
            sum(
                instance.initial_value(species_name)
                for species_name in bundle.output_species
            )
            + instance.initial_value(bundle.inactive_output_species)
        )
        final_output_pool_total = float(
            sum(final_concentrations) + float(final_Z_free)
        )
        if final_output_pool_total > float(config.eps):
            active_output_fraction = float(
                sum(final_concentrations) / final_output_pool_total
            )
        output_pool_conservation_error = abs(
            final_output_pool_total - initial_output_pool_total
        )

    return PySBSimulationResult(
        final_concentrations=tuple(final_concentrations),
        output_tail=output_tail,
        final_Z_free=final_Z_free,
        initial_output_pool_total=initial_output_pool_total,
        final_output_pool_total=final_output_pool_total,
        active_output_fraction=active_output_fraction,
        output_pool_conservation_error=output_pool_conservation_error,
        final_response_concentrations=tuple(final_response_concentrations),
        response_tail=response_tail,
    )


def simulate_crn_instance_with_pysb(
    instance: CRNInstance,
    *,
    config: PySBValidationConfig | None = None,
) -> tuple[float, ...]:
    """Simulate one CRN instance with PySB and return final output concentrations."""

    return simulate_crn_instance_with_pysb_diagnostics(
        instance,
        config=config,
    ).final_concentrations

def validate_crn_instance_with_pysb(
    instance: CRNInstance,
    *,
    reference_r: Sequence[float] | None = None,
    reference_Z: Sequence[float] | None = None,
    reference_Z_free: float | None = None,
    reference_active_output_fraction: float | None = None,
    reference_normalized_class_scores: Sequence[float] | None = None,
    reference_probabilities: Sequence[float] | None = None,
    target: int | None = None,
    sample_index: int | None = None,
    config: PySBValidationConfig | None = None,
) -> PySBValidationResult:
    """Validate one CRN instance against the digital terminal-output state.

    ``reference_Z`` and ``reference_Z_free`` are physical output
    concentrations. ``reference_normalized_class_scores`` contains
    ``Z_k / sum_j Z_j`` and excludes ``Z_free``. ``reference_probabilities``
    remains a compatibility alias for normalized class scores.
    """

    config = config or PySBValidationConfig()
    simulation = simulate_crn_instance_with_pysb_diagnostics(instance, config=config)
    concentrations = simulation.final_concentrations
    pysb_probabilities = normalized_class_scores_from_concentrations(
        concentrations,
        eps=config.eps,
    )
    pysb_pred = int(np.argmax(np.asarray(pysb_probabilities, dtype=float)))

    if (
        reference_normalized_class_scores is not None
        and reference_probabilities is not None
    ):
        normalized_array = np.asarray(
            reference_normalized_class_scores,
            dtype=float,
        ).reshape(-1)
        legacy_array = np.asarray(
            reference_probabilities,
            dtype=float,
        ).reshape(-1)
        if normalized_array.shape != legacy_array.shape or not np.allclose(
            normalized_array,
            legacy_array,
            atol=0.0,
            rtol=0.0,
        ):
            raise ValueError(
                "reference_normalized_class_scores and "
                "reference_probabilities disagree."
            )

    normalized_reference = (
        reference_normalized_class_scores
        if reference_normalized_class_scores is not None
        else reference_probabilities
    )
    reference_tuple = _optional_vector_tuple(
        normalized_reference,
        expected_size=len(pysb_probabilities),
        name="reference_normalized_class_scores",
    )
    reference_r_tuple = _optional_vector_tuple(
        reference_r,
        expected_size=len(pysb_probabilities),
        name="reference_r",
    )
    reference_Z_tuple = _optional_vector_tuple(
        reference_Z,
        expected_size=len(pysb_probabilities),
        name="reference_Z",
    )
    reference_Z_free_value = _optional_nonnegative_float(
        reference_Z_free,
        name="reference_Z_free",
    )
    reference_active_fraction_value = _optional_nonnegative_float(
        reference_active_output_fraction,
        name="reference_active_output_fraction",
    )
    if (
        reference_active_fraction_value is not None
        and reference_active_fraction_value > 1.0 + float(config.eps)
    ):
        raise ValueError(
            "reference_active_output_fraction must not exceed 1."
        )

    reference_pred = (
        int(np.argmax(np.asarray(reference_tuple, dtype=float)))
        if reference_tuple is not None
        else None
    )

    max_abs_error: float | None = None
    max_rel_error: float | None = None
    normalized_scores_within_tolerance: bool | None = None
    class_match: bool | None = None

    if reference_tuple is not None:
        reference_array = np.asarray(reference_tuple, dtype=float)
        pysb_array = np.asarray(pysb_probabilities, dtype=float)
        abs_errors = np.abs(pysb_array - reference_array)
        rel_errors = abs_errors / np.maximum(np.abs(reference_array), float(config.eps))
        max_abs_error = float(np.max(abs_errors))
        max_rel_error = float(np.max(rel_errors))
        normalized_scores_within_tolerance = bool(
            np.allclose(pysb_array, reference_array, atol=float(config.atol), rtol=float(config.rtol))
        )
        class_match = bool(pysb_pred == reference_pred)

    pysb_r_tuple: tuple[float, ...] | None = (
        tuple(float(value) for value in simulation.final_response_concentrations)
        if simulation.final_response_concentrations
        else None
    )
    max_abs_r_error: float | None = None
    max_rel_r_error: float | None = None
    response_within_tolerance: bool | None = None
    if reference_r_tuple is not None:
        if pysb_r_tuple is not None:
            reference_r_array = np.asarray(reference_r_tuple, dtype=float)
            pysb_r_array = np.asarray(pysb_r_tuple, dtype=float)
            r_abs_errors = np.abs(pysb_r_array - reference_r_array)
            r_rel_errors = r_abs_errors / np.maximum(
                np.abs(reference_r_array),
                float(config.eps),
            )
            max_abs_r_error = float(np.max(r_abs_errors))
            max_rel_r_error = float(np.max(r_rel_errors))
            response_within_tolerance = bool(
                np.allclose(
                    pysb_r_array,
                    reference_r_array,
                    atol=float(config.atol),
                    rtol=float(config.rtol),
                )
            )

    max_abs_Z_error: float | None = None
    max_rel_Z_error: float | None = None
    Z_within_tolerance: bool | None = None
    if reference_Z_tuple is not None:
        reference_Z_array = np.asarray(reference_Z_tuple, dtype=float)
        pysb_Z_array = np.asarray(concentrations, dtype=float)
        Z_abs_errors = np.abs(pysb_Z_array - reference_Z_array)
        Z_rel_errors = Z_abs_errors / np.maximum(
            np.abs(reference_Z_array),
            float(config.eps),
        )
        max_abs_Z_error = float(np.max(Z_abs_errors))
        max_rel_Z_error = float(np.max(Z_rel_errors))
        Z_within_tolerance = bool(
            np.allclose(
                pysb_Z_array,
                reference_Z_array,
                atol=float(config.atol),
                rtol=float(config.rtol),
            )
        )

    abs_Z_free_error: float | None = None
    rel_Z_free_error: float | None = None
    Z_free_within_tolerance: bool | None = None
    if (
        reference_Z_free_value is not None
        and simulation.final_Z_free is not None
    ):
        abs_Z_free_error = abs(
            float(simulation.final_Z_free) - reference_Z_free_value
        )
        rel_Z_free_error = abs_Z_free_error / max(
            abs(reference_Z_free_value),
            float(config.eps),
        )
        Z_free_within_tolerance = bool(
            np.isclose(
                float(simulation.final_Z_free),
                reference_Z_free_value,
                atol=float(config.atol),
                rtol=float(config.rtol),
            )
        )
    elif reference_Z_free_value is not None:
        Z_free_within_tolerance = False

    raw_checks = [
        value
        for value in (
            response_within_tolerance,
            Z_within_tolerance,
            Z_free_within_tolerance,
        )
        if value is not None
    ]
    raw_concentrations_within_tolerance = (
        bool(all(raw_checks)) if raw_checks else None
    )

    reference_output_pool_total: float | None = None
    if reference_Z_tuple is not None and reference_Z_free_value is not None:
        reference_output_pool_total = float(
            sum(reference_Z_tuple) + reference_Z_free_value
        )
        if reference_active_fraction_value is None:
            if reference_output_pool_total > float(config.eps):
                reference_active_fraction_value = float(
                    sum(reference_Z_tuple) / reference_output_pool_total
                )

    output_pool_conserved: bool | None = None
    if (
        simulation.initial_output_pool_total is not None
        and simulation.final_output_pool_total is not None
    ):
        output_pool_conserved = bool(
            np.isclose(
                simulation.final_output_pool_total,
                simulation.initial_output_pool_total,
                atol=float(config.atol),
                rtol=float(config.rtol),
            )
        )

    overall_checks = [
        value
        for value in (
            normalized_scores_within_tolerance,
            raw_concentrations_within_tolerance,
            output_pool_conserved,
        )
        if value is not None
    ]
    within_tolerance = bool(all(overall_checks)) if overall_checks else None

    target_int = int(target) if target is not None else None
    target_match = bool(pysb_pred == target_int) if target_int is not None else None

    return PySBValidationResult(
        sample_index=sample_index,
        target=target_int,
        reference_pred=reference_pred,
        pysb_pred=pysb_pred,
        reference_probabilities=reference_tuple,
        pysb_probabilities=pysb_probabilities,
        pysb_concentrations=tuple(float(value) for value in concentrations),
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        class_match=class_match,
        target_match=target_match,
        within_tolerance=within_tolerance,
        output_tail=simulation.output_tail,
        reference_r=reference_r_tuple,
        reference_Z=reference_Z_tuple,
        reference_Z_free=reference_Z_free_value,
        pysb_Z_free=simulation.final_Z_free,
        reference_output_pool_total=reference_output_pool_total,
        pysb_output_pool_total=simulation.final_output_pool_total,
        output_pool_conservation_error=(
            simulation.output_pool_conservation_error
        ),
        reference_active_output_fraction=(
            reference_active_fraction_value
        ),
        pysb_active_output_fraction=simulation.active_output_fraction,
        max_abs_Z_error=max_abs_Z_error,
        max_rel_Z_error=max_rel_Z_error,
        abs_Z_free_error=abs_Z_free_error,
        rel_Z_free_error=rel_Z_free_error,
        raw_concentrations_within_tolerance=(
            raw_concentrations_within_tolerance
        ),
        normalized_scores_within_tolerance=(
            normalized_scores_within_tolerance
        ),
        output_pool_conserved=output_pool_conserved,
        pysb_r=pysb_r_tuple,
        max_abs_r_error=max_abs_r_error,
        max_rel_r_error=max_rel_r_error,
        response_within_tolerance=response_within_tolerance,
        response_tail=simulation.response_tail,
    )

def validate_model_spec_with_pysb(
    model_spec: Mapping[str, Any] | Any,
    inputs: Sequence[Sequence[float]] | np.ndarray,
    *,
    reference_r: Sequence[Sequence[float]] | np.ndarray | None = None,
    reference_Z: Sequence[Sequence[float]] | np.ndarray | None = None,
    reference_Z_free: Sequence[float] | np.ndarray | None = None,
    reference_active_output_fraction: (
        Sequence[float] | np.ndarray | None
    ) = None,
    reference_normalized_class_scores: (
        Sequence[Sequence[float]] | np.ndarray | None
    ) = None,
    reference_probabilities: Sequence[Sequence[float]] | np.ndarray | None = None,
    targets: Sequence[int] | np.ndarray | None = None,
    sample_indices: Sequence[int] | None = None,
    config: PySBValidationConfig | None = None,
    instance_options: CRNInstanceOptions | None = None,
) -> PySBBatchValidationResult:
    """Validate a frozen compiled model spec on many preprocessed input samples."""

    config = config or PySBValidationConfig()
    input_array = _as_2d_float_array(inputs, name="inputs")
    n_samples = int(input_array.shape[0])

    if (
        reference_normalized_class_scores is not None
        and reference_probabilities is not None
    ):
        normalized_array = _as_2d_float_array(
            reference_normalized_class_scores,
            name="reference_normalized_class_scores",
        )
        legacy_array = _as_2d_float_array(
            reference_probabilities,
            name="reference_probabilities",
        )
        if normalized_array.shape != legacy_array.shape or not np.allclose(
            normalized_array,
            legacy_array,
            atol=0.0,
            rtol=0.0,
        ):
            raise ValueError(
                "reference_normalized_class_scores and "
                "reference_probabilities disagree."
            )

    normalized_reference_values = (
        reference_normalized_class_scores
        if reference_normalized_class_scores is not None
        else reference_probabilities
    )
    reference_array = _optional_reference_matrix(
        normalized_reference_values,
        name="reference_normalized_class_scores",
        n_samples=n_samples,
    )
    reference_r_array = _optional_reference_matrix(
        reference_r,
        name="reference_r",
        n_samples=n_samples,
    )
    reference_Z_array = _optional_reference_matrix(
        reference_Z,
        name="reference_Z",
        n_samples=n_samples,
    )
    reference_Z_free_array = _optional_reference_vector(
        reference_Z_free,
        name="reference_Z_free",
        n_samples=n_samples,
    )
    reference_active_fraction_array = _optional_reference_vector(
        reference_active_output_fraction,
        name="reference_active_output_fraction",
        n_samples=n_samples,
    )

    target_array = None
    if targets is not None:
        target_array = np.asarray(targets, dtype=int).reshape(-1)
        if target_array.shape[0] != n_samples:
            raise ValueError("targets must have the same number of elements as inputs rows.")

    if sample_indices is None:
        sample_index_tuple: tuple[int, ...] = tuple(range(n_samples))
    else:
        sample_index_tuple = tuple(int(index) for index in sample_indices)
        if len(sample_index_tuple) != n_samples:
            raise ValueError("sample_indices must have the same length as inputs rows.")

    base_options = instance_options or CRNInstanceOptions()
    results: list[PySBValidationResult] = []
    for row_index in range(n_samples):
        options = replace(base_options, input_values=tuple(float(v) for v in input_array[row_index]))
        instance = build_crn_instance(model_spec, options=options)
        result = validate_crn_instance_with_pysb(
            instance,
            reference_r=(
                None
                if reference_r_array is None
                else reference_r_array[row_index]
            ),
            reference_Z=(
                None
                if reference_Z_array is None
                else reference_Z_array[row_index]
            ),
            reference_Z_free=(
                None
                if reference_Z_free_array is None
                else float(reference_Z_free_array[row_index])
            ),
            reference_active_output_fraction=(
                None
                if reference_active_fraction_array is None
                else float(reference_active_fraction_array[row_index])
            ),
            reference_normalized_class_scores=(
                None
                if reference_array is None
                else reference_array[row_index]
            ),
            target=None if target_array is None else int(target_array[row_index]),
            sample_index=sample_index_tuple[row_index],
            config=config,
        )
        results.append(result)

    result_tuple = tuple(results)
    return PySBBatchValidationResult(results=result_tuple, summary=summarize_pysb_validation(result_tuple))


def summarize_pysb_validation(results: Sequence[PySBValidationResult]) -> PySBValidationSummary:
    """Aggregate per-sample PySB validation results."""

    result_tuple = tuple(results)
    n_samples = len(result_tuple)
    n_with_reference = sum(result.class_match is not None for result in result_tuple)
    n_with_target = sum(result.target_match is not None for result in result_tuple)
    n_with_tolerance = sum(result.within_tolerance is not None for result in result_tuple)

    n_class_matches = sum(result.class_match is True for result in result_tuple)
    n_target_matches = sum(result.target_match is True for result in result_tuple)
    n_within_tolerance = sum(result.within_tolerance is True for result in result_tuple)

    abs_errors = [
        result.max_abs_error
        for result in result_tuple
        if result.max_abs_error is not None
    ]
    rel_errors = [
        result.max_rel_error
        for result in result_tuple
        if result.max_rel_error is not None
    ]

    output_tails = [
        result.output_tail
        for result in result_tuple
        if result.output_tail is not None
    ]
    tail_abs_deltas = [tail.max_abs_delta for tail in output_tails]
    tail_rel_deltas = [tail.max_rel_delta for tail in output_tails]

    n_output_tail_checked = len(output_tails)
    n_output_tail_steady_state_like = sum(
        tail.steady_state_like is True
        for tail in output_tails
    )

    n_with_raw_reference = sum(
        result.raw_concentrations_within_tolerance is not None
        for result in result_tuple
    )
    n_raw_concentrations_within_tolerance = sum(
        result.raw_concentrations_within_tolerance is True
        for result in result_tuple
    )
    n_with_normalized_score_reference = sum(
        result.normalized_scores_within_tolerance is not None
        for result in result_tuple
    )
    n_normalized_scores_within_tolerance = sum(
        result.normalized_scores_within_tolerance is True
        for result in result_tuple
    )
    n_output_pool_conservation_checked = sum(
        result.output_pool_conserved is not None
        for result in result_tuple
    )
    n_output_pool_conserved = sum(
        result.output_pool_conserved is True
        for result in result_tuple
    )

    Z_abs_errors = [
        result.max_abs_Z_error
        for result in result_tuple
        if result.max_abs_Z_error is not None
    ]
    Z_rel_errors = [
        result.max_rel_Z_error
        for result in result_tuple
        if result.max_rel_Z_error is not None
    ]
    Z_free_abs_errors = [
        result.abs_Z_free_error
        for result in result_tuple
        if result.abs_Z_free_error is not None
    ]
    Z_free_rel_errors = [
        result.rel_Z_free_error
        for result in result_tuple
        if result.rel_Z_free_error is not None
    ]
    conservation_errors = [
        result.output_pool_conservation_error
        for result in result_tuple
        if result.output_pool_conservation_error is not None
    ]
    n_with_response_reference = sum(
        result.response_within_tolerance is not None
        for result in result_tuple
    )
    n_responses_within_tolerance = sum(
        result.response_within_tolerance is True
        for result in result_tuple
    )
    r_abs_errors = [
        result.max_abs_r_error
        for result in result_tuple
        if result.max_abs_r_error is not None
    ]
    r_rel_errors = [
        result.max_rel_r_error
        for result in result_tuple
        if result.max_rel_r_error is not None
    ]
    response_tails = [
        result.response_tail
        for result in result_tuple
        if result.response_tail is not None
    ]
    n_response_tail_steady = sum(
        tail.steady_state_like is True for tail in response_tails
    )

    return PySBValidationSummary(
        n_samples=n_samples,
        n_with_reference=n_with_reference,
        n_class_matches=n_class_matches,
        n_target_matches=n_target_matches,
        n_within_tolerance=n_within_tolerance,
        class_match_rate=_safe_rate(n_class_matches, n_with_reference),
        target_match_rate=_safe_rate(n_target_matches, n_with_target),
        tolerance_pass_rate=_safe_rate(n_within_tolerance, n_with_tolerance),
        max_abs_error=float(max(abs_errors)) if abs_errors else None,
        max_rel_error=float(max(rel_errors)) if rel_errors else None,
        n_output_tail_checked=n_output_tail_checked,
        n_output_tail_steady_state_like=n_output_tail_steady_state_like,
        output_tail_steady_state_rate=_safe_rate(
            n_output_tail_steady_state_like,
            n_output_tail_checked,
        ),
        max_output_tail_abs_delta=(
            float(max(tail_abs_deltas)) if tail_abs_deltas else None
        ),
        max_output_tail_rel_delta=(
            float(max(tail_rel_deltas)) if tail_rel_deltas else None
        ),
        mean_output_tail_abs_delta=(
            float(sum(tail_abs_deltas) / len(tail_abs_deltas))
            if tail_abs_deltas
            else None
        ),
        mean_output_tail_rel_delta=(
            float(sum(tail_rel_deltas) / len(tail_rel_deltas))
            if tail_rel_deltas
            else None
        ),
        n_with_raw_reference=n_with_raw_reference,
        n_raw_concentrations_within_tolerance=(
            n_raw_concentrations_within_tolerance
        ),
        raw_concentration_tolerance_pass_rate=_safe_rate(
            n_raw_concentrations_within_tolerance,
            n_with_raw_reference,
        ),
        n_normalized_scores_within_tolerance=(
            n_normalized_scores_within_tolerance
        ),
        normalized_score_tolerance_pass_rate=_safe_rate(
            n_normalized_scores_within_tolerance,
            n_with_normalized_score_reference,
        ),
        n_output_pool_conservation_checked=(
            n_output_pool_conservation_checked
        ),
        n_output_pool_conserved=n_output_pool_conserved,
        output_pool_conservation_pass_rate=_safe_rate(
            n_output_pool_conserved,
            n_output_pool_conservation_checked,
        ),
        max_abs_Z_error=(
            float(max(Z_abs_errors)) if Z_abs_errors else None
        ),
        max_rel_Z_error=(
            float(max(Z_rel_errors)) if Z_rel_errors else None
        ),
        max_abs_Z_free_error=(
            float(max(Z_free_abs_errors))
            if Z_free_abs_errors
            else None
        ),
        max_rel_Z_free_error=(
            float(max(Z_free_rel_errors))
            if Z_free_rel_errors
            else None
        ),
        max_output_pool_conservation_error=(
            float(max(conservation_errors))
            if conservation_errors
            else None
        ),
        n_with_response_reference=n_with_response_reference,
        n_responses_within_tolerance=n_responses_within_tolerance,
        response_tolerance_pass_rate=_safe_rate(
            n_responses_within_tolerance,
            n_with_response_reference,
        ),
        max_abs_r_error=(
            float(max(r_abs_errors)) if r_abs_errors else None
        ),
        max_rel_r_error=(
            float(max(r_rel_errors)) if r_rel_errors else None
        ),
        n_response_tail_checked=len(response_tails),
        n_response_tail_steady_state_like=n_response_tail_steady,
        response_tail_steady_state_rate=_safe_rate(
            n_response_tail_steady,
            len(response_tails),
        ),
    )

def _validate_instance(instance: CRNInstance) -> None:
    species_set = set(instance.species)
    if not instance.species:
        raise CRNInstanceError("CRN instance must contain at least one species.")
    if not instance.final_output_species:
        raise CRNInstanceError("CRN instance must define final_output_species.")
    for species_name in instance.final_output_species:
        if species_name not in species_set:
            raise CRNInstanceError(f"Unknown final output species: {species_name!r}.")
    for species_name in instance.final_response_species:
        if species_name not in species_set:
            raise CRNInstanceError(
                f"Unknown final response species: {species_name!r}."
            )

    for reaction in instance.reactions:
        for species_name in (*reaction.reactants, *reaction.products):
            if species_name not in species_set:
                raise CRNInstanceError(f"Reaction references unknown species: {species_name!r}.")
        if not math.isfinite(float(reaction.rate_value)) or float(reaction.rate_value) < 0.0:
            raise CRNInstanceError(f"Reaction has invalid non-negative finite rate: {reaction!r}.")

    for species_name in instance.species:
        initial_value = instance.initial_value(species_name)
        if not math.isfinite(initial_value) or initial_value < 0.0:
            raise CRNInstanceError(f"Species has invalid non-negative finite initial value: {species_name!r}.")


def _reaction_side_pattern(species_names: Sequence[str], monomers: Mapping[str, Any]) -> Any:
    if not species_names:
        return None
    iterator = iter(species_names)
    first = next(iterator)
    pattern = monomers[first]()
    for species_name in iterator:
        pattern = pattern + monomers[species_name]()
    return pattern


def _safe_pysb_identifier(value: str) -> str:
    text = re.sub(r"\W+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "component"
    if text[0].isdigit():
        text = f"c_{text}"
    if text.startswith("__"):
        text = f"c{text}"
    return text


def _unique_component_name(value: str, used_names: set[str]) -> str:
    base = _safe_pysb_identifier(value)
    candidate = base
    suffix = 1
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _optional_vector_tuple(
    values: Sequence[float] | np.ndarray | None,
    *,
    expected_size: int,
    name: str,
) -> tuple[float, ...] | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != expected_size:
        raise ValueError(
            f"{name} must contain {expected_size} values, got {array.size}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    return tuple(float(value) for value in array)


def _optional_probability_tuple(
    values: Sequence[float] | np.ndarray | None,
    *,
    expected_size: int,
) -> tuple[float, ...] | None:
    """Compatibility wrapper for normalized class-score references."""

    return _optional_vector_tuple(
        values,
        expected_size=expected_size,
        name="reference_probabilities",
    )


def _optional_nonnegative_float(
    value: float | None,
    *,
    name: str,
) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number.")
    return result


def _optional_reference_matrix(
    values: Sequence[Sequence[float]] | np.ndarray | None,
    *,
    name: str,
    n_samples: int,
) -> np.ndarray | None:
    if values is None:
        return None
    array = _as_2d_float_array(values, name=name)
    if array.shape[0] != n_samples:
        raise ValueError(
            f"{name} must have the same number of rows as inputs."
        )
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    return array


def _optional_reference_vector(
    values: Sequence[float] | np.ndarray | None,
    *,
    name: str,
    n_samples: int,
) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.shape[0] != n_samples:
        raise ValueError(
            f"{name} must have the same number of elements as inputs rows."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    return array


def _as_2d_float_array(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D numeric array.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


__all__ = [
    "PySBBatchValidationResult",
    "PySBModelBundle",
    "PySBOutputTailDiagnostic",
    "PySBSimulationResult",
    "PySBUnavailableError",
    "PySBValidationConfig",
    "PySBValidationError",
    "PySBValidationResult",
    "PySBValidationSummary",
    "build_pysb_model",
    "concentration_probabilities",
    "diagnose_output_tail_convergence",
    "normalized_class_scores_from_concentrations",
    "simulate_crn_instance_with_pysb",
    "simulate_crn_instance_with_pysb_diagnostics",
    "summarize_pysb_validation",
    "validate_crn_instance_with_pysb",
    "validate_model_spec_with_pysb",
]
