from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from crn_validation.pysb import PySBValidationConfig, validate_model_spec_with_pysb
from export.crn_exporter import CRNExportOptions, write_visual_dsd_crn
from export.crn_instance import build_crn_instance


@dataclass(frozen=True)
class CRNReferenceCase:
    input_values: tuple[float, ...]
    target: int
    predicted: int
    sample_index: int
    r: tuple[float, ...]
    Z: tuple[float, ...]
    Z_free: float
    active_output_fraction: float
    normalized_scores: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_values": self.input_values,
            "target": self.target,
            "predicted": self.predicted,
            "sample_index": self.sample_index,
            "reference_r": self.r,
            "reference_Z": self.Z,
            "reference_Z_free": self.Z_free,
            "active_output_fraction": self.active_output_fraction,
            "normalized_class_scores": self.normalized_scores,
        }


def network_output_state(state: Any) -> Any:
    nested = getattr(state, "network", None)
    return state if nested is None else nested


@torch.no_grad()
def reference_case_from_state(
    state: Any,
    *,
    crn_input: torch.Tensor,
    target: int,
    sample_index: int,
) -> CRNReferenceCase:
    network_state = network_output_state(state)
    if crn_input.ndim != 1:
        raise ValueError("crn_input must be a one-dimensional sample.")
    return CRNReferenceCase(
        input_values=tuple(float(item) for item in crn_input.detach().cpu().tolist()),
        target=int(target),
        predicted=int(network_state.predicted_classes[0].item()),
        sample_index=int(sample_index),
        r=tuple(
            float(item)
            for item in network_state.final_rational_state.r[0]
            .detach()
            .cpu()
            .tolist()
        ),
        Z=tuple(
            float(item)
            for item in network_state.Z[0].detach().cpu().tolist()
        ),
        Z_free=float(network_state.Z_free[0, 0].item()),
        active_output_fraction=float(
            network_state.output.active_output_fraction[0, 0].item()
        ),
        normalized_scores=tuple(
            float(item)
            for item in network_state.normalized_scores()[0]
            .detach()
            .cpu()
            .tolist()
        ),
    )


def compile_crn_artifacts(
    export_model: Any,
    network: Any,
    *,
    reference: CRNReferenceCase | None = None,
    output_path: str | Path | None = None,
    count_metadata: Mapping[str, Any] | None = None,
    extra_comments: Sequence[str] = (),
    final_time: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Build structural counts and optionally export one Visual DSD CRN."""

    export_method = getattr(export_model, "export_model_spec", None)
    if not callable(export_method):
        raise TypeError("export_model must expose export_model_spec().")
    model_spec = export_method()
    instance = build_crn_instance(model_spec)
    counts: dict[str, Any] = {
        "static_compile_passed": True,
        "species_count": instance.species_count,
        "reaction_count": instance.reaction_count,
        "parameter_count": instance.parameter_count,
        "rational_depth": int(getattr(network, "depth")),
        "widths": tuple(int(item) for item in getattr(network, "widths")),
    }
    if count_metadata is not None:
        counts.update(dict(count_metadata))
    if output_path is None:
        return counts, None
    if reference is None:
        raise ValueError("reference is required when output_path is provided.")
    destination = Path(output_path)
    export = write_visual_dsd_crn(
        model_spec,
        destination,
        CRNExportOptions(
            input_values=reference.input_values,
            target_class=reference.target,
            predicted_class=reference.predicted,
            reference_r=reference.r,
            reference_Z=reference.Z,
            reference_Z_free=reference.Z_free,
            active_output_fraction=reference.active_output_fraction,
            normalized_class_scores=reference.normalized_scores,
            final_time=final_time,
            extra_comments=tuple(extra_comments),
        ),
    )
    return counts, {
        "path": destination.as_posix(),
        "species_count": export.species_count,
        "reaction_count": export.reaction_count,
        "parameter_count": export.parameter_count,
    }


def adaptive_pysb_validation(
    model_spec: Any,
    reference: CRNReferenceCase,
    *,
    initial_t_end: float = 20.0,
    max_t_end: float = 320.0,
    n_timepoints: int = 201,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    tail_atol: float = 1e-6,
    tail_rtol: float = 1e-6,
) -> dict[str, Any]:
    """Run one adaptive PySB validation case with a shared pass criterion."""

    if not 0.0 < initial_t_end <= max_t_end:
        raise ValueError("require 0 < initial_t_end <= max_t_end.")
    attempts: list[dict[str, Any]] = []
    t_end = float(initial_t_end)
    last_result: dict[str, Any] | None = None
    while True:
        batch = validate_model_spec_with_pysb(
            model_spec,
            np.asarray([reference.input_values], dtype=np.float64),
            reference_r=np.asarray([reference.r], dtype=np.float64),
            reference_Z=np.asarray([reference.Z], dtype=np.float64),
            reference_Z_free=np.asarray([reference.Z_free], dtype=np.float64),
            reference_active_output_fraction=np.asarray(
                [reference.active_output_fraction], dtype=np.float64
            ),
            reference_normalized_class_scores=np.asarray(
                [reference.normalized_scores], dtype=np.float64
            ),
            targets=np.asarray([reference.target], dtype=np.int64),
            sample_indices=[reference.sample_index],
            config=PySBValidationConfig(
                t_end=t_end,
                n_timepoints=n_timepoints,
                atol=atol,
                rtol=rtol,
                tail_atol=tail_atol,
                tail_rtol=tail_rtol,
            ),
        ).to_dict()
        last_result = batch["results"][0]
        output_tail = last_result.get("output_pool_tail") or {}
        response_tail = last_result.get("response_tail") or {}
        passed = bool(
            last_result.get("overall_within_tolerance")
            and output_tail.get("steady_state_like")
            and response_tail.get("steady_state_like")
        )
        attempts.append(
            {
                "t_end": t_end,
                "passed": passed,
                "overall_within_tolerance": last_result.get(
                    "overall_within_tolerance"
                ),
                "output_tail_steady_state_like": output_tail.get(
                    "steady_state_like"
                ),
                "response_tail_steady_state_like": response_tail.get(
                    "steady_state_like"
                ),
                "max_abs_r_error": last_result.get("max_abs_r_error"),
                "max_abs_Z_error": last_result.get("max_abs_Z_error"),
                "max_abs_normalized_score_error": last_result.get(
                    "max_abs_normalized_score_error"
                ),
            }
        )
        if passed or t_end >= max_t_end:
            return {
                "passed": passed,
                "final_t_end": t_end,
                "attempts": attempts,
                "result": last_result,
            }
        t_end = min(float(max_t_end), 2.0 * t_end)


__all__ = [
    "CRNReferenceCase",
    "adaptive_pysb_validation",
    "compile_crn_artifacts",
    "network_output_state",
    "reference_case_from_state",
]
