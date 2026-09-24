from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from export.crn_instance import (
    CRNInstance,
    CRNInstanceError,
    CRNInstanceOptions,
    build_crn_instance,
    safe_crn_identifier,
)
from export.spec_utils import spec_to_payload


class CRNExportError(ValueError):
    """Raised when a compiled model spec cannot be exported as CRN DSL."""


@dataclass(frozen=True)
class CRNExportOptions:
    """Options for exporting one frozen model spec and one input sample."""

    input_values: Sequence[float] | None = None
    target_class: int | None = None
    predicted_class: int | None = None
    reference_r: Sequence[float] | None = None
    reference_Z: Sequence[float] | None = None
    reference_Z_free: float | None = None
    active_output_fraction: float | None = None
    normalized_class_scores: Sequence[float] | None = None
    expected_output: Any | None = None
    predicted_output: Any | None = None
    predicted_scores: Sequence[float] | None = None
    final_time: float = 80.0
    plots: Sequence[str] | None = None
    initial_z_free: float = 1.0
    require_frozen_rates: bool = True
    include_comments: bool = True
    include_simulation_directives: bool = True
    numeric_format: str = ".8g"
    parameter_indent: str = "  "
    extra_comments: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class CRNExportResult:
    """Rendered CRN DSL text plus structural counts."""

    text: str
    species_count: int
    reaction_count: int
    parameter_count: int
    species: tuple[str, ...]
    parameters: Mapping[str, float]
    metadata: Mapping[str, Any]

    def write(self, path: str | Path, *, encoding: str = "utf-8") -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.text, encoding=encoding)
        return output_path


def export_visual_dsd_crn(
    spec: Any,
    options: CRNExportOptions | None = None,
) -> CRNExportResult:
    """Export a frozen model specification as Visual DSD-style CRN text."""

    opts = options or CRNExportOptions()

    try:
        instance = build_crn_instance(
            spec,
            CRNInstanceOptions(
                input_values=opts.input_values,
                initial_z_free=opts.initial_z_free,
                require_frozen_rates=opts.require_frozen_rates,
            ),
        )
    except CRNInstanceError as exc:
        raise CRNExportError(str(exc)) from exc

    lines = _render_visual_dsd_text(instance, opts)
    metadata = _metadata(instance, opts)

    return CRNExportResult(
        text="\n".join(lines).rstrip() + "\n",
        species_count=instance.species_count,
        reaction_count=instance.reaction_count,
        parameter_count=instance.parameter_count,
        species=instance.species,
        parameters=dict(instance.parameters),
        metadata=metadata,
    )


def write_visual_dsd_crn(
    spec: Any,
    output_file: str | Path,
    options: CRNExportOptions | None = None,
    *,
    encoding: str = "utf-8",
) -> CRNExportResult:
    """Export a CRN DSL file and return the rendered result metadata."""

    result = export_visual_dsd_crn(spec, options=options)
    result.write(output_file, encoding=encoding)
    return result


def _render_visual_dsd_text(instance: CRNInstance, options: CRNExportOptions) -> list[str]:
    lines: list[str] = []

    if options.include_comments:
        lines.extend(_comment_lines(instance, options))
        lines.append("")

    if options.include_simulation_directives:
        plot_names = _plot_names(instance, options)
        plot_payload = "; ".join(plot_names)
        lines.append(
            f"directive simulation "
            f"{{final={_format_number(options.final_time, options)}; "
            f"plots=[{plot_payload}]; }}"
        )
        lines.append("directive simulator deterministic")

    lines.append("directive parameters [")
    for key, value in instance.parameters.items():
        lines.append(f"{options.parameter_indent}{key} = {_format_number(value, options)};")
    lines.append("]")
    lines.append("")

    for species, initial in instance.initials.items():
        lines.append(f"| {_format_number(initial, options)} {species}")
    lines.append("")

    for reaction in instance.reactions:
        left = _format_side(reaction.reactants)
        right = _format_side(reaction.products)
        lines.append(f"| {left} ->{{{reaction.rate_key}}} {right}")

    return lines


def _comment_lines(instance: CRNInstance, options: CRNExportOptions) -> list[str]:
    metadata = _metadata(instance, options)

    comments = [
        "// Exported CRN DSL: Visual DSD-compatible CRN syntax.",
        f"// model_name: {metadata['model_name']}",
        f"// species_count: {metadata['species_count']}",
        f"// reaction_count: {metadata['reaction_count']}",
        f"// parameter_count: {metadata['parameter_count']}",
        f"// input: {_json_compact(metadata['input'])}",
    ]

    if metadata.get("target_class") is not None:
        comments.append(f"// target_class: {_json_compact(metadata['target_class'])}")
    elif metadata.get("expected_output") is not None:
        comments.append(f"// expected_output: {_json_compact(metadata['expected_output'])}")

    if metadata.get("predicted_class") is not None:
        comments.append(
            f"// predicted_class: {_json_compact(metadata['predicted_class'])}"
        )
    elif metadata.get("predicted_output") is not None:
        comments.append(f"// predicted_output: {_json_compact(metadata['predicted_output'])}")

    if metadata.get("reference_r") is not None:
        comments.append(f"// reference_r: {_json_compact(metadata['reference_r'])}")

    if metadata.get("reference_Z") is not None:
        comments.append(f"// reference_Z: {_json_compact(metadata['reference_Z'])}")

    if metadata.get("reference_Z_free") is not None:
        comments.append(
            f"// reference_Z_free: {_json_compact(metadata['reference_Z_free'])}"
        )

    if metadata.get("active_output_fraction") is not None:
        comments.append(
            "// active_output_fraction: "
            f"{_json_compact(metadata['active_output_fraction'])}"
        )

    if metadata.get("normalized_class_scores") is not None:
        comments.append(
            "// normalized_class_scores: "
            f"{_json_compact(metadata['normalized_class_scores'])}"
        )
        comments.append(
            "// normalized_class_scores[k] = "
            "reference_Z[k] / sum_j(reference_Z[j]); "
            "Z_free is excluded."
        )
    elif metadata.get("predicted_scores") is not None:
        comments.append(f"// predicted_scores: {_json_compact(metadata['predicted_scores'])}")

    for comment in options.extra_comments:
        comments.append(f"// {str(comment)}")

    return comments


def _metadata(instance: CRNInstance, options: CRNExportOptions) -> dict[str, Any]:
    return {
        "model_name": instance.model_name,
        "species_count": instance.species_count,
        "reaction_count": instance.reaction_count,
        "parameter_count": instance.parameter_count,
        "input": spec_to_payload(options.input_values),
        "target_class": spec_to_payload(options.target_class),
        "predicted_class": spec_to_payload(options.predicted_class),
        "reference_r": spec_to_payload(options.reference_r),
        "reference_Z": spec_to_payload(options.reference_Z),
        "reference_Z_free": spec_to_payload(options.reference_Z_free),
        "active_output_fraction": spec_to_payload(
            options.active_output_fraction
        ),
        "normalized_class_scores": spec_to_payload(
            options.normalized_class_scores
        ),
        "expected_output": spec_to_payload(options.expected_output),
        "predicted_output": spec_to_payload(options.predicted_output),
        "predicted_scores": spec_to_payload(options.predicted_scores),
        "final_output_species": spec_to_payload(instance.final_output_species),
        "inactive_output_species": instance.inactive_output_species,
    }


def _plot_names(instance: CRNInstance, options: CRNExportOptions) -> list[str]:
    if options.plots is not None:
        return [safe_crn_identifier(str(name)) for name in options.plots]

    active_z = [
        species
        for species in instance.initials
        if species.endswith("Z_free") or re.search(r"(^|_)Z_\d+$", species)
    ]

    if active_z:
        return active_z

    return list(instance.initials)[: min(8, len(instance.initials))]


def _format_number(value: Any, options: CRNExportOptions) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CRNExportError(f"number formatting: expected a finite number, got {value!r}.") from exc

    if number != number or number in (float("inf"), float("-inf")):
        raise CRNExportError(f"number formatting: expected a finite number, got {value!r}.")

    return format(number, options.numeric_format)


def _format_side(species: Sequence[str]) -> str:
    return " + ".join(species)


def _json_compact(value: Any) -> str:
    return json.dumps(spec_to_payload(value), ensure_ascii=False, separators=(",", ":"))


__all__ = [
    "CRNExportError",
    "CRNExportOptions",
    "CRNExportResult",
    "export_visual_dsd_crn",
    "write_visual_dsd_crn",
]
