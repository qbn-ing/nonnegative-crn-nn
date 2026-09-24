from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from experiments.common.artifacts import (
    LoadedRun,
    load_completed_run,
)
from experiments.common.crn import (
    adaptive_pysb_validation,
    compile_crn_artifacts,
    network_output_state,
    reference_case_from_state,
)
from reporting import write_csv_rows
from utils.serialization import write_json


def run_validation(
    run_dirs: Iterable[str | Path],
    output_root: str | Path,
    *,
    st003390_csv: str | Path = Path("data/st003390/st003390_m1m2.csv"),
    chinese_mnist_data_root: str | Path = Path("data/chinese_mnist"),
    device: torch.device | str = "cpu",
    max_samples_per_run: int = 4,
    samples_per_class: int | None = None,
    initial_t_end: float = 20.0,
    max_t_end: float = 320.0,
    n_timepoints: int = 201,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    tail_atol: float = 1e-6,
    tail_rtol: float = 1e-6,
    skip_pysb: bool = False,
    fail_fast: bool = False,
) -> dict[str, Any]:
    if max_samples_per_run <= 0:
        raise ValueError("max_samples_per_run must be positive.")
    if samples_per_class is not None and samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive.")
    if not 0.0 < initial_t_end <= max_t_end:
        raise ValueError("require 0 < initial_t_end <= max_t_end.")
    destination = Path(output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    run_summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for source in run_dirs:
        try:
            loaded = load_completed_run(
                source,
                st003390_csv=st003390_csv,
                chinese_mnist_data_root=chinese_mnist_data_root,
                device=device,
            )
            summary = _validate_loaded_run(
                loaded,
                destination / loaded.run_id,
                device=device,
                max_samples=max_samples_per_run,
                samples_per_class=samples_per_class,
                initial_t_end=initial_t_end,
                max_t_end=max_t_end,
                n_timepoints=n_timepoints,
                atol=atol,
                rtol=rtol,
                tail_atol=tail_atol,
                tail_rtol=tail_rtol,
                skip_pysb=skip_pysb,
                fail_fast=fail_fast,
            )
            run_summaries.append(summary)
            if not summary["passed"] and fail_fast:
                raise RuntimeError(
                    f"numerical CRN validation failed for {loaded.run_id}."
                )
        except Exception as exc:
            failures.append(
                {
                    "run_dir": str(Path(source).resolve()),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            if fail_fast:
                raise
    payload = {
        "schema_version": 2,
        "completed_runs": len(run_summaries),
        "failed_runs": len(failures),
        "validation_failed_runs": sum(
            not bool(summary["passed"]) for summary in run_summaries
        ),
        "skip_pysb": skip_pysb,
        "runs": run_summaries,
        "failures": failures,
    }
    write_json(payload, destination / "summary.json")
    write_csv_rows(failures, destination / "failures.csv")
    return payload


@torch.no_grad()
def _validate_loaded_run(
    loaded: LoadedRun,
    output_dir: Path,
    *,
    device: torch.device | str,
    max_samples: int,
    samples_per_class: int | None,
    initial_t_end: float,
    max_t_end: float,
    n_timepoints: int,
    atol: float,
    rtol: float,
    tail_atol: float,
    tail_rtol: float,
    skip_pysb: bool,
    fail_fast: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded.model.to(device).eval()
    selected = _select_samples(
        loaded,
        device=device,
        maximum=max_samples,
        samples_per_class=samples_per_class,
    )
    model_spec = loaded.model.export_model_spec()
    structural_counts, _ = compile_crn_artifacts(loaded.model, loaded.network)
    cases: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for selected_case in selected:
        position = int(selected_case["position"])
        x = loaded.x_test[position : position + 1].to(device)
        target = loaded.y_test[position : position + 1].to(device)
        state = loaded.model(x)
        network_state = network_output_state(state)
        crn_input = loaded.crn_inputs(x)[0]
        reference = reference_case_from_state(
            state,
            crn_input=crn_input,
            target=int(target.item()),
            sample_index=int(loaded.sample_indices[position]),
        )
        reference_r = network_state.final_rational_state.r.detach().cpu().numpy()
        reference_Z = network_state.Z.detach().cpu().numpy()
        reference_Z_free = network_state.Z_free[:, 0].detach().cpu().numpy()
        reference_active = (
            network_state.output.active_output_fraction[:, 0]
            .detach()
            .cpu()
            .numpy()
        )
        reference_scores = network_state.normalized_scores().detach().cpu().numpy()
        sample_index = int(loaded.sample_indices[position])
        case_id = (
            f"{loaded.run_id}__{selected_case['category']}__i{sample_index}"
        )
        adaptive: dict[str, Any] | None = None
        error: dict[str, str] | None = None
        if not skip_pysb:
            try:
                adaptive = adaptive_pysb_validation(
                    model_spec,
                    reference,
                    initial_t_end=initial_t_end,
                    max_t_end=max_t_end,
                    n_timepoints=n_timepoints,
                    atol=atol,
                    rtol=rtol,
                    tail_atol=tail_atol,
                    tail_rtol=tail_rtol,
                )
                if not adaptive["passed"] and fail_fast:
                    raise RuntimeError(
                        f"PySB validation did not converge within tolerance: {case_id}"
                    )
            except Exception as exc:
                error = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                if fail_fast:
                    raise
        final_time = (
            initial_t_end
            if adaptive is None
            else float(adaptive["final_t_end"])
        )
        crn_path = output_dir / f"{case_id}.crn"
        _, export = compile_crn_artifacts(
            loaded.model,
            loaded.network,
            reference=reference,
            output_path=crn_path,
            final_time=final_time,
            extra_comments=(
                f"case_id: {case_id}",
                f"selection_category: {selected_case['category']}",
            ),
        )
        assert export is not None
        case_payload = {
            "case_id": case_id,
            "selection": selected_case,
            "sample_index": sample_index,
            "target": int(target.item()),
            "reference_prediction": int(network_state.predicted_classes[0].item()),
            "reference_r": reference_r[0].tolist(),
            "reference_Z": reference_Z[0].tolist(),
            "reference_Z_free": float(reference_Z_free[0]),
            "reference_normalized_scores": reference_scores[0].tolist(),
            "pysb": adaptive,
            "pysb_error": error,
            "visual_dsd_crn": crn_path.name,
            "species_count": export["species_count"],
            "reaction_count": export["reaction_count"],
            "parameter_count": export["parameter_count"],
        }
        cases.append(case_payload)
        manifest.append(
            {
                "case_id": case_id,
                "run_id": loaded.run_id,
                "kind": loaded.kind,
                "selection_category": selected_case["category"],
                "sample_index": sample_index,
                "target": int(target.item()),
                "reference_prediction": int(network_state.predicted_classes[0].item()),
                "reference_r_json": json.dumps(reference_r[0].tolist(), separators=(",", ":")),
                "reference_Z_json": json.dumps(reference_Z[0].tolist(), separators=(",", ":")),
                "reference_Z_free": float(reference_Z_free[0]),
                "reference_scores_json": json.dumps(reference_scores[0].tolist(), separators=(",", ":")),
                "crn_file": crn_path.name,
                "final_time": final_time,
            }
        )
    write_csv_rows(manifest, output_dir / "visual_dsd_manifest.csv")
    payload = {
        "schema_version": 2,
        "run_id": loaded.run_id,
        "kind": loaded.kind,
        "depth": loaded.depth,
        "dataset_fingerprint": loaded.dataset_fingerprint,
        "n_cases": len(cases),
        "selection_strategy": (
            "target_balanced_margin_quantiles"
            if samples_per_class is not None
            else "correctness_margin_coverage"
        ),
        "samples_per_class": samples_per_class,
        "target_counts": {
            str(target): sum(case["target"] == target for case in cases)
            for target in sorted({int(case["target"]) for case in cases})
        },
        "n_pysb_passed": sum(
            case["pysb"] is not None and case["pysb"].get("passed", False)
            for case in cases
        ),
        "n_pysb_errors": sum(case["pysb_error"] is not None for case in cases),
        "structural_counts": {
            "species_count": structural_counts["species_count"],
            "reaction_count": structural_counts["reaction_count"],
            "parameter_count": structural_counts["parameter_count"],
        },
        "cases": cases,
    }
    passed = bool(
        skip_pysb
        or (
            payload["n_pysb_passed"] == payload["n_cases"]
            and payload["n_pysb_errors"] == 0
        )
    )
    payload["passed"] = passed
    write_json(payload, output_dir / "validation.json")
    return {
        "run_id": loaded.run_id,
        "kind": loaded.kind,
        "depth": loaded.depth,
        "n_cases": len(cases),
        "selection_strategy": payload["selection_strategy"],
        "samples_per_class": payload["samples_per_class"],
        "target_counts": payload["target_counts"],
        "n_pysb_passed": payload["n_pysb_passed"],
        "n_pysb_errors": payload["n_pysb_errors"],
        "passed": passed,
        **payload["structural_counts"],
    }


def _select_samples(
    loaded: LoadedRun,
    *,
    device: torch.device | str,
    maximum: int,
    samples_per_class: int | None = None,
) -> list[dict[str, Any]]:
    probabilities: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    loaded.model.eval()
    for start in range(0, len(loaded.x_test), 4_096):
        state = loaded.model(loaded.x_test[start : start + 4_096].to(device))
        network_state = network_output_state(state)
        probabilities.append(network_state.normalized_scores().detach().cpu())
        predictions.append(network_state.predicted_classes.detach().cpu())
    scores = torch.cat(probabilities)
    predicted = torch.cat(predictions)
    labels = loaded.y_test.detach().cpu()
    top = torch.topk(scores, k=2, dim=1).values
    margins = top[:, 0] - top[:, 1]
    correct = predicted == labels
    if samples_per_class is not None:
        return _select_target_balanced_samples(
            labels=labels,
            predicted=predicted,
            margins=margins,
            correct=correct,
            maximum=maximum,
            samples_per_class=samples_per_class,
        )
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    categories = (
        ("correct_far", correct, True),
        ("correct_near", correct, False),
        ("error_far", ~correct, True),
        ("error_near", ~correct, False),
    )
    for category, mask, take_maximum in categories:
        positions = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if positions.numel() == 0:
            continue
        order = torch.argsort(margins[positions], descending=take_maximum)
        position = next(
            (
                int(positions[index].item())
                for index in order.tolist()
                if int(positions[index].item()) not in used
            ),
            None,
        )
        if position is None:
            continue
        used.add(position)
        selected.append(
            {
                "category": category,
                "position": position,
                "margin": float(margins[position].item()),
                "correct": bool(correct[position].item()),
                "target": int(labels[position].item()),
                "prediction": int(predicted[position].item()),
            }
        )
        if len(selected) >= maximum:
            break
    if len(selected) < maximum:
        remaining = torch.argsort(margins)
        for position_tensor in remaining:
            position = int(position_tensor.item())
            if position in used:
                continue
            used.add(position)
            selected.append(
                {
                    "category": "margin_coverage",
                    "position": position,
                    "margin": float(margins[position].item()),
                    "correct": bool(correct[position].item()),
                    "target": int(labels[position].item()),
                    "prediction": int(predicted[position].item()),
                }
            )
            if len(selected) >= maximum:
                break
    return selected


def _select_target_balanced_samples(
    *,
    labels: torch.Tensor,
    predicted: torch.Tensor,
    margins: torch.Tensor,
    correct: torch.Tensor,
    maximum: int,
    samples_per_class: int,
) -> list[dict[str, Any]]:
    """Select equal target-label quotas with deterministic margin coverage."""

    unique_labels = tuple(sorted(int(item) for item in torch.unique(labels).tolist()))
    if not unique_labels:
        raise ValueError("target-balanced PySB sampling requires non-empty labels.")
    required = len(unique_labels) * samples_per_class
    if required > maximum:
        raise ValueError(
            f"target-balanced PySB sampling needs {required} cases for "
            f"{len(unique_labels)} classes, but maximum is {maximum}."
        )
    selected: list[dict[str, Any]] = []
    for target in unique_labels:
        positions = torch.nonzero(labels == target, as_tuple=False).reshape(-1)
        if positions.numel() < samples_per_class:
            raise ValueError(
                f"target {target} has only {positions.numel()} test samples; "
                f"need {samples_per_class}."
            )
        ordered = positions[torch.argsort(margins[positions])]
        if samples_per_class == 1:
            ranks = torch.tensor(
                [(ordered.numel() - 1) // 2],
                dtype=torch.int64,
            )
        else:
            ranks = torch.linspace(
                0,
                ordered.numel() - 1,
                steps=samples_per_class,
                dtype=torch.float64,
            ).round().to(torch.int64)
        for quantile_index, rank in enumerate(ranks.tolist()):
            position = int(ordered[rank].item())
            selected.append(
                {
                    "category": f"target_{target}_margin_q{quantile_index:02d}",
                    "position": position,
                    "margin": float(margins[position].item()),
                    "correct": bool(correct[position].item()),
                    "target": int(labels[position].item()),
                    "prediction": int(predicted[position].item()),
                }
            )
    return selected


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate selected trained checkpoints against PySB and export Visual DSD CRNs."
    )
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--st003390-csv", type=Path, default=Path("data/st003390/st003390_m1m2.csv"))
    parser.add_argument(
        "--chinese-mnist-data-root",
        type=Path,
        default=Path("data/chinese_mnist"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples-per-run", type=int, default=4)
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=None,
        help=(
            "Require this many cases from every target class in each run; "
            "one case uses the median margin; larger quotas span the "
            "within-class decision-margin range."
        ),
    )
    parser.add_argument("--initial-t-end", type=float, default=20.0)
    parser.add_argument("--max-t-end", type=float, default=320.0)
    parser.add_argument("--n-timepoints", type=int, default=201)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--tail-atol", type=float, default=1e-6)
    parser.add_argument("--tail-rtol", type=float, default=1e-6)
    parser.add_argument("--skip-pysb", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    summary = run_validation(
        args.run_dir,
        args.output_root,
        st003390_csv=args.st003390_csv,
        chinese_mnist_data_root=args.chinese_mnist_data_root,
        device=args.device,
        max_samples_per_run=args.max_samples_per_run,
        samples_per_class=args.samples_per_class,
        initial_t_end=args.initial_t_end,
        max_t_end=args.max_t_end,
        n_timepoints=args.n_timepoints,
        atol=args.atol,
        rtol=args.rtol,
        tail_atol=args.tail_atol,
        tail_rtol=args.tail_rtol,
        skip_pysb=args.skip_pysb,
        fail_fast=args.fail_fast,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if summary["failed_runs"] or summary["validation_failed_runs"] else 0


__all__ = ["main", "run_validation"]
