from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

from experiments.common import (
    execute_run_plan,
    is_completed,
    parse_csv_ints,
    parse_csv_strings,
    resolve_device,
    select_runs,
    write_execution_summary,
)
from utils.environment import save_environment_info
from utils.serialization import write_json

from .analyze_results import analyze_results
from .data import make_writer_split, validate_all_outer_folds
from .data_prepare import DEFAULT_DATA_ROOT, validate_processed
from .protocol import (
    CONDITIONS,
    PROFILES,
    RunSpec,
    build_run_plan,
    get_profile,
    override_profile,
)
from .training import run_one


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 15-class Chinese-MNIST depth and I-by-C experiments."
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--folds", type=str, default=None)
    parser.add_argument("--depths", type=str, default=None)
    parser.add_argument("--conditions", type=str, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-root", type=Path, default=Path("runs") / "chinese_mnist"
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-indices", type=str, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--skip-completed", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-crn", action="store_true")
    parser.add_argument(
        "--analyze-after-run",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive.")
    profile = override_profile(
        get_profile(args.profile),
        folds=parse_csv_ints(args.folds),
        depths=parse_csv_ints(args.depths),
        conditions=parse_csv_strings(args.conditions),
        total_steps=args.total_steps,
    )
    plan = build_run_plan(profile)
    selected = select_runs(
        plan,
        indices=parse_csv_ints(args.run_indices),
        max_runs=args.max_runs,
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(profile.to_dict(), output_root / "profile.json")
    write_json(
        {
            "schema_version": 2,
            "profile": profile.name,
            "available_profiles": tuple(PROFILES),
            "available_conditions": CONDITIONS,
            "total_runs": len(plan),
            "selected_runs": len(selected),
            "selected_run_ids": [spec.run_id for _, spec in selected],
            "dataset_classes": 15,
            "writer_disjoint": True,
            "depth_definition": (
                "D counts all E/I-to-r layers including terminal K=15 "
                "class evidence; competitive O is excluded"
            ),
            "runs": [
                {"index": index, **spec.to_dict()}
                for index, spec in enumerate(plan)
            ],
        },
        output_root / "run_plan.json",
    )
    for index, spec in selected:
        print(
            f"{index:04d} {spec.run_id} stages="
            f"{list(zip(spec.optimizer_stage_depths, spec.stage_steps))}",
            flush=True,
        )
    if args.dry_run:
        return 0

    validate_processed(args.data_root, profile.image_size)
    selected_specs = tuple(spec for _, spec in selected)
    by_fold: dict[int, RunSpec] = {}
    for spec in selected_specs:
        by_fold.setdefault(spec.fold, spec)
    for spec in by_fold.values():
        make_writer_split(args.data_root, spec)
    validate_all_outer_folds(args.data_root, selected_specs)

    device = resolve_device(args.device)
    save_environment_info(
        output_root / "environment.json",
        project_root=Path(__file__).resolve().parents[2],
        include_packages=False,
        extra={
            "device_argument": str(device),
            "deterministic": args.deterministic,
            "workers": args.workers,
            "data_root": args.data_root.resolve().as_posix(),
        },
    )
    pending: list[tuple[int, RunSpec]] = []
    skipped = 0
    for item in selected:
        if args.skip_completed and is_completed(
            output_root / item[1].run_id,
            required_artifacts=(
                "resolved_protocol.json",
                "metrics.json",
                "predictions_test.npz",
            ),
        ):
            skipped += 1
        else:
            pending.append(item)
    result = execute_run_plan(
        pending,
        worker=_worker,
        worker_kwargs={
            "output_root": output_root.as_posix(),
            "data_root": args.data_root.resolve().as_posix(),
            "device": str(device),
            "deterministic": args.deterministic,
            "save_crn": args.save_crn,
            "overwrite": args.overwrite,
            "resume": args.resume,
        },
        workers=args.workers,
        fail_fast=args.fail_fast,
    )
    write_execution_summary(
        output_root,
        selected=len(selected),
        skipped=skipped,
        result=result,
    )
    if args.analyze_after_run and (result.completed or skipped):
        analyze_results(output_root)
    return 1 if result.failures else 0


def _worker(
    spec_payload: dict[str, Any],
    *,
    output_root: str,
    data_root: str,
    device: str,
    deterministic: bool,
    save_crn: bool,
    overwrite: bool,
    resume: bool,
) -> dict[str, float]:
    spec = RunSpec.from_dict(spec_payload)
    resume_existing = resume and (
        Path(output_root) / spec.run_id / "resolved_protocol.json"
    ).is_file()
    result = run_one(
        spec,
        output_root,
        data_root=data_root,
        device=device,
        deterministic=deterministic,
        save_crn=save_crn,
        overwrite=overwrite,
        resume=resume_existing,
        progress=False,
    )
    return {
        "test_bacc": float(result.metrics["test"]["recall_macro"]),
        "elapsed_seconds": result.elapsed_seconds,
    }


__all__ = ["build_parser", "main"]
