from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

from experiments.common import (
    execute_run_plan,
    is_completed,
    parse_csv_ints,
    resolve_device,
    select_runs,
    write_execution_summary,
)
from utils.environment import save_environment_info
from utils.serialization import write_json

from .analyze_results import analyze_results
from .protocol import (
    DATASETS,
    PROFILES,
    REPRESENTATIONS,
    RunSpec,
    build_run_plan,
    get_profile,
    override_profile,
    parse_csv_values,
)
from .training import run_one


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run D1 raw/quadratic single-layer experiments."
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--representations", type=str, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--outer-folds", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--circle-n-samples", type=int, default=None)
    parser.add_argument("--circle-factor", type=float, default=None)
    parser.add_argument("--circle-noise", type=float, default=None)
    parser.add_argument("--circle-seed", type=int, default=None)
    parser.add_argument(
        "--st003390-csv",
        type=Path,
        default=Path("data") / "st003390" / "st003390_m1m2.csv",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("runs") / "single_layer"
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--run-indices", type=str, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-crn", action="store_true")
    parser.add_argument("--pysb-samples", type=int, default=0)
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
        datasets=parse_csv_values(args.datasets),
        representations=parse_csv_values(args.representations),
        repeats=args.repeats,
        outer_folds=args.outer_folds,
        total_steps=args.total_steps,
        circle_n_samples=args.circle_n_samples,
        circle_factor=args.circle_factor,
        circle_noise=args.circle_noise,
        circle_seed=args.circle_seed,
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
            "available_datasets": DATASETS,
            "available_representations": REPRESENTATIONS,
            "total_runs": len(plan),
            "selected_runs": len(selected),
            "selected_run_ids": [spec.run_id for _, spec in selected],
            "depth_definition": (
                "D=1 is the terminal class-evidence E/I-to-r layer; "
                "competitive O is excluded"
            ),
            "circle_parameters_frozen": profile.circle.parameters_frozen,
            "runs": [
                {"index": index, **spec.to_dict()}
                for index, spec in enumerate(plan)
            ],
        },
        output_root / "run_plan.json",
    )
    for index, spec in selected:
        print(
            f"{index:04d} {spec.run_id} dataset={spec.dataset} "
            f"basis={spec.representation} D=1",
            flush=True,
        )
    if args.dry_run:
        return 0

    selected_datasets = {spec.dataset for _, spec in selected}
    if "st003390_m1m2" in selected_datasets and not args.st003390_csv.is_file():
        raise FileNotFoundError(
            f"{args.st003390_csv} is missing. Run "
            "python experiments/single_layer/prepare_st003390.py first."
        )
    device = resolve_device(args.device)
    save_environment_info(
        output_root / "environment.json",
        project_root=Path(__file__).resolve().parents[2],
        include_packages=False,
        extra={
            "device_argument": str(device),
            "deterministic": args.deterministic,
            "workers": args.workers,
            "pysb_samples": args.pysb_samples,
        },
    )
    pending: list[tuple[int, RunSpec]] = []
    skipped = 0
    for item in selected:
        if args.skip_completed and is_completed(output_root / item[1].run_id):
            skipped += 1
        else:
            pending.append(item)
    result = execute_run_plan(
        pending,
        worker=_worker,
        worker_kwargs={
            "output_root": output_root.as_posix(),
            "st003390_csv": args.st003390_csv.resolve().as_posix(),
            "device": str(device),
            "deterministic": args.deterministic,
            "save_crn": args.save_crn,
            "pysb_samples": args.pysb_samples,
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
    st003390_csv: str,
    device: str,
    deterministic: bool,
    save_crn: bool,
    pysb_samples: int,
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
        st003390_csv=st003390_csv,
        device=device,
        deterministic=deterministic,
        save_crn=save_crn,
        pysb_samples=pysb_samples,
        overwrite=overwrite,
        resume=resume_existing,
        progress=False,
    )
    return {
        "test_bacc": float(result.metrics["test"]["recall_macro"]),
        "elapsed_seconds": result.elapsed_seconds,
    }


__all__ = ["build_parser", "main"]
