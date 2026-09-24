from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.serialization import write_json
from reporting import write_csv_rows


def compare_visual_dsd_results(
    validation_root: str | Path,
    results_csv: str | Path,
    output_dir: str | Path,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    allow_missing: bool = False,
) -> dict[str, Any]:
    manifests = _load_manifests(Path(validation_root).resolve())
    observed = _read_csv(Path(results_csv).resolve())
    by_case = {row["case_id"]: row for row in observed}
    duplicates = len(by_case) != len(observed)
    if duplicates:
        raise ValueError("Visual DSD results contain duplicate case_id values.")
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for case_id, reference in sorted(manifests.items()):
        result = by_case.get(case_id)
        if result is None:
            missing.append(case_id)
            continue
        rows.append(_compare_case(reference, result, atol=atol, rtol=rtol))
    unexpected = sorted(set(by_case) - set(manifests))
    if (missing or unexpected) and not allow_missing:
        raise ValueError(
            f"Visual DSD case mismatch: missing={missing}, unexpected={unexpected}."
        )
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    write_csv_rows(rows, destination / "visual_dsd_comparison.csv")
    passed_cases = sum(row["overall_within_tolerance"] for row in rows)
    payload = {
        "schema_version": 2,
        "expected_cases": len(manifests),
        "compared_cases": len(rows),
        "passed_cases": passed_cases,
        "failed_cases": len(rows) - passed_cases,
        "missing_case_ids": missing,
        "unexpected_case_ids": unexpected,
        "atol": atol,
        "rtol": rtol,
        "max_abs_r_error": _optional_max(rows, "max_abs_r_error"),
        "max_abs_Z_error": _optional_max(rows, "max_abs_Z_error"),
        "max_abs_Z_free_error": _optional_max(rows, "abs_Z_free_error"),
        "max_abs_normalized_score_error": _optional_max(
            rows,
            "max_abs_normalized_score_error",
        ),
    }
    write_json(payload, destination / "summary.json")
    return payload


def _load_manifests(root: Path) -> dict[str, dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob("*/visual_dsd_manifest.csv")):
        rows.extend(_read_csv(path))
    if not rows:
        raise FileNotFoundError(
            f"no visual_dsd_manifest.csv files found beneath {root}"
        )
    result = {row["case_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("validation manifests contain duplicate case IDs.")
    return result


def _compare_case(
    reference: dict[str, str],
    observed: dict[str, str],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    expected_r = np.asarray(json.loads(reference["reference_r_json"]), dtype=float)
    expected_Z = np.asarray(json.loads(reference["reference_Z_json"]), dtype=float)
    expected_scores = np.asarray(
        json.loads(reference["reference_scores_json"]),
        dtype=float,
    )
    width = len(expected_Z)
    actual_r = np.asarray(
        [_finite(observed[f"R_{index}"]) for index in range(width)],
        dtype=float,
    )
    actual_Z = np.asarray(
        [_finite(observed[f"Z_{index}"]) for index in range(width)],
        dtype=float,
    )
    actual_Z_free = _finite(observed["Z_free"])
    if np.any(actual_r < 0.0) or np.any(actual_Z < 0.0) or actual_Z_free < 0.0:
        raise ValueError(f"case {reference['case_id']} contains negative concentration.")
    actual_scores = actual_Z / max(float(actual_Z.sum()), 1e-12)
    expected_Z_free = float(reference["reference_Z_free"])
    r_ok = bool(np.allclose(actual_r, expected_r, atol=atol, rtol=rtol))
    Z_ok = bool(np.allclose(actual_Z, expected_Z, atol=atol, rtol=rtol))
    free_ok = bool(np.isclose(actual_Z_free, expected_Z_free, atol=atol, rtol=rtol))
    score_ok = bool(
        np.allclose(actual_scores, expected_scores, atol=atol, rtol=rtol)
    )
    pool_total = float(actual_Z.sum() + actual_Z_free)
    pool_ok = bool(np.isclose(pool_total, 1.0, atol=atol, rtol=rtol))
    return {
        "case_id": reference["case_id"],
        "run_id": reference["run_id"],
        "selection_category": reference["selection_category"],
        "sample_index": int(reference["sample_index"]),
        "max_abs_r_error": float(np.max(np.abs(actual_r - expected_r))),
        "max_abs_Z_error": float(np.max(np.abs(actual_Z - expected_Z))),
        "abs_Z_free_error": abs(actual_Z_free - expected_Z_free),
        "max_abs_normalized_score_error": float(
            np.max(np.abs(actual_scores - expected_scores))
        ),
        "output_pool_total": pool_total,
        "response_within_tolerance": r_ok,
        "Z_within_tolerance": Z_ok,
        "Z_free_within_tolerance": free_ok,
        "normalized_scores_within_tolerance": score_ok,
        "output_pool_conserved": pool_ok,
        "prediction_match": int(np.argmax(actual_scores))
        == int(reference["reference_prediction"]),
        "overall_within_tolerance": r_ok and Z_ok and free_ok and score_ok and pool_ok,
        "reported_final_time": observed.get("final_time"),
        "reported_tail_max_abs_delta": observed.get("tail_max_abs_delta"),
        "reported_tail_max_rel_delta": observed.get("tail_max_rel_delta"),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"expected finite concentration, got {value!r}.")
    return number


def _optional_max(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return max(values) if values else None


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare imported Visual DSD concentrations.")
    parser.add_argument("validation_root", type=Path)
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)
    result = compare_visual_dsd_results(
        args.validation_root,
        args.results_csv,
        args.output_dir,
        atol=args.atol,
        rtol=args.rtol,
        allow_missing=args.allow_missing,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 1 if result["failed_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["compare_visual_dsd_results", "main"]
