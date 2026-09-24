from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments.common import (
    aggregate_metric,
    exact_sign_test_pvalue,
    planned_run_directories,
)
from reporting import write_csv_rows
from utils.serialization import write_json


METRICS: tuple[str, ...] = (
    "test_accuracy",
    "test_bacc",
    "test_f1_macro",
    "test_auc",
    "test_aupr",
    "test_loss",
)


def analyze_results(
    output_root: str | Path,
    *,
    analysis_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    destination = (
        Path(analysis_dir).resolve()
        if analysis_dir is not None
        else root / "analysis"
    )
    destination.mkdir(parents=True, exist_ok=True)
    rows, issues = _load_rows(root)
    aggregates = _aggregate_rows(rows)
    differences, paired = _paired_representation_effects(rows)
    write_csv_rows(rows, destination / "runs.csv")
    write_csv_rows(aggregates, destination / "aggregate.csv")
    write_csv_rows(differences, destination / "paired_differences.csv")
    write_csv_rows(paired, destination / "paired_summary.csv")
    write_csv_rows(issues, destination / "integrity_issues.csv")
    summary = {
        "completed_runs": len(rows),
        "aggregate_rows": len(aggregates),
        "paired_difference_rows": len(differences),
        "paired_summary_rows": len(paired),
        "integrity_issue_count": len(issues),
    }
    write_json(summary, destination / "summary.json")
    (destination / "report.md").write_text(
        _render_report(summary, paired, issues),
        encoding="utf-8",
    )
    return summary


def _load_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    run_dirs, plan_issues = planned_run_directories(root)
    issues.extend(plan_issues)
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            protocol = json.loads(
                (run_dir / "resolved_protocol.json").read_text(encoding="utf-8")
            )
            status = json.loads(
                (run_dir / "run_status.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(_issue(run_dir.name, "unreadable_artifact", str(exc)))
            continue
        if status.get("status") != "completed":
            issues.append(
                _issue(
                    run_dir.name,
                    "metrics_without_completed_status",
                    str(status.get("status")),
                )
            )
            continue
        try:
            rows.append(_row_from_artifacts(run_dir, protocol, metrics))
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(_issue(run_dir.name, "invalid_schema", str(exc)))

    by_pair: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[(row["dataset"], row["repeat"], row["fold"])].append(row)
    for cell, cell_rows in sorted(by_pair.items()):
        representations = [row["representation"] for row in cell_rows]
        if len(representations) != len(set(representations)):
            issues.append(
                _issue("|".join(map(str, cell)), "duplicate_representation", "")
            )
        fingerprints = {row["dataset_fingerprint"] for row in cell_rows}
        if len(fingerprints) != 1:
            issues.append(
                _issue(
                    "|".join(map(str, cell)),
                    "paired_dataset_fingerprint_mismatch",
                    str(sorted(fingerprints)),
                )
            )
        seeds = {row["model_seed"] for row in cell_rows}
        if len(seeds) != 1:
            issues.append(
                _issue(
                    "|".join(map(str, cell)),
                    "paired_model_seed_mismatch",
                    str(sorted(seeds)),
                )
            )
    return rows, issues


def _row_from_artifacts(
    run_dir: Path,
    protocol: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    test = _mapping(metrics, "test")
    crn = _mapping(metrics, "crn_counts")
    return {
        "run_id": str(protocol.get("run_id", run_dir.name)),
        "profile": _required(protocol, "profile_name"),
        "dataset": _required(protocol, "dataset"),
        "representation": _required(protocol, "representation"),
        "depth": _required(protocol, "depth"),
        "repeat": _required(protocol, "repeat"),
        "fold": _required(protocol, "fold"),
        "split_seed": _required(protocol, "split_seed"),
        "model_seed": _required(protocol, "model_seed"),
        "dataset_fingerprint": _required(metrics, "dataset_fingerprint"),
        "raw_input_dim": _required(metrics, "raw_input_dim"),
        "basis_dim": _required(metrics, "basis_dim"),
        "n_classes": _required(metrics, "n_classes"),
        "total_optimizer_steps": _required(
            metrics, "total_optimizer_steps"
        ),
        "best_global_step": _required(metrics, "best_global_step"),
        "best_validation_loss": _required(metrics, "best_validation_loss"),
        "test_accuracy": _number(test.get("accuracy", test.get("acc"))),
        "test_bacc": _number(test.get("recall_macro")),
        "test_f1_macro": _number(test.get("f1_macro")),
        "test_auc": _number(test.get("auc_macro")),
        "test_aupr": _number(test.get("aupr_macro")),
        "test_loss": _number(test.get("classification")),
        "species_count": _required(crn, "species_count"),
        "reaction_count": _required(crn, "reaction_count"),
        "parameter_count": _required(crn, "parameter_count"),
        "static_compile_passed": _required(crn, "static_compile_passed"),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["representation"])].append(row)
    output: list[dict[str, Any]] = []
    for (dataset, representation), group in sorted(groups.items()):
        for metric in METRICS:
            values = [row[metric] for row in group if _finite(row[metric])]
            if not values:
                continue
            output.append(
                {
                    "dataset": dataset,
                    "representation": representation,
                    "metric": metric,
                    **aggregate_metric(values).to_dict(),
                }
            )
    return output


def _paired_representation_effects(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cells: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        cells[(row["dataset"], row["repeat"], row["fold"])][
            row["representation"]
        ] = row
    differences: list[dict[str, Any]] = []
    for (dataset, repeat, fold), pair in sorted(cells.items()):
        if not {"raw", "quadratic"}.issubset(pair):
            continue
        for metric in METRICS:
            raw = pair["raw"][metric]
            quadratic = pair["quadratic"][metric]
            if not (_finite(raw) and _finite(quadratic)):
                continue
            differences.append(
                {
                    "dataset": dataset,
                    "repeat": repeat,
                    "fold": fold,
                    "model_seed": pair["raw"]["model_seed"],
                    "metric": metric,
                    "raw": raw,
                    "quadratic": quadratic,
                    "difference": float(quadratic) - float(raw),
                    "effect": "quadratic_minus_raw",
                }
            )
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in differences:
        groups[(row["dataset"], row["metric"])].append(row["difference"])
    summary: list[dict[str, Any]] = []
    for (dataset, metric), values in sorted(groups.items()):
        aggregate = aggregate_metric(values).to_dict()
        wins = sum(value > 0.0 for value in values)
        losses = sum(value < 0.0 for value in values)
        ties = len(values) - wins - losses
        summary.append(
            {
                "dataset": dataset,
                "metric": metric,
                "effect": "quadratic_minus_raw",
                **aggregate,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "win_rate_excluding_ties": (
                    wins / (wins + losses) if wins + losses else math.nan
                ),
                "exact_sign_p": (
                    exact_sign_test_pvalue(wins, losses)
                    if wins + losses
                    else math.nan
                ),
            }
        )
    return differences, summary


def _render_report(
    summary: dict[str, Any],
    paired: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> str:
    lines = [
        "# 单层 raw / quadratic 实验汇总",
        "",
        f"- 完成运行：{summary['completed_runs']}",
        f"- 完整性问题：{summary['integrity_issue_count']}",
        "- 配对效应统一定义为 quadratic − raw。",
        "",
        "| 数据集 | 指标 | n | 均值差 | 标准差 | 胜/平/负 | sign p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        if row["metric"] not in {"test_bacc", "test_auc", "test_accuracy"}:
            continue
        lines.append(
            "| {dataset} | {metric} | {n} | {mean:.6f} | {std:.6f} | "
            "{wins}/{ties}/{losses} | {p:.6g} |".format(
                dataset=row["dataset"],
                metric=row["metric"],
                n=row["n"],
                mean=row["mean"],
                std=row["standard_deviation"],
                wins=row["wins"],
                ties=row["ties"],
                losses=row["losses"],
                p=row["exact_sign_p"],
            )
        )
    if issues:
        lines.extend(
            [
                "",
                "结果不完整；具体条目见 `integrity_issues.csv`。",
            ]
        )
    return "\n".join(lines) + "\n"


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = _required(payload, key)
    if not isinstance(value, dict):
        raise TypeError(f"{key!r} must be a JSON object.")
    return value


def _required(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise KeyError(f"required field {key!r} is missing")
    return payload[key]


def _number(value: Any) -> float:
    if value is None:
        return math.nan
    return float(value)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _issue(run_id: str, issue: str, detail: str) -> dict[str, str]:
    return {"run_id": run_id, "issue": issue, "detail": detail}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze D1 single-layer runs.")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--analysis-dir", type=Path, default=None)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = analyze_results(
        args.output_root,
        analysis_dir=args.analysis_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["METRICS", "analyze_results", "build_parser", "main"]
