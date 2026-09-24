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
    "test_loss",
    "grid_accuracy",
    "edge_recall",
    "edge_precision",
)
EFFECTS: dict[str, tuple[dict[str, float], str]] = {
    "I_at_C0": (
        {"identity_full_start": 1.0, "dense_full_start": -1.0},
        "I1_C0 - I0_C0",
    ),
    "I_at_C1": (
        {"continuation": 1.0, "dense_continuation": -1.0},
        "I1_C1 - I0_C1",
    ),
    "C_at_I0": (
        {"dense_continuation": 1.0, "dense_full_start": -1.0},
        "I0_C1 - I0_C0",
    ),
    "C_at_I1": (
        {"continuation": 1.0, "identity_full_start": -1.0},
        "I1_C1 - I1_C0",
    ),
    "I_by_C_interaction": (
        {
            "continuation": 1.0,
            "identity_full_start": -1.0,
            "dense_continuation": -1.0,
            "dense_full_start": 1.0,
        },
        "(I1_C1-I0_C1) - (I1_C0-I0_C0)",
    ),
}


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
    differences, paired = _paired_effects(rows)
    capacity_pairs, capacity_summary = _capacity_matched_pairs(rows)
    write_csv_rows(rows, destination / "runs.csv")
    write_csv_rows(aggregates, destination / "aggregate.csv")
    write_csv_rows(differences, destination / "paired_differences.csv")
    write_csv_rows(paired, destination / "paired_summary.csv")
    write_csv_rows(capacity_pairs, destination / "capacity_matched_pairs.csv")
    write_csv_rows(
        capacity_summary, destination / "capacity_matched_summary.csv"
    )
    write_csv_rows(issues, destination / "integrity_issues.csv")
    summary = {
        "completed_runs": len(rows),
        "aggregate_rows": len(aggregates),
        "paired_difference_rows": len(differences),
        "paired_summary_rows": len(paired),
        "capacity_matched_pair_rows": len(capacity_pairs),
        "capacity_matched_summary_rows": len(capacity_summary),
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
            issues.append(
                {
                    "run_id": run_dir.name,
                    "issue": "unreadable_completed_artifact",
                    "detail": str(exc),
                }
            )
            continue
        if status.get("status") != "completed":
            issues.append(
                {
                    "run_id": run_dir.name,
                    "issue": "metrics_without_completed_status",
                    "detail": str(status.get("status")),
                }
            )
            continue
        try:
            row = _row_from_artifacts(run_dir, protocol, metrics)
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(
                {
                    "run_id": run_dir.name,
                    "issue": "invalid_completed_artifact_schema",
                    "detail": str(exc),
                }
            )
            continue
        rows.append(row)

    by_cell: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[
            (row["task"], row["width"], row["depth"], row["seed"])
        ].append(row)
    for cell, cell_rows in sorted(by_cell.items()):
        conditions = [row["condition"] for row in cell_rows]
        if len(conditions) != len(set(conditions)):
            issues.append(
                {
                    "run_id": "|".join(map(str, cell)),
                    "issue": "duplicate_condition",
                    "detail": str(conditions),
                }
            )
        fingerprints = {row["dataset_fingerprint"] for row in cell_rows}
        if len(fingerprints) != 1:
            issues.append(
                {
                    "run_id": "|".join(map(str, cell)),
                    "issue": "paired_dataset_fingerprint_mismatch",
                    "detail": str(sorted(fingerprints)),
                }
            )
    return rows, issues


def _row_from_artifacts(
    run_dir: Path,
    protocol: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Join immutable run configuration with measured result fields.

    The resolved protocol is the authoritative source for task and topology.
    ``metrics.json`` owns selected-checkpoint and evaluation values.  Keeping
    this boundary explicit prevents optional duplication in the metrics file
    from becoming a hard schema dependency during analysis.
    """

    task = _mapping(protocol, "task")
    test = _mapping(metrics, "test")
    grid = _mapping(metrics, "grid")
    crn_counts = _mapping(metrics, "crn_counts")
    return {
        "run_id": str(protocol.get("run_id", metrics.get("run_id", run_dir.name))),
        "task": _required(task, "name"),
        "degree": _required(task, "degree"),
        "input_encoding": _required(task, "input_encoding"),
        "width": _required(protocol, "width"),
        "depth": _required(protocol, "depth"),
        "seed": _required(protocol, "seed"),
        "condition": _required(protocol, "condition"),
        "factor_code": _required(protocol, "factor_code"),
        "dataset_fingerprint": _required(metrics, "dataset_fingerprint"),
        "total_optimizer_steps": _required(metrics, "total_optimizer_steps"),
        "best_global_step": _required(metrics, "best_global_step"),
        "best_validation_loss": _required(metrics, "best_validation_loss"),
        "test_accuracy": _number(test.get("accuracy", test.get("acc"))),
        "test_bacc": _number(test.get("recall_macro")),
        "test_f1_macro": _number(test.get("f1_macro")),
        "test_auc": _number(test.get("auc_macro")),
        "test_loss": _number(test.get("classification")),
        "grid_accuracy": _number(grid.get("accuracy")),
        "edge_recall": _number(grid.get("edge_recall")),
        "edge_precision": _number(grid.get("edge_precision")),
        "species_count": _required(crn_counts, "species_count"),
        "reaction_count": _required(crn_counts, "reaction_count"),
        "parameter_count": _required(crn_counts, "parameter_count"),
        "profile": _required(protocol, "profile_name"),
    }


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = _required(payload, key)
    if not isinstance(value, dict):
        raise TypeError(f"{key!r} must be a JSON object.")
    return value


def _required(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise KeyError(f"required field {key!r} is missing")
    return payload[key]


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (row["task"], row["width"], row["depth"], row["condition"])
        ].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        task, width, depth, condition = key
        for metric in METRICS:
            values = [row[metric] for row in group if _finite(row[metric])]
            if not values:
                continue
            aggregate = aggregate_metric(values)
            output.append(
                {
                    "task": task,
                    "width": width,
                    "depth": depth,
                    "condition": condition,
                    "metric": metric,
                    **aggregate.to_dict(),
                }
            )
    return output


def _paired_effects(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cells: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["task"], row["width"], row["depth"], row["seed"])
        cells[key][row["condition"]] = row
    differences: list[dict[str, Any]] = []
    for key, conditions in sorted(cells.items()):
        task, width, depth, seed = key
        for effect, (weights, expression) in EFFECTS.items():
            if not set(weights).issubset(conditions):
                continue
            for metric in METRICS:
                values = {
                    condition: conditions[condition][metric]
                    for condition in weights
                }
                if not all(_finite(value) for value in values.values()):
                    continue
                difference = sum(
                    weights[condition] * float(values[condition])
                    for condition in weights
                )
                differences.append(
                    {
                        "task": task,
                        "width": width,
                        "depth": depth,
                        "seed": seed,
                        "effect": effect,
                        "expression": expression,
                        "metric": metric,
                        "difference": difference,
                    }
                )

    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    expressions: dict[tuple[Any, ...], str] = {}
    for row in differences:
        key = (
            row["task"],
            row["width"],
            row["depth"],
            row["effect"],
            row["metric"],
        )
        groups[key].append(float(row["difference"]))
        expressions[key] = str(row["expression"])
    summaries: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        task, width, depth, effect, metric = key
        aggregate = aggregate_metric(values)
        wins = sum(value > 0.0 for value in values)
        losses = sum(value < 0.0 for value in values)
        ties = len(values) - wins - losses
        summaries.append(
            {
                "task": task,
                "width": width,
                "depth": depth,
                "effect": effect,
                "expression": expressions[key],
                "metric": metric,
                "n_pairs": len(values),
                "mean_difference": aggregate.mean,
                "median_difference": aggregate.median,
                "q1_difference": aggregate.q1,
                "q3_difference": aggregate.q3,
                "iqr_difference": aggregate.iqr,
                "standard_deviation": aggregate.standard_deviation,
                "standard_error": aggregate.standard_error,
                "ci95_low": aggregate.ci95_low,
                "ci95_high": aggregate.ci95_high,
                "ci95_method": aggregate.ci95_method,
                "minimum_difference": aggregate.minimum,
                "maximum_difference": aggregate.maximum,
                "n_wins": wins,
                "n_losses": losses,
                "n_ties": ties,
                "win_rate_excluding_ties": (
                    None if wins + losses == 0 else wins / (wins + losses)
                ),
                "sign_test_pvalue": (
                    None
                    if wins + losses == 0
                    else exact_sign_test_pvalue(wins, losses)
                ),
            }
        )
    return differences, summaries


def _capacity_matched_pairs(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Match each deep run to the closest lower-depth run in actual parameters."""

    pairs: list[dict[str, Any]] = []
    for target in sorted(rows, key=lambda row: str(row["run_id"])):
        candidates = [
            row
            for row in rows
            if row["task"] == target["task"]
            and row["seed"] == target["seed"]
            and row["condition"] == target["condition"]
            and int(row["depth"]) < int(target["depth"])
        ]
        if not candidates:
            continue
        target_parameters = int(target["parameter_count"])
        baseline = min(
            candidates,
            key=lambda row: (
                abs(
                    math.log(
                        max(1, int(row["parameter_count"]))
                        / max(1, target_parameters)
                    )
                ),
                -int(row["depth"]),
                int(row["width"]),
                str(row["run_id"]),
            ),
        )
        for metric in METRICS:
            if not _finite(target[metric]) or not _finite(baseline[metric]):
                continue
            pairs.append(
                {
                    "task": target["task"],
                    "seed": target["seed"],
                    "condition": target["condition"],
                    "metric": metric,
                    "baseline_run_id": baseline["run_id"],
                    "baseline_width": baseline["width"],
                    "baseline_depth": baseline["depth"],
                    "baseline_value": baseline[metric],
                    "target_run_id": target["run_id"],
                    "target_width": target["width"],
                    "target_depth": target["depth"],
                    "target_value": target[metric],
                    "difference": float(target[metric]) - float(baseline[metric]),
                    "baseline_parameter_count": baseline["parameter_count"],
                    "target_parameter_count": target["parameter_count"],
                    "parameter_ratio_target_over_baseline": (
                        target_parameters / max(1, int(baseline["parameter_count"]))
                    ),
                    "species_ratio_target_over_baseline": (
                        int(target["species_count"])
                        / max(1, int(baseline["species_count"]))
                    ),
                    "reaction_ratio_target_over_baseline": (
                        int(target["reaction_count"])
                        / max(1, int(baseline["reaction_count"]))
                    ),
                    "matching_rule": (
                        "same task/seed/condition; lower depth; minimum "
                        "absolute log parameter-count ratio"
                    ),
                }
            )

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[
            (
                row["task"],
                row["condition"],
                row["baseline_width"],
                row["baseline_depth"],
                row["target_width"],
                row["target_depth"],
                row["metric"],
            )
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        (
            task,
            condition,
            baseline_width,
            baseline_depth,
            target_width,
            target_depth,
            metric,
        ) = key
        values = [float(row["difference"]) for row in group]
        aggregate = aggregate_metric(values)
        wins = sum(value > 0.0 for value in values)
        losses = sum(value < 0.0 for value in values)
        ties = len(values) - wins - losses
        summaries.append(
            {
                "task": task,
                "condition": condition,
                "baseline_width": baseline_width,
                "baseline_depth": baseline_depth,
                "target_width": target_width,
                "target_depth": target_depth,
                "metric": metric,
                **aggregate.to_dict(),
                "n_wins": wins,
                "n_losses": losses,
                "n_ties": ties,
                "sign_test_pvalue": (
                    None
                    if wins + losses == 0
                    else exact_sign_test_pvalue(wins, losses)
                ),
                "mean_parameter_ratio": sum(
                    float(row["parameter_ratio_target_over_baseline"])
                    for row in group
                )
                / len(group),
                "mean_reaction_ratio": sum(
                    float(row["reaction_ratio_target_over_baseline"])
                    for row in group
                )
                / len(group),
            }
        )
    return pairs, summaries


def _render_report(
    summary: dict[str, Any],
    paired: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> str:
    lines = [
        "# 切比雪夫四象限实验汇总",
        "",
        f"- 完成运行：{summary['completed_runs']}",
        f"- 完整性问题：{summary['integrity_issue_count']}",
        f"- 配对汇总行：{summary['paired_summary_rows']}",
        f"- 参数量匹配汇总行：{summary['capacity_matched_summary_rows']}",
        "",
        "主效应按同一 task、width、depth、seed 配对计算。交互项定义为 "
        "`(I1_C1-I0_C1)-(I1_C0-I0_C0)`。符号检验排除恰好为零的配对。",
    ]
    if issues:
        lines.extend(
            [
                "",
                "## 完整性状态",
                "",
                "存在缺失、重复、状态不完整或数据指纹不一致；解释配对效应前应先清零 "
                "`integrity_issues.csv`。",
            ]
        )
    if paired:
        lines.extend(
            [
                "",
                "## 结果文件",
                "",
                "`paired_summary.csv` 给出 I、C 及 I×C 交互的逐指标配对统计；"
                "`paired_differences.csv` 保留每个种子的原始差值。",
            ]
        )
    if summary["capacity_matched_pair_rows"]:
        lines.extend(
            [
                "",
                "## 同模型族容量控制",
                "",
                "`capacity_matched_pairs.csv` 在同一 task、seed、condition 内，"
                "为每个深层运行选择实际参数量最接近的较浅运行；"
                "`capacity_matched_summary.csv` 汇总性能差、参数比和反应数比。",
            ]
        )
    return "\n".join(lines) + "\n"


def _number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _finite(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze paired Chebyshev runs.")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--analysis-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = analyze_results(args.output_root, analysis_dir=args.analysis_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EFFECTS", "METRICS", "analyze_results", "main"]
