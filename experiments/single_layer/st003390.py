from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ANALYSES: dict[str, str] = {
    "M1": "AN005559",
    "M2": "AN005560",
}
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "."}


@dataclass(frozen=True)
class Assay:
    method: str
    analysis_id: str
    unit: str
    samples: tuple[str, ...]
    labels: tuple[int, ...]
    feature_names: tuple[str, ...]
    original_names: tuple[str, ...]
    values: np.ndarray


def valid_mwtab(path: Path, analysis_id: str) -> bool:
    if not path.is_file() or path.stat().st_size < 10_000:
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return (
        analysis_id in text
        and "MS_METABOLITE_DATA_START" in text
        and "MS_METABOLITE_DATA_END" in text
    )


def download_one(analysis_id: str, output: Path, *, retries: int) -> None:
    if valid_mwtab(output, analysis_id):
        print(f"[download] existing valid {output}", flush=True)
        return
    url = (
        "https://www.metabolomicsworkbench.org/rest/study/analysis_id/"
        f"{analysis_id}/mwtab/txt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "CRN-research-data-audit/1.0",
                    "Accept": "text/plain,*/*;q=0.8",
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                temporary.write_bytes(response.read())
            if not valid_mwtab(temporary, analysis_id):
                raise RuntimeError("downloaded body is not a valid mwTab file.")
            temporary.replace(output)
            print(f"[download] {analysis_id} -> {output}", flush=True)
            return
        except Exception as exc:
            last = exc
            if temporary.exists():
                temporary.unlink()
            print(
                f"[download] attempt {attempt}/{retries} failed: {exc}",
                flush=True,
            )
            if attempt < retries:
                time.sleep(min(30, 2 * attempt))
    raise RuntimeError(f"failed to download {analysis_id}: {last}")


def parse_mwtab(path: Path, method: str, analysis_id: str) -> Assay:
    text = path.read_text(encoding="utf-8", errors="replace")
    if analysis_id not in text:
        raise ValueError(f"{path} does not contain {analysis_id}.")
    lines = [line.rstrip("\r\n") for line in text.splitlines()]
    unit = "unknown"
    for line in lines:
        if line.startswith("MS_METABOLITE_DATA:UNITS"):
            parts = line.split("\t", 1)
            unit = parts[1].strip() if len(parts) == 2 else "unknown"
            break
    try:
        start = lines.index("MS_METABOLITE_DATA_START") + 1
        end = lines.index("MS_METABOLITE_DATA_END", start)
    except ValueError as exc:
        raise ValueError(f"missing metabolite-data markers in {path}.") from exc
    block = [line for line in lines[start:end] if line.strip()]
    if len(block) < 4:
        raise ValueError(f"metabolite-data block is too short in {path}.")
    header = _split_tsv(block[0])
    factors = _split_tsv(block[1])
    if header[0].strip().lower() != "samples":
        raise ValueError("unexpected sample header.")
    if factors[0].strip().lower() != "factors":
        raise ValueError("unexpected factors header.")
    samples = tuple(value.strip() for value in header[1:])
    if len(samples) != len(set(samples)):
        raise ValueError(f"duplicate sample IDs in {path}.")
    if len(factors) != len(header):
        raise ValueError("Factors/sample column mismatch.")
    labels = tuple(_parse_factor_label(value) for value in factors[1:])
    feature_names: list[str] = []
    original_names: list[str] = []
    columns: list[np.ndarray] = []
    used: set[str] = set()
    for row_number, line in enumerate(block[2:], start=1):
        parts = _split_tsv(line)
        if len(parts) != len(header):
            raise ValueError(
                f"feature row width mismatch in {path}, row {row_number}."
            )
        original = re.sub(r"\s+", " ", parts[0].strip())
        if not original:
            continue
        feature_name = _unique_feature_name(method, original, used)
        values = np.asarray([_parse_number(value) for value in parts[1:]])
        original_names.append(original)
        feature_names.append(feature_name)
        columns.append(values)
    matrix = np.stack(columns, axis=1)
    return Assay(
        method=method,
        analysis_id=analysis_id,
        unit=unit,
        samples=samples,
        labels=labels,
        feature_names=tuple(feature_names),
        original_names=tuple(original_names),
        values=matrix,
    )


def merge_assays(assays: Iterable[Assay]) -> Assay:
    items = tuple(assays)
    if not items:
        raise ValueError("at least one assay is required.")
    reference = items[0]
    matrices: list[np.ndarray] = []
    feature_names: list[str] = []
    original_names: list[str] = []
    for assay in items:
        if set(assay.samples) != set(reference.samples):
            raise ValueError("M1/M2 sample sets differ.")
        row_by_sample = {
            sample: index for index, sample in enumerate(assay.samples)
        }
        label_by_sample = dict(zip(assay.samples, assay.labels))
        if any(
            label_by_sample[sample] != reference.labels[index]
            for index, sample in enumerate(reference.samples)
        ):
            raise ValueError("M1/M2 phenotype labels differ.")
        order = [row_by_sample[sample] for sample in reference.samples]
        matrices.append(assay.values[order])
        feature_names.extend(assay.feature_names)
        original_names.extend(assay.original_names)
    return Assay(
        method="M1M2",
        analysis_id="AN005559+AN005560",
        unit="method_specific",
        samples=reference.samples,
        labels=reference.labels,
        feature_names=tuple(feature_names),
        original_names=tuple(original_names),
        values=np.concatenate(matrices, axis=1),
    )


def write_dataset(assay: Assay, output: Path) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label", "group", *assay.feature_names])
        for sample, label, values in zip(
            assay.samples, assay.labels, assay.values
        ):
            writer.writerow(
                [
                    sample,
                    label,
                    sample,
                    *[
                        "" if np.isnan(value) else format(float(value), ".17g")
                        for value in values
                    ],
                ]
            )
    labels, counts = np.unique(assay.labels, return_counts=True)
    audit: dict[str, object] = {
        "schema_version": 1,
        "study_id": "ST003390",
        "project_id": "PR002101",
        "analyses": ANALYSES,
        "included_methods": ["M1", "M2"],
        "M3_excluded_as_module": True,
        "M4_excluded": True,
        "n_samples": len(assay.samples),
        "n_features": len(assay.feature_names),
        "class_counts": {
            str(int(label)): int(count)
            for label, count in zip(labels, counts)
        },
        "sample_ids_unique": len(assay.samples) == len(set(assay.samples)),
        "missing_values": int(np.isnan(assay.values).sum()),
        "negative_values": int(np.sum(np.nan_to_num(assay.values) < 0.0)),
        "output": output.as_posix(),
        "preprocessing_scope": "all imputation/filtering/scaling is deferred to each training fold",
    }
    if audit["n_samples"] != 300 or audit["n_features"] != 201:
        raise ValueError(
            "expected ST003390 M1/M2 shape (300, 201), got "
            f"({audit['n_samples']}, {audit['n_features']})."
        )
    if audit["class_counts"] != {"0": 200, "1": 100}:
        raise ValueError("expected class counts 200 healthy / 100 T2DM.")
    audit_path = output.with_name("data_audit.json")
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and prepare ST003390 M1/M2 concentration data."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data") / "st003390" / "raw",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "st003390" / "st003390_m1m2.csv",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--retries", type=int, default=10)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.retries <= 0:
        raise ValueError("retries must be positive.")
    assays: list[Assay] = []
    for method, analysis_id in ANALYSES.items():
        path = args.raw_dir / f"{method}_{analysis_id}.txt"
        if not valid_mwtab(path, analysis_id):
            if not args.download:
                raise FileNotFoundError(path)
            download_one(analysis_id, path, retries=args.retries)
        assays.append(parse_mwtab(path, method, analysis_id))
    audit = write_dataset(merge_assays(assays), args.output)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


def _split_tsv(line: str) -> list[str]:
    parts = line.split("\t")
    while parts and parts[-1] == "":
        parts.pop()
    return parts


def _parse_factor_label(text: str) -> int:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if re.search(r"phenotype\s*:\s*t2dm(?:\s*\||$)", normalized):
        return 1
    if re.search(
        r"phenotype\s*:\s*healthy control(?:\s*\||$)", normalized
    ):
        return 0
    raise ValueError(f"unrecognized phenotype factor: {text!r}.")


def _parse_number(token: str) -> float:
    stripped = token.strip()
    if stripped.lower() in MISSING_TOKENS:
        return float("nan")
    value = float(stripped)
    if value < -1e-12:
        raise ValueError(f"negative concentration: {value}.")
    return max(value, 0.0)


def _unique_feature_name(method: str, original: str, used: set[str]) -> str:
    base = f"{method}|{original}"
    name = base
    index = 2
    while name in used:
        name = f"{base}__dup{index}"
        index += 1
    used.add(name)
    return name


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSES",
    "Assay",
    "build_parser",
    "download_one",
    "main",
    "merge_assays",
    "parse_mwtab",
    "valid_mwtab",
    "write_dataset",
]
