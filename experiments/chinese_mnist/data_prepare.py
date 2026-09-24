from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "chinese_mnist"
OFFICIAL_URL = "https://data.ncl.ac.uk/ndownloader/articles/10280831/versions/1"
MIRROR_URL = "https://github.com/CoderSerio/chinese-mnist/archive/refs/heads/master.zip"
EXPECTED_RECORDS = 15_000
EXPECTED_WRITERS = 100
EXPECTED_CLASSES = 15
EXPECTED_REPEATS = 10
PREPROCESSING_TAG = "otsu_graycentroid_originalscale_box"
VALUE_ORDER = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, 10000, 100000000)
VALUE_TO_INDEX = {value: index for index, value in enumerate(VALUE_ORDER)}
CHARACTER_BY_VALUE = {
    0: "零",
    1: "一",
    2: "二",
    3: "三",
    4: "四",
    5: "五",
    6: "六",
    7: "七",
    8: "八",
    9: "九",
    10: "十",
    100: "百",
    1000: "千",
    10000: "万",
    100000000: "亿",
}


@dataclass(frozen=True)
class PreprocessResult:
    image: np.ndarray
    threshold: int
    bbox_left: int
    bbox_top: int
    bbox_right: int
    bbox_bottom: int
    foreground_pixels: int
    centroid_x: float
    centroid_y: float
    shift_x: int
    shift_y: int


def processed_root(data_root: str | Path, image_size: int) -> Path:
    return Path(data_root).resolve() / "processed" / (
        f"chinese_mnist_{image_size}_{PREPROCESSING_TAG}"
    )


def processed_paths(data_root: str | Path, image_size: int) -> dict[str, Path]:
    root = processed_root(data_root, image_size)
    return {
        "root": root,
        "images": root / "images.npy",
        "labels": root / "labels.npy",
        "groups": root / "groups.npy",
        "values": root / "values.npy",
        "metadata": root / "metadata.csv",
        "info": root / "dataset_info.json",
        "preview": root / "preview_15_classes.png",
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def processed_content_hashes(paths: dict[str, Path]) -> dict[str, str]:
    """Hash every scientific input artifact used by training and splitting."""

    keys = ("images", "labels", "groups", "values", "metadata")
    return {key: sha256_file(paths[key]) for key in keys}


def dataset_fingerprint_from_info(info: dict[str, object]) -> str:
    payload = dict(info)
    payload.pop("dataset_fingerprint", None)
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def download_archive(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    temporary = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    if temporary.stat().st_size < 100_000:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("downloaded Chinese-MNIST archive is unexpectedly small.")
    os.replace(temporary, target)
    return target


def resolve_archive(
    data_root: str | Path,
    *,
    archive: str | Path | None = None,
    allow_download: bool = True,
) -> tuple[Path, str]:
    root = Path(data_root).resolve()
    if archive is not None:
        path = Path(archive).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, "explicit_archive"
    environment_archive = os.environ.get("CHINESE_MNIST_RAW_ARCHIVE")
    if environment_archive:
        path = Path(environment_archive).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, "environment_archive"
    candidates = sorted((root / "manual").glob("*.zip")) + sorted(
        (root / "raw").glob("*.zip")
    )
    candidates = [item for item in candidates if item.stat().st_size >= 100_000]
    if candidates:
        return candidates[0].resolve(), "existing_archive"
    if not allow_download:
        raise FileNotFoundError(
            f"No archive found beneath {root}. Supply --archive or enable download."
        )
    target = root / "raw" / "handwritten_chinese_numbers_official.zip"
    try:
        return download_archive(OFFICIAL_URL, target), "official_figshare"
    except Exception as official_error:
        print(f"[prepare] official download failed: {official_error}", flush=True)
        mirror = root / "raw" / "chinese_mnist_github_mirror.zip"
        return download_archive(MIRROR_URL, mirror), "github_mirror"


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if resolved_destination not in target.parents and target != resolved_destination:
                raise RuntimeError(f"unsafe archive member: {member.filename}")
        handle.extractall(destination)


def extract_nested_archives(archive: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    _safe_extract(archive, destination)
    extracted: set[Path] = set()
    for _ in range(3):
        pending = [
            item
            for item in sorted(destination.rglob("*.zip"))
            if item not in extracted
        ]
        if not pending:
            break
        for nested in pending:
            target = nested.parent / f"{nested.stem}_extracted"
            _safe_extract(nested, target)
            extracted.add(nested)


def discover_metadata_csv(root: Path) -> Path:
    required = {"suite_id", "sample_id", "code", "value"}
    for path in sorted(root.rglob("*.csv")):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                header = next(csv.reader(handle))
        except (OSError, StopIteration, UnicodeError):
            continue
        if required.issubset(set(header)):
            return path
    raise FileNotFoundError(f"metadata CSV was not found beneath {root}")


def discover_images(root: Path) -> list[Path]:
    output: list[Path] = []
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        output.extend(root.rglob(suffix))
    paths = sorted(set(output))
    if len(paths) < EXPECTED_RECORDS:
        raise RuntimeError(f"expected 15000 images, found {len(paths)}")
    return paths


def image_key(path: Path) -> tuple[int, int, int]:
    match = re.search(r"input_(\d+)_(\d+)_(\d+)", path.stem, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"unsupported image filename: {path.name}")
    return tuple(int(value) for value in match.groups())


def otsu_threshold(array: np.ndarray) -> int:
    if array.dtype != np.uint8:
        raise TypeError("Otsu input must be uint8.")
    histogram = np.bincount(array.reshape(-1), minlength=256).astype(np.float64)
    total = float(histogram.sum())
    if total <= 0:
        return 0
    probabilities = histogram / total
    omega = np.cumsum(probabilities)
    means = np.cumsum(probabilities * np.arange(256, dtype=np.float64))
    denominator = omega * (1.0 - omega)
    numerator = (means[-1] * omega - means) ** 2
    score = np.zeros_like(numerator)
    valid = denominator > 0
    score[valid] = numerator[valid] / denominator[valid]
    return int(np.argmax(score))


def foreground_mask(array: np.ndarray, threshold: int) -> tuple[np.ndarray, bool]:
    high = array > threshold
    border = np.concatenate((high[0], high[-1], high[:, 0], high[:, -1]))
    background_is_high = bool(border.mean() >= 0.5)
    foreground = ~high if background_is_high else high
    if not np.any(foreground):
        foreground = array != int(np.median(array))
    return foreground.astype(bool), background_is_high


def translate_zero_fill(array: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError("translation expects a 2-D image.")
    height, width = array.shape
    output = np.zeros_like(array)
    src_y0 = max(0, -shift_y)
    src_y1 = min(height, height - shift_y)
    src_x0 = max(0, -shift_x)
    src_x1 = min(width, width - shift_x)
    dst_y0 = max(0, shift_y)
    dst_x0 = max(0, shift_x)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
    if src_y1 > src_y0 and src_x1 > src_x0:
        output[dst_y0:dst_y1, dst_x0:dst_x1] = array[
            src_y0:src_y1, src_x0:src_x1
        ]
    return output


def preprocess_array(array: np.ndarray, image_size: int) -> PreprocessResult:
    """Locate with Otsu, center on the source canvas, then resize without crop."""

    if array.ndim != 2:
        raise ValueError("preprocessing expects one grayscale image.")
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    threshold = otsu_threshold(array)
    foreground, background_is_high = foreground_mask(array, threshold)
    yy_nonzero, xx_nonzero = np.nonzero(foreground)
    if len(xx_nonzero) == 0:
        raise RuntimeError("foreground localization produced an empty mask.")
    left = int(xx_nonzero.min())
    right = int(xx_nonzero.max()) + 1
    top = int(yy_nonzero.min())
    bottom = int(yy_nonzero.max()) + 1
    grayscale = 255 - array if background_is_high else array.copy()
    grayscale = np.where(foreground, grayscale, 0).astype(np.uint8)
    weights = grayscale.astype(np.float64)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        weights = foreground.astype(np.float64)
        weight_sum = float(weights.sum())
    yy, xx = np.indices(array.shape, dtype=np.float64)
    centroid_y = float((yy * weights).sum() / weight_sum)
    centroid_x = float((xx * weights).sum() / weight_sum)
    shift_y = int(np.rint((array.shape[0] - 1) / 2.0 - centroid_y))
    shift_x = int(np.rint((array.shape[1] - 1) / 2.0 - centroid_x))
    centered = translate_zero_fill(grayscale, shift_y, shift_x)
    resized = Image.fromarray(centered, mode="L").resize(
        (image_size, image_size), resample=Image.Resampling.BOX
    )
    output = np.asarray(resized, dtype=np.float32) / 255.0
    return PreprocessResult(
        image=output,
        threshold=threshold,
        bbox_left=left,
        bbox_top=top,
        bbox_right=right,
        bbox_bottom=bottom,
        foreground_pixels=int(foreground.sum()),
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        shift_x=shift_x,
        shift_y=shift_y,
    )


def preprocess_image(path: Path, image_size: int) -> PreprocessResult:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    return preprocess_array(array, image_size)


def _read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_RECORDS:
        raise RuntimeError(f"expected 15000 metadata rows, found {len(rows)}")
    return rows


def build_processed_dataset(
    extracted_root: Path,
    output_root: Path,
    *,
    image_size: int,
    source_kind: str,
    archive: Path,
) -> None:
    metadata_path = discover_metadata_csv(extracted_root)
    rows = _read_metadata(metadata_path)
    image_map: dict[tuple[int, int, int], Path] = {}
    for path in discover_images(extracted_root):
        try:
            key = image_key(path)
        except ValueError:
            continue
        if key in image_map:
            raise RuntimeError(f"duplicate image key: {key}")
        image_map[key] = path

    output_root.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(
        output_root / "images.npy",
        mode="w+",
        dtype=np.float32,
        shape=(EXPECTED_RECORDS, image_size * image_size),
    )
    labels = np.empty(EXPECTED_RECORDS, dtype=np.int64)
    groups = np.empty(EXPECTED_RECORDS, dtype=np.int64)
    values = np.empty(EXPECTED_RECORDS, dtype=np.int64)
    metadata_fields = (
        "row_index",
        "suite_id",
        "sample_id",
        "code",
        "value",
        "label_index",
        "character",
        "source_file",
        "otsu_threshold",
        "bbox_left",
        "bbox_top",
        "bbox_right",
        "bbox_bottom",
        "foreground_pixels",
        "centroid_x",
        "centroid_y",
        "shift_x",
        "shift_y",
    )
    with (output_root / "metadata.csv").open(
        "w", encoding="utf-8", newline=""
    ) as metadata_handle:
        writer = csv.DictWriter(metadata_handle, fieldnames=metadata_fields)
        writer.writeheader()
        for row_index, row in enumerate(rows):
            suite_id = int(row["suite_id"])
            sample_id = int(row["sample_id"])
            code = int(row["code"])
            value = int(row["value"])
            key = (suite_id, sample_id, code)
            image_path = image_map.get(key)
            if image_path is None:
                raise FileNotFoundError(f"missing image for metadata key {key}")
            if value not in VALUE_TO_INDEX:
                raise RuntimeError(f"unexpected class value: {value}")
            result = preprocess_image(image_path, image_size)
            images[row_index] = result.image.reshape(-1)
            labels[row_index] = VALUE_TO_INDEX[value]
            groups[row_index] = suite_id
            values[row_index] = value
            writer.writerow(
                {
                    "row_index": row_index,
                    "suite_id": suite_id,
                    "sample_id": sample_id,
                    "code": code,
                    "value": value,
                    "label_index": int(labels[row_index]),
                    "character": row.get("character", "").strip()
                    or CHARACTER_BY_VALUE[value],
                    "source_file": image_path.relative_to(extracted_root).as_posix(),
                    "otsu_threshold": result.threshold,
                    "bbox_left": result.bbox_left,
                    "bbox_top": result.bbox_top,
                    "bbox_right": result.bbox_right,
                    "bbox_bottom": result.bbox_bottom,
                    "foreground_pixels": result.foreground_pixels,
                    "centroid_x": result.centroid_x,
                    "centroid_y": result.centroid_y,
                    "shift_x": result.shift_x,
                    "shift_y": result.shift_y,
                }
            )
            if (row_index + 1) % 1000 == 0:
                print(f"[prepare] {row_index + 1}/{EXPECTED_RECORDS}", flush=True)
    images.flush()
    np.save(output_root / "labels.npy", labels)
    np.save(output_root / "groups.npy", groups)
    np.save(output_root / "values.npy", values)

    unique_groups, group_counts = np.unique(groups, return_counts=True)
    class_counts = np.bincount(labels, minlength=EXPECTED_CLASSES)
    if len(unique_groups) != EXPECTED_WRITERS or not np.all(group_counts == 150):
        raise RuntimeError("writer integrity check failed.")
    if not np.all(class_counts == EXPECTED_WRITERS * EXPECTED_REPEATS):
        raise RuntimeError("class integrity check failed.")
    info = {
        "schema_version": 2,
        "source_kind": source_kind,
        "source_archive_name": archive.name,
        "source_archive_sha256": sha256_file(archive),
        "official_url": OFFICIAL_URL,
        "mirror_url": MIRROR_URL,
        "license": "CC BY 4.0",
        "preprocessing_tag": PREPROCESSING_TAG,
        "image_size": [image_size, image_size],
        "input_dim": image_size * image_size,
        "num_records": EXPECTED_RECORDS,
        "num_writers": EXPECTED_WRITERS,
        "num_classes": EXPECTED_CLASSES,
        "repeats_per_writer_per_class": EXPECTED_REPEATS,
        "value_order": VALUE_ORDER,
        "preprocessing": [
            "grayscale conversion",
            "global Otsu threshold used only for foreground localization and polarity",
            "grayscale foreground retained and background set to zero",
            "grayscale center of mass translated to source-canvas center with zero fill",
            "original character scale and aspect ratio retained; no bounding-box crop",
            f"full source canvas downsampled to {image_size}x{image_size} with PIL BOX",
            "divide by 255 and flatten",
        ],
        "crop_normalization": False,
        "scale_normalization": False,
        "content_sha256": processed_content_hashes(
            processed_paths(output_root.parent.parent, image_size)
        ),
    }
    info["dataset_fingerprint"] = dataset_fingerprint_from_info(info)
    (output_root / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    create_preview(output_root, image_size)


def create_preview(output_root: Path, image_size: int) -> None:
    images = np.load(output_root / "images.npy", mmap_mode="r")
    labels = np.load(output_root / "labels.npy", mmap_mode="r")
    canvas = Image.new("L", (5 * image_size, 3 * image_size), color=0)
    for label in range(EXPECTED_CLASSES):
        index = int(np.flatnonzero(labels == label)[0])
        tile = Image.fromarray(
            np.rint(images[index].reshape(image_size, image_size) * 255).astype(np.uint8),
            mode="L",
        )
        canvas.paste(tile, ((label % 5) * image_size, (label // 5) * image_size))
    canvas.save(output_root / "preview_15_classes.png")


def validate_processed(data_root: str | Path, image_size: int) -> dict[str, object]:
    paths = processed_paths(data_root, image_size)
    missing = [str(path) for key, path in paths.items() if key != "root" and not path.is_file()]
    if missing:
        raise FileNotFoundError("processed dataset is incomplete: " + ", ".join(missing))
    images = np.load(paths["images"], mmap_mode="r")
    labels = np.load(paths["labels"], mmap_mode="r")
    groups = np.load(paths["groups"], mmap_mode="r")
    if images.shape != (EXPECTED_RECORDS, image_size * image_size):
        raise RuntimeError(f"unexpected images shape: {images.shape}")
    if labels.shape != (EXPECTED_RECORDS,) or groups.shape != (EXPECTED_RECORDS,):
        raise RuntimeError("unexpected label/group array shape.")
    if not np.isfinite(images).all() or float(images.min()) < 0 or float(images.max()) > 1:
        raise RuntimeError("processed inputs must be finite and lie in [0,1].")
    if len(np.unique(labels)) != EXPECTED_CLASSES or len(np.unique(groups)) != EXPECTED_WRITERS:
        raise RuntimeError("processed writer/class integrity failed.")
    info = json.loads(paths["info"].read_text(encoding="utf-8"))
    if int(info.get("schema_version", 0)) < 2:
        raise RuntimeError(
            "processed dataset fingerprint is metadata-only; rebuild it with the current preparer."
        )
    if info.get("preprocessing_tag") != PREPROCESSING_TAG:
        raise RuntimeError("processed preprocessing tag does not match v4.1.")
    if int(info.get("num_classes", -1)) != EXPECTED_CLASSES:
        raise RuntimeError("processed class count is not 15.")
    expected_hashes = info.get("content_sha256")
    observed_hashes = processed_content_hashes(paths)
    if expected_hashes != observed_hashes:
        raise RuntimeError("processed dataset content hash mismatch.")
    observed_fingerprint = dataset_fingerprint_from_info(info)
    if info.get("dataset_fingerprint") != observed_fingerprint:
        raise RuntimeError("processed dataset fingerprint mismatch.")
    return info


def prepare_dataset(
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    archive: str | Path | None = None,
    image_size: int = 28,
    force: bool = False,
    allow_download: bool = True,
) -> dict[str, object]:
    root = Path(data_root).resolve()
    output_root = processed_root(root, image_size)
    if output_root.exists() and not force:
        return validate_processed(root, image_size)
    selected, source_kind = resolve_archive(
        root, archive=archive, allow_download=allow_download
    )
    if output_root.exists():
        shutil.rmtree(output_root)
    (root / "raw").mkdir(parents=True, exist_ok=True)

    def build(selected_archive: Path, selected_kind: str) -> None:
        with tempfile.TemporaryDirectory(dir=root / "raw") as temporary:
            staging = Path(temporary) / "extracted"
            extract_nested_archives(selected_archive, staging)
            build_processed_dataset(
                staging,
                output_root,
                image_size=image_size,
                source_kind=selected_kind,
                archive=selected_archive,
            )

    try:
        build(selected, source_kind)
    except Exception:
        if source_kind != "official_figshare" or not allow_download:
            shutil.rmtree(output_root, ignore_errors=True)
            raise
        shutil.rmtree(output_root, ignore_errors=True)
        mirror = root / "raw" / "chinese_mnist_github_mirror.zip"
        if not mirror.is_file():
            download_archive(MIRROR_URL, mirror)
        build(mirror, "github_mirror")
    return validate_processed(root, image_size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the 15-class Chinese-MNIST dataset.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    info = prepare_dataset(
        data_root=args.data_root,
        archive=args.archive,
        image_size=args.image_size,
        force=args.force,
        allow_download=args.download,
    )
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
    return 0


__all__ = [
    "DEFAULT_DATA_ROOT",
    "EXPECTED_CLASSES",
    "EXPECTED_RECORDS",
    "EXPECTED_WRITERS",
    "PREPROCESSING_TAG",
    "PreprocessResult",
    "dataset_fingerprint_from_info",
    "prepare_dataset",
    "preprocess_array",
    "processed_content_hashes",
    "processed_paths",
    "translate_zero_fill",
    "validate_processed",
]
