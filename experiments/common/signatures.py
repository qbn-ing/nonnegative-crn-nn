from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from utils.serialization import to_jsonable


def scientific_signature(
    payload: Any,
    *,
    exclude: Iterable[str] = (),
    length: int = 12,
) -> str:
    """Return a stable digest for a complete scientific run configuration."""

    if isinstance(length, bool) or not isinstance(length, int):
        raise TypeError("length must be an int.")
    if length < 8 or length > 64:
        raise ValueError("length must lie in [8, 64].")
    normalized = _normalize(payload)
    if not isinstance(normalized, dict):
        raise TypeError("scientific run payload must normalize to a mapping.")
    for key in exclude:
        normalized.pop(str(key), None)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def signed_run_id(base: str, payload: Any, *, length: int = 12) -> str:
    """Append a configuration signature to a human-readable run identifier."""

    if not isinstance(base, str) or not base.strip():
        raise ValueError("base must be a non-empty string.")
    signature = scientific_signature(
        payload,
        exclude=("run_id", "run_signature"),
        length=length,
    )
    return f"{base}__cfg_{signature}"


def validate_run_identity(instance: Any, payload: Mapping[str, Any]) -> None:
    """Reject serialized protocols whose derived identity was altered."""

    for key in ("run_id", "run_signature"):
        if key in payload and payload[key] != getattr(instance, key):
            raise ValueError(f"serialized {key} does not match the run fields.")


def _normalize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return to_jsonable(value)


__all__ = [
    "scientific_signature",
    "signed_run_id",
    "validate_run_identity",
]
