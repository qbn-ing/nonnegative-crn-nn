from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from utils.serialization import to_jsonable


class SpecExportError(ValueError):
    """Base error for export-spec validation failures."""

class UnfrozenRateError(SpecExportError):
    """Raised when an export spec contains trainable/unresolved rate values."""

    def __init__(self, records: list[RateRefRecord]) -> None:
        self.records = tuple(records)

        preview = ", ".join(
            f"{record.path}({record.key or '<unnamed>'})"
            for record in records[:5]
        )
        suffix = "" if len(records) <= 5 else f", ... +{len(records) - 5} more"

        super().__init__(
            "Export spec contains rate references without finite numeric "
            f"values: {preview}{suffix}."
        )

class DuplicateRateKeyError(SpecExportError):
    """Raised when frozen_rate_value_map sees duplicate rate keys."""

    def __init__(self, key: str, first_path: str, second_path: str) -> None:
        self.key = key
        self.first_path = first_path
        self.second_path = second_path

        super().__init__(
            f"Duplicate rate key {key!r} at {first_path!r} and {second_path!r}."
        )

@dataclass(frozen=True)
class RateRefRecord:
    """A discovered rate-reference payload inside a compiled/exported spec."""

    path: str
    key: str | None
    value: Any
    trainable: bool | None
    param_ref: str | None
    payload: Mapping[str, Any]

    @property
    def has_finite_numeric_value(self) -> bool:
        return is_finite_number(self.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "key": self.key,
            "value": self.value,
            "trainable": self.trainable,
            "param_ref": self.param_ref,
            "has_finite_numeric_value": self.has_finite_numeric_value,
            "payload": to_jsonable(dict(self.payload)),
        }
    
def spec_to_payload(spec: Any) -> Any:
    """Convert a model/layer spec into a JSON-like traversable payload.

    This delegates all dataclass, to_dict, tensor, NumPy, and Path conversion
    to the centralized utils.serialization.to_jsonable implementation.
    """

    return to_jsonable(spec)


def iter_spec_nodes(spec: Any) -> Iterator[tuple[str, Any]]:
    """Yield every node in a spec payload using stable JSONPath-like paths."""

    payload = spec_to_payload(spec)
    yield from _iter_nodes(payload, "$")

def collect_rate_refs(spec: Any) -> list[RateRefRecord]:
    """Collect all RateRef-like payloads from a spec.

    A RateRef-like payload is a mapping with a ``value`` field and at least one
    of ``param_ref`` or ``trainable``. It should also expose one naming field:
    ``key``, ``rate_key``, or ``name``.
    """

    payload = spec_to_payload(spec)
    records: list[RateRefRecord] = []

    for path, node in _iter_nodes(payload, "$"):
        if is_rate_ref_payload(node):
            records.append(_make_rate_ref_record(path, node))

    return records


def find_unfrozen_rate_refs(spec: Any) -> list[RateRefRecord]:
    """Return RateRefs whose values are not finite Python numbers."""

    return [
        record
        for record in collect_rate_refs(spec)
        if not record.has_finite_numeric_value
    ]

def assert_frozen_rates(spec: Any) -> Any:
    """Assert that every RateRef in a spec has a finite numeric value.

    Returns the JSON-like payload when validation succeeds. Exporters can call
    this before producing PySB code, CRN DSL text, or standalone JSON.
    """

    payload = spec_to_payload(spec)
    unfrozen = find_unfrozen_rate_refs(payload)

    if unfrozen:
        raise UnfrozenRateError(unfrozen)

    return payload

def frozen_rate_value_map(spec: Any) -> dict[str, float]:
    """Return ``{rate_key: value}`` after asserting all RateRefs are frozen.

    Duplicate rate keys are rejected because they make downstream exporter
    behavior ambiguous.
    """

    payload = assert_frozen_rates(spec)
    values: dict[str, float] = {}
    paths_by_key: dict[str, str] = {}

    for record in collect_rate_refs(payload):
        key = record.key if record.key is not None else record.path

        if key in values:
            raise DuplicateRateKeyError(
                key=key,
                first_path=paths_by_key[key],
                second_path=record.path,
            )

        values[key] = float(record.value)
        paths_by_key[key] = record.path

    return values

def is_rate_ref_payload(value: Any) -> bool:
    """Return True if a mapping looks like a trainable/exported RateRef."""

    if not isinstance(value, Mapping):
        return False

    if "value" not in value:
        return False

    has_rate_metadata = "param_ref" in value or "trainable" in value
    has_rate_name = any(key in value for key in ("key", "rate_key", "name"))

    return has_rate_metadata and has_rate_name

def is_finite_number(value: Any) -> bool:
    """Return True only for finite int/float values, excluding bool."""

    if isinstance(value, bool):
        return False

    if not isinstance(value, (int, float)):
        return False

    return math.isfinite(float(value))

def _iter_nodes(value: Any, path: str) -> Iterator[tuple[str, Any]]:
    yield path, value

    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_nodes(item, _join_path(path, str(key)))
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_nodes(item, f"{path}[{index}]")


def _join_path(parent: str, key: str) -> str:
    escaped_key = _escape_path_key(key)

    if escaped_key.startswith("["):
        return f"{parent}{escaped_key}"

    return f"{parent}.{escaped_key}"

def _make_rate_ref_record(
    path: str,
    payload: Mapping[str, Any],
) -> RateRefRecord:
    key = _optional_str(
        payload.get(
            "key",
            payload.get(
                "rate_key",
                payload.get("name"),
            ),
        )
    )
    trainable = payload.get("trainable")
    if trainable is not None and not isinstance(trainable, bool):
        trainable = None

    return RateRefRecord(
        path=path,
        key=key,
        value=payload.get("value"),
        trainable=trainable,
        param_ref=_optional_str(payload.get("param_ref")),
        payload=payload,
    )

def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _escape_path_key(key: str) -> str:
    if key == "":
        return "''"

    if _is_simple_path_key(key):
        return key

    escaped = key.replace("\\", "\\\\").replace("'", "\\'")
    return f"['{escaped}']"


def _is_simple_path_key(key: str) -> bool:
    if not key:
        return False

    first = key[0]
    if not (first.isalpha() or first == "_"):
        return False

    return all(char.isalnum() or char == "_" for char in key)