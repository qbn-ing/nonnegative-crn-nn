from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Mapping

from .paths import normalize_path


def to_jsonable(value: Any) -> Any:
    """Convert common project objects into JSON-safe Python values.

    Supported inputs include:

    - mappings
    - list / tuple / set / frozenset
    - dataclass instances
    - objects exposing to_dict()
    - NumPy-like arrays and scalars, without requiring NumPy at import time
    - PyTorch tensors
    - pathlib / os.PathLike paths
    """

    # Project specification dataclasses use ``to_dict()`` to expose computed
    # export fields such as generated species and reactions.  Prefer that
    # public representation over ``dataclasses.asdict()``, which only includes
    # declared fields and would silently drop the computed export structure.
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_jsonable(to_dict())

    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))

    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]

    if isinstance(value, (set, frozenset)):
        return [to_jsonable(item) for item in value]

    if value.__class__.__module__.split(".", 1)[0] == "numpy":
        if hasattr(value, "tolist"):
            return to_jsonable(value.tolist())
        if hasattr(value, "item"):
            return to_jsonable(value.item())

    if value.__class__.__module__.split(".", 1)[0] == "torch":
        detached = value.detach().cpu()
        if detached.ndim == 0:
            return to_jsonable(detached.item())
        return to_jsonable(detached.tolist())

    if isinstance(value, (Path, PathLike)):
        return Path(value).as_posix()

    return value


def write_json(
    payload: Any,
    path: str | PathLike[str],
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> Path:
    """Write a JSON file and return the normalized output path."""

    output_path = normalize_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            to_jsonable(payload),
            file,
            indent=indent,
            ensure_ascii=False,
            sort_keys=sort_keys,
        )

    return output_path
