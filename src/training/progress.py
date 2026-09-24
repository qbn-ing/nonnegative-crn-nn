from __future__ import annotations

import importlib
import sys
from typing import Any, Iterable, Literal, TypeVar


T = TypeVar("T")
ProgressSetting = bool | Literal["auto"]


def progress_iter(
    iterable: Iterable[T],
    *,
    progress: ProgressSetting = "auto",
    total: int | None = None,
    desc: str | None = None,
    unit: str = "it",
    leave: bool = True,
) -> Iterable[T]:
    """Wrap an iterable with tqdm when requested by the caller."""

    enabled = progress_is_enabled(progress)
    if not enabled:
        return iterable

    try:
        module = importlib.import_module("tqdm.auto")
    except ModuleNotFoundError as exc:
        if exc.name not in {"tqdm", "tqdm.auto"}:
            raise
        raise RuntimeError(
            "Progress display requires tqdm. Install it with "
            "`python -m pip install tqdm`, or pass progress=False."
        ) from exc

    return module.tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        dynamic_ncols=True,
    )


def progress_is_enabled(progress: ProgressSetting) -> bool:
    """Resolve True/False/'auto' without importing tqdm."""

    if progress == "auto":
        return _interactive_output()
    if isinstance(progress, bool):
        return progress
    raise ValueError("progress must be True, False, or 'auto'.")


def set_progress_postfix(
    iterator: Any,
    values: dict[str, Any],
) -> None:
    """Update a tqdm postfix while remaining a no-op for plain iterables."""

    setter = getattr(iterator, "set_postfix", None)
    if callable(setter):
        setter(values, refresh=False)


def _interactive_output() -> bool:
    return bool(
        getattr(sys.stderr, "isatty", lambda: False)()
        or "ipykernel" in sys.modules
    )


__all__ = [
    "ProgressSetting",
    "progress_is_enabled",
    "progress_iter",
    "set_progress_postfix",
]
