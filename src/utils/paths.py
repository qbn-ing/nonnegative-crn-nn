from __future__ import annotations

from os import PathLike
from pathlib import Path

PathInput = str | PathLike[str]


def normalize_path(path: PathInput) -> Path:
    """Normalize a string or pathlib-compatible path into a Path object."""

    if isinstance(path, (str, PathLike)):
        return Path(path)

    raise TypeError(
        "path must be a str or os.PathLike object. "
        f"Got {type(path).__name__}."
    )


def path_to_posix(path: PathInput) -> str:
    """Return a stable POSIX-style path string.

    This is mainly used for JSON/reporting payloads so that tests and result
    files are platform-independent.
    """

    return normalize_path(path).as_posix()


def validate_filename(filename: PathInput, *, name: str = "filename") -> str:
    """Validate that a value is a plain file name, not a path.

    Both POSIX and Windows separators are rejected explicitly, independent of
    the host platform.
    """

    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string.")

    if not isinstance(filename, (str, PathLike)):
        raise TypeError(
            f"{name} must be a str or os.PathLike object. "
            f"Got {type(filename).__name__}."
        )

    text = str(filename)

    if text == "":
        raise ValueError(f"{name} must not be empty.")

    if text in {".", ".."}:
        raise ValueError(f"{name} must be a plain file name, got {text!r}.")

    if "/" in text or "\\" in text:
        raise ValueError(
            f"{name} must be a plain file name without path separators, "
            f"got {text!r}."
        )

    return text


def ensure_filename(filename: PathInput, *, name: str = "filename") -> str:
    """Alias for validate_filename for call sites that prefer imperative names."""

    return validate_filename(filename, name=name)