from __future__ import annotations

from numbers import Integral
from math import isfinite


def is_int_not_bool(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def check_positive_int(name: str, value: int) -> None:
    if not is_int_not_bool(value):
        raise TypeError(f"{name} must be an int. Got {type(value).__name__}.")

    if value <= 0:
        raise ValueError(f"{name} must be positive. Got {value}.")


def check_nonnegative_int(name: str, value: int) -> None:
    if not is_int_not_bool(value):
        raise TypeError(f"{name} must be an int. Got {type(value).__name__}.")

    if value < 0:
        raise ValueError(f"{name} must be non-negative. Got {value}.")


def check_non_empty_str(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    
def check_positive_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be int or float. Got {type(value).__name__}.")

    if not isfinite(float(value)):
        raise ValueError(f"{name} must be finite. Got {value}.")

    if value <= 0:
        raise ValueError(f"{name} must be positive. Got {value}.")

def check_nonnegative_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be int or float. Got {type(value).__name__}.")

    if not isfinite(float(value)):
        raise ValueError(f"{name} must be finite. Got {value}.")

    if value < 0:
        raise ValueError(f"{name} must be non-negative. Got {value}.")
    
def check_bool(value: bool, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(
            f"{name} must be a bool. Got {type(value).__name__}."
        )