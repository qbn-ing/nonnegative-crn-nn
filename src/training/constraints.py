from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn

from models.positive import ExactNonnegativeParam


def iter_exact_nonnegative_parameters(
    module: nn.Module,
) -> Iterator[nn.Parameter]:
    """Yield tensors managed by :class:`ExactNonnegativeParam` modules."""

    _check_module(module)
    for submodule in module.modules():
        if isinstance(submodule, ExactNonnegativeParam):
            yield submodule.value


@torch.no_grad()
def apply_parameter_constraints_(module: nn.Module) -> None:
    """Apply the constraint implied by every managed parameterization.

    ``PositiveParam`` is feasible by construction and requires no action.
    ``ExactNonnegativeParam`` is directly optimized, so it is projected onto
    the nonnegative orthant after every optimizer step.  Callers do not choose
    a separate update policy: the parameter module determines the constraint.
    """

    _check_module(module)
    for submodule in module.modules():
        if isinstance(submodule, ExactNonnegativeParam):
            submodule.project_()


def project_exact_nonnegative_parameters_(module: nn.Module) -> None:
    """Backward-compatible name for :func:`apply_parameter_constraints_`."""

    apply_parameter_constraints_(module)


def _check_module(module: nn.Module) -> None:
    if not isinstance(module, nn.Module):
        raise TypeError(
            "module must be a torch.nn.Module. "
            f"Got {type(module).__name__}."
        )
