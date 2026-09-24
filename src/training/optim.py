from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from models.positive import ExactNonnegativeParam, PositiveParam
from utils.validation import check_nonnegative_float


@dataclass(frozen=True)
class OptimizerParameterGroupAudit:
    """Auditable partition of trainable parameters by constraint semantics."""

    positive_names: tuple[str, ...]
    exact_nonnegative_names: tuple[str, ...]
    ordinary_names: tuple[str, ...]
    positive_scalars: int
    exact_nonnegative_scalars: int
    ordinary_scalars: int

    @property
    def total_scalars(self) -> int:
        return (
            self.positive_scalars
            + self.exact_nonnegative_scalars
            + self.ordinary_scalars
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "positive_names": self.positive_names,
            "exact_nonnegative_names": self.exact_nonnegative_names,
            "ordinary_names": self.ordinary_names,
            "positive_scalars": self.positive_scalars,
            "exact_nonnegative_scalars": self.exact_nonnegative_scalars,
            "ordinary_scalars": self.ordinary_scalars,
            "total_scalars": self.total_scalars,
        }


def build_optimizer_parameter_groups(
    module: nn.Module,
    *,
    positive_weight_decay: float = 0.0,
    exact_nonnegative_weight_decay: float = 0.0,
    ordinary_weight_decay: float | None = None,
) -> tuple[list[dict[str, Any]], OptimizerParameterGroupAudit]:
    """Build complete, non-overlapping optimizer groups.

    Parameters represented by :class:`PositiveParam` and
    :class:`ExactNonnegativeParam` are recognized from their owning modules,
    rather than from fragile name patterns.  An unclassified trainable
    ``nn.Parameter`` is considered ordinary.  ``ordinary_weight_decay=None``
    makes such parameters an explicit error, which is the safe default for the
    CRN main model: every learned scalar is expected to have declared
    nonnegative semantics.
    """

    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module.")
    check_nonnegative_float(
        "positive_weight_decay",
        positive_weight_decay,
    )
    check_nonnegative_float(
        "exact_nonnegative_weight_decay",
        exact_nonnegative_weight_decay,
    )
    if ordinary_weight_decay is not None:
        check_nonnegative_float(
            "ordinary_weight_decay",
            ordinary_weight_decay,
        )

    named_trainable = {
        id(parameter): (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    classified: dict[int, str] = {}

    for submodule in module.modules():
        if isinstance(submodule, PositiveParam):
            _register_kind(
                classified,
                submodule.raw,
                "positive",
            )
        elif isinstance(submodule, ExactNonnegativeParam):
            _register_kind(
                classified,
                submodule.value,
                "exact_nonnegative",
            )

    positive = [
        item
        for parameter_id, item in named_trainable.items()
        if classified.get(parameter_id) == "positive"
    ]
    exact_nonnegative = [
        item
        for parameter_id, item in named_trainable.items()
        if classified.get(parameter_id) == "exact_nonnegative"
    ]
    ordinary = [
        item
        for parameter_id, item in named_trainable.items()
        if parameter_id not in classified
    ]

    if ordinary and ordinary_weight_decay is None:
        names = ", ".join(name for name, _ in ordinary)
        raise ValueError(
            "Unclassified ordinary trainable parameters require an explicit "
            f"ordinary_weight_decay policy. Found: {names}."
        )

    groups: list[dict[str, Any]] = []
    _append_group(
        groups,
        name="positive",
        items=positive,
        weight_decay=float(positive_weight_decay),
    )
    _append_group(
        groups,
        name="exact_nonnegative",
        items=exact_nonnegative,
        weight_decay=float(exact_nonnegative_weight_decay),
    )
    if ordinary_weight_decay is not None:
        _append_group(
            groups,
            name="ordinary",
            items=ordinary,
            weight_decay=float(ordinary_weight_decay),
        )
    if not groups:
        raise ValueError("module has no trainable parameters.")

    audit = OptimizerParameterGroupAudit(
        positive_names=tuple(name for name, _ in positive),
        exact_nonnegative_names=tuple(
            name for name, _ in exact_nonnegative
        ),
        ordinary_names=tuple(name for name, _ in ordinary),
        positive_scalars=sum(
            parameter.numel() for _, parameter in positive
        ),
        exact_nonnegative_scalars=sum(
            parameter.numel() for _, parameter in exact_nonnegative
        ),
        ordinary_scalars=sum(
            parameter.numel() for _, parameter in ordinary
        ),
    )
    expected = sum(
        parameter.numel()
        for _, parameter in named_trainable.values()
    )
    if audit.total_scalars != expected:
        raise RuntimeError(
            "Optimizer parameter grouping did not cover every trainable "
            "scalar exactly once."
        )
    return groups, audit


def _register_kind(
    classified: dict[int, str],
    parameter: nn.Parameter,
    kind: str,
) -> None:
    parameter_id = id(parameter)
    previous = classified.get(parameter_id)
    if previous is not None and previous != kind:
        raise ValueError(
            "A shared parameter cannot have two constraint semantics: "
            f"{previous!r} and {kind!r}."
        )
    classified[parameter_id] = kind


def _append_group(
    groups: list[dict[str, Any]],
    *,
    name: str,
    items: list[tuple[str, nn.Parameter]],
    weight_decay: float,
) -> None:
    if not items:
        return
    groups.append(
        {
            "params": [parameter for _, parameter in items],
            "weight_decay": weight_decay,
            "constraint_group": name,
        }
    )


__all__ = [
    "OptimizerParameterGroupAudit",
    "build_optimizer_parameter_groups",
]
