from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from utils.validation import check_non_empty_str

@dataclass(frozen=True)
class ParameterCount:
    """Scalar parameter count for one named torch parameter."""

    name: str
    shape: tuple[int, ...]
    numel: int
    requires_grad: bool
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "numel": self.numel,
            "requires_grad": self.requires_grad,
            "dtype": self.dtype,
        }
    
@dataclass(frozen=True)
class LogicalCountSummary:
    """Logical structure summary extracted from structure_info().

    This is not a final physical CRN count. It summarizes the model-side
    logical species/reaction metadata currently exposed by basis, rational network,
    evidence, response, and output modules.
    """

    path: str
    type: str | None = None
    name: str | None = None

    input_dim: int | None = None
    n_basis: int | None = None
    depth: int | None = None
    n_outputs: int | None = None
    n_classes: int | None = None

    widths: tuple[int, ...] | None = None
    hidden_widths: tuple[int, ...] | None = None

    n_species: int | None = None
    n_constant_species: int | None = None
    n_evidence_species: int | None = None
    n_response_species: int | None = None
    n_output_species: int | None = None

    n_reactions: int | None = None
    n_rational_reactions: int | None = None
    n_output_reactions: int | None = None

    @classmethod
    def from_structure_info(
        cls,
        info: Mapping[str, Any],
        *,
        path: str = "model",
    ) -> LogicalCountSummary:
        if not isinstance(info, Mapping):
            raise TypeError(
                "info must be a mapping returned by structure_info(). "
                f"Got {type(info).__name__}."
            )

        check_non_empty_str("path", path)

        return cls(
            path=path,
            type=_optional_str(info.get("type")),
            name=_optional_str(info.get("name")),
            input_dim=_optional_int(info.get("input_dim")),
            n_basis=_optional_int(info.get("n_basis")),
            depth=_optional_int(info.get("depth")),
            n_outputs=_optional_int(info.get("n_outputs")),
            n_classes=_optional_int(info.get("n_classes")),
            widths=_optional_int_tuple(info.get("widths")),
            hidden_widths=_optional_int_tuple(info.get("hidden_widths")),
            n_species=_optional_int(info.get("n_species")),
            n_constant_species=_optional_int(
                info.get("n_constant_species")
            ),
            n_evidence_species=_optional_int(info.get("n_evidence_species")),
            n_response_species=_optional_int(info.get("n_response_species")),
            n_output_species=_optional_int(info.get("n_output_species")),
            n_reactions=_optional_int(info.get("n_reactions")),
            n_rational_reactions=_optional_int(info.get("n_rational_reactions")),
            n_output_reactions=_optional_int(info.get("n_output_reactions")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "type": self.type,
            "name": self.name,
            "input_dim": self.input_dim,
            "n_basis": self.n_basis,
            "depth": self.depth,
            "n_outputs": self.n_outputs,
            "n_classes": self.n_classes,
            "widths": self.widths,
            "hidden_widths": self.hidden_widths,
            "n_species": self.n_species,
            "n_constant_species": self.n_constant_species,
            "n_evidence_species": self.n_evidence_species,
            "n_response_species": self.n_response_species,
            "n_output_species": self.n_output_species,
            "n_reactions": self.n_reactions,
            "n_rational_reactions": self.n_rational_reactions,
            "n_output_reactions": self.n_output_reactions,
        }


@dataclass(frozen=True)
class ModelCountReport:
    """Combined logical structure and torch parameter count report."""

    root: LogicalCountSummary
    nodes: tuple[LogicalCountSummary, ...]
    parameters: tuple[ParameterCount, ...]

    @property
    def total_parameter_scalars(self) -> int:
        return sum(item.numel for item in self.parameters)

    @property
    def trainable_parameter_scalars(self) -> int:
        return sum(item.numel for item in self.parameters if item.requires_grad)

    @property
    def frozen_parameter_scalars(self) -> int:
        return sum(item.numel for item in self.parameters if not item.requires_grad)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root.to_dict(),
            "nodes": [item.to_dict() for item in self.nodes],
            "parameters": [item.to_dict() for item in self.parameters],
            "total_parameter_scalars": self.total_parameter_scalars,
            "trainable_parameter_scalars": self.trainable_parameter_scalars,
            "frozen_parameter_scalars": self.frozen_parameter_scalars,
            "n_nodes": self.n_nodes,
        }


def count_parameter_scalars(
    module: nn.Module,
    *,
    trainable_only: bool = False,
) -> int:
    """Count scalar torch parameters in a module."""

    if not isinstance(module, nn.Module):
        raise TypeError(
            "module must be a torch.nn.Module. "
            f"Got {type(module).__name__}."
        )

    if not isinstance(trainable_only, bool):
        raise TypeError(
            "trainable_only must be a bool. "
            f"Got {type(trainable_only).__name__}."
        )

    total = 0

    for parameter in module.parameters():
        if trainable_only and not parameter.requires_grad:
            continue
        total += parameter.numel()

    return int(total)


def named_parameter_counts(
    module: nn.Module,
    *,
    include_frozen: bool = True,
) -> tuple[ParameterCount, ...]:
    """Return per-parameter scalar counts."""

    if not isinstance(module, nn.Module):
        raise TypeError(
            "module must be a torch.nn.Module. "
            f"Got {type(module).__name__}."
        )

    if not isinstance(include_frozen, bool):
        raise TypeError(
            "include_frozen must be a bool. "
            f"Got {type(include_frozen).__name__}."
        )

    result: list[ParameterCount] = []

    for name, parameter in module.named_parameters():
        if not include_frozen and not parameter.requires_grad:
            continue

        result.append(
            ParameterCount(
                name=name,
                shape=tuple(int(dim) for dim in parameter.shape),
                numel=int(parameter.numel()),
                requires_grad=bool(parameter.requires_grad),
                dtype=str(parameter.dtype),
            )
        )

    return tuple(result)

def flatten_logical_counts(
    obj_or_info: Any,
    *,
    root_path: str = "model",
) -> tuple[LogicalCountSummary, ...]:
    """Flatten nested structure_info() dictionaries into logical count nodes.

    Parameters
    ----------
    obj_or_info:
        Either an object exposing structure_info(), or a raw structure_info dict.

    root_path:
        Path name assigned to the root summary.
    """

    check_non_empty_str("root_path", root_path)

    info = _get_structure_info(obj_or_info)

    result: list[LogicalCountSummary] = []
    _collect_logical_counts(info, path=root_path, result=result)

    return tuple(result)


def logical_counts_to_rows(
    summaries: tuple[LogicalCountSummary, ...] | list[LogicalCountSummary],
) -> list[dict[str, Any]]:
    """Convert logical count summaries into table-like dictionaries."""

    if not isinstance(summaries, (tuple, list)):
        raise TypeError(
            "summaries must be a tuple or list of LogicalCountSummary. "
            f"Got {type(summaries).__name__}."
        )

    rows: list[dict[str, Any]] = []

    for index, item in enumerate(summaries):
        if not isinstance(item, LogicalCountSummary):
            raise TypeError(
                "Every item in summaries must be a LogicalCountSummary. "
                f"Got {type(item).__name__} at index {index}."
            )
        rows.append(item.to_dict())

    return rows


def summarize_model_counts(module: nn.Module) -> ModelCountReport:
    """Build a full count report for a model exposing structure_info()."""

    if not isinstance(module, nn.Module):
        raise TypeError(
            "module must be a torch.nn.Module. "
            f"Got {type(module).__name__}."
        )

    nodes = flatten_logical_counts(module)
    parameters = named_parameter_counts(module, include_frozen=True)

    return ModelCountReport(
        root=nodes[0],
        nodes=nodes,
        parameters=parameters,
    )


def _collect_logical_counts(
    info: Mapping[str, Any],
    *,
    path: str,
    result: list[LogicalCountSummary],
) -> None:
    result.append(LogicalCountSummary.from_structure_info(info, path=path))

    for child_name, child_info in _iter_structure_children(info):
        child_path = f"{path}.{child_name}"
        _collect_logical_counts(child_info, path=child_path, result=result)


def _iter_structure_children(
    info: Mapping[str, Any],
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    children: list[tuple[str, Mapping[str, Any]]] = []

    for key, value in info.items():
        if isinstance(value, Mapping):
            if _looks_like_structure_info(value):
                children.append((str(key), value))
            continue

        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, Mapping) and _looks_like_structure_info(item):
                    children.append((f"{key}[{index}]", item))

    return tuple(children)


def _looks_like_structure_info(value: Mapping[str, Any]) -> bool:
    structure_keys = {
        "type",
        "name",
        "input_dim",
        "n_basis",
        "depth",
        "n_outputs",
        "n_species",
        "n_reactions",
    }

    return any(key in value for key in structure_keys)


def _get_structure_info(obj_or_info: Any) -> Mapping[str, Any]:
    if isinstance(obj_or_info, Mapping):
        return obj_or_info

    structure_info = getattr(obj_or_info, "structure_info", None)

    if structure_info is None or not callable(structure_info):
        raise TypeError(
            "obj_or_info must be a structure_info dict or expose a callable "
            "structure_info() method."
        )

    info = structure_info()

    if not isinstance(info, Mapping):
        raise TypeError(
            "structure_info() must return a mapping. "
            f"Got {type(info).__name__}."
        )

    return info


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, bool):
        raise TypeError("Expected int-like value, got bool.")

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Expected int-like value or None. Got {value!r}."
        ) from exc


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str):
        raise TypeError(
            f"Expected str or None. Got {type(value).__name__}."
        )

    return value


def _optional_int_tuple(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None

    if isinstance(value, (str, bytes)):
        raise TypeError("Expected a sequence of ints, got string-like value.")

    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(
            f"Expected a sequence of ints or None. Got {type(value).__name__}."
        ) from exc

    result: list[int] = []

    for item in items:
        if isinstance(item, bool):
            raise TypeError("Expected int-like value, got bool.")
        try:
            result.append(int(item))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Expected int-like value in sequence. Got {item!r}."
            ) from exc

    return tuple(result)
