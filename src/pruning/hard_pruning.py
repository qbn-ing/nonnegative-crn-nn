from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from pruning.simplify import StructuralSimplifyConfig, simplify_structural_state
from pruning.structural_state import apply_structural_state_to_model, make_current_structural_state


@dataclass(frozen=True)
class HardPruningConfig:
    """Thresholds for deterministic post-training hard pruning.

    The operation is intentionally conservative by default: each input gate,
    basis-feature gate, and class evidence channel can keep at least one active
    edge to avoid deleting a class accidentally in early experiments.
    """

    enabled: bool = False
    input_gate_threshold: float = 1e-3
    basis_gate_threshold: float = 1e-3
    evidence_rate_threshold: float = 1e-3
    keep_at_least_one_input: bool = True
    keep_at_least_one_basis: bool = True
    keep_evidence_per_class: bool = True
    structural_simplify: bool = True
    structural_simplify_config: StructuralSimplifyConfig | None = None
    prune_inputs: bool = True
    prune_basis: bool = True
    prune_evidence: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(
                "enabled must be a bool. "
                f"Got {type(self.enabled).__name__}."
            )
        for name in (
            "input_gate_threshold",
            "basis_gate_threshold",
            "evidence_rate_threshold",
        ):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative. Got {value}.")
        for name in (
            "keep_at_least_one_input",
            "keep_at_least_one_basis",
            "keep_evidence_per_class",
            "structural_simplify",
            "prune_inputs",
            "prune_basis",
            "prune_evidence",
        ):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(
                    f"{name} must be a bool. Got {type(value).__name__}."
                )

    @classmethod
    def for_stage(cls, stage: str, **kwargs: Any) -> "HardPruningConfig":
        """Create a config for one layerwise pruning stage.

        ``stage`` may be ``input``, ``basis``, ``evidence`` or ``all``.
        Additional keyword arguments are forwarded to :class:`HardPruningConfig`.
        """

        normalized = str(stage).strip().lower().replace("-", "_")
        if normalized in {"inputs", "input_gate"}:
            normalized = "input"
        elif normalized in {"feature", "features", "basis_gate"}:
            normalized = "basis"
        elif normalized in {"edge", "edges", "evidence_edges"}:
            normalized = "evidence"

        flags = {
            "input": dict(prune_inputs=True, prune_basis=False, prune_evidence=False),
            "basis": dict(prune_inputs=False, prune_basis=True, prune_evidence=False),
            "evidence": dict(prune_inputs=False, prune_basis=False, prune_evidence=True),
            "all": dict(prune_inputs=True, prune_basis=True, prune_evidence=True),
        }
        if normalized not in flags:
            raise ValueError(
                "stage must be one of 'input', 'basis', 'evidence', or 'all'. "
                f"Got {stage!r}."
            )

        merged = {"enabled": True, **flags[normalized], **kwargs}
        return cls(**merged)


@dataclass(frozen=True)
class HardPruningResult:
    """Structured result returned after hard pruning."""

    enabled: bool
    applied: bool
    thresholds: dict[str, float]
    before: dict[str, Any]
    after: dict[str, Any]
    components: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "applied": self.applied,
            "thresholds": self.thresholds,
            "before": self.before,
            "after": self.after,
            "components": self.components,
        }


def apply_hard_pruning(
    model: nn.Module,
    config: HardPruningConfig | None = None,
) -> HardPruningResult:
    """Apply deterministic hard pruning to gates and evidence edges."""

    if not isinstance(model, nn.Module):
        raise TypeError(
            "model must be a torch.nn.Module. "
            f"Got {type(model).__name__}."
        )

    cfg = config if config is not None else HardPruningConfig(enabled=True)
    if not isinstance(cfg, HardPruningConfig):
        raise TypeError(
            "config must be a HardPruningConfig or None. "
            f"Got {type(cfg).__name__}."
        )

    before = _model_snapshot(model, cfg)
    components: dict[str, Any] = {}

    if not cfg.enabled:
        return HardPruningResult(
            enabled=False,
            applied=False,
            thresholds=_thresholds(cfg),
            before=before,
            after=before,
            components=components,
        )

    input_gate = getattr(model, "input_gate", None)
    if cfg.prune_inputs and input_gate is not None and hasattr(input_gate, "apply_hard_pruning"):
        components["input_gate"] = input_gate.apply_hard_pruning(
            threshold=cfg.input_gate_threshold,
            keep_at_least_one=cfg.keep_at_least_one_input,
        )

    basis_gate = getattr(model, "feature_gate", None)
    if basis_gate is None:
        basis_gate = getattr(model, "basis_gate", None)
    if cfg.prune_basis and basis_gate is not None and hasattr(basis_gate, "apply_hard_pruning"):
        components["basis_gate"] = basis_gate.apply_hard_pruning(
            threshold=cfg.basis_gate_threshold,
            keep_at_least_one=cfg.keep_at_least_one_basis,
        )

    evidence_results: list[dict[str, Any]] = []
    if cfg.prune_evidence:
        evidence_layers = list(_iter_evidence_layers(model))
        for index, evidence_layer in enumerate(evidence_layers):
            if not hasattr(evidence_layer, "apply_hard_pruning"):
                continue
            result = evidence_layer.apply_hard_pruning(
                threshold=cfg.evidence_rate_threshold,
                keep_per_class=cfg.keep_evidence_per_class,
            )
            evidence_results.append(
                {
                    "index": index,
                    "name": getattr(evidence_layer, "_name", None),
                    **result,
                }
            )

    if evidence_results:
        components["evidence_layers"] = evidence_results

    if cfg.structural_simplify:
        simplify_cfg = cfg.structural_simplify_config or StructuralSimplifyConfig(
            enabled=True,
            keep_at_least_one_input=cfg.keep_at_least_one_input,
            keep_at_least_one_basis=cfg.keep_at_least_one_basis,
            min_positive_edges_per_class=1 if cfg.keep_evidence_per_class else 0,
            min_negative_edges_per_class=0,
        )
        state = make_current_structural_state(model)
        simplified_state, simplify_report = simplify_structural_state(
            model,
            state,
            simplify_cfg,
        )
        applied_report = apply_structural_state_to_model(model, simplified_state)
        components["structural_simplify"] = {
            **simplify_report,
            "applied_masks": applied_report,
        }

    after = _model_snapshot(model, cfg)

    return HardPruningResult(
        enabled=True,
        applied=True,
        thresholds=_thresholds(cfg),
        before=before,
        after=after,
        components=components,
    )


@torch.no_grad()
def hard_pruning_report(
    model: nn.Module,
    config: HardPruningConfig | None = None,
) -> HardPruningResult:
    """Return current prunable counts without mutating the model."""

    if not isinstance(model, nn.Module):
        raise TypeError(
            "model must be a torch.nn.Module. "
            f"Got {type(model).__name__}."
        )

    cfg = config if config is not None else HardPruningConfig(enabled=False)
    if not isinstance(cfg, HardPruningConfig):
        raise TypeError(
            "config must be a HardPruningConfig or None. "
            f"Got {type(cfg).__name__}."
        )

    snapshot = _model_snapshot(model, cfg)
    return HardPruningResult(
        enabled=cfg.enabled,
        applied=False,
        thresholds=_thresholds(cfg),
        before=snapshot,
        after=snapshot,
        components={},
    )


def _thresholds(config: HardPruningConfig) -> dict[str, float]:
    return {
        "input_gate": float(config.input_gate_threshold),
        "basis_gate": float(config.basis_gate_threshold),
        "evidence_rate": float(config.evidence_rate_threshold),
    }


@torch.no_grad()
def _model_snapshot(model: nn.Module, config: HardPruningConfig) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}

    input_gate = getattr(model, "input_gate", None)
    if input_gate is not None and hasattr(input_gate, "active_mask"):
        structural_mask = input_gate.active_mask(0.0)
        threshold_mask = input_gate.active_mask(config.input_gate_threshold)
        snapshot["input_gate"] = _mask_count(structural_mask)
        snapshot["input_gate"]["above_threshold"] = _mask_count(threshold_mask)

    basis_gate = getattr(model, "feature_gate", None)
    if basis_gate is None:
        basis_gate = getattr(model, "basis_gate", None)
    if basis_gate is not None and hasattr(basis_gate, "active_mask"):
        structural_mask = basis_gate.active_mask(0.0)
        threshold_mask = basis_gate.active_mask(config.basis_gate_threshold)
        snapshot["basis_gate"] = _mask_count(structural_mask)
        snapshot["basis_gate"]["above_threshold"] = _mask_count(threshold_mask)

    evidence_layers = list(_iter_evidence_layers(model))
    evidence_snapshots: list[dict[str, Any]] = []
    total_before_threshold = 0
    total_after_threshold = 0

    for index, evidence_layer in enumerate(evidence_layers):
        item: dict[str, Any] = {
            "index": index,
            "name": getattr(evidence_layer, "_name", None),
        }

        if hasattr(evidence_layer, "active_edge_masks"):
            pos_mask, neg_mask = evidence_layer.active_edge_masks(0.0)
            pos_threshold_mask, neg_threshold_mask = evidence_layer.active_edge_masks(
                config.evidence_rate_threshold
            )
            pos_count = _mask_count(pos_mask)
            neg_count = _mask_count(neg_mask)
            item["positive"] = pos_count
            item["positive"]["above_threshold"] = _mask_count(pos_threshold_mask)
            item["negative"] = neg_count
            item["negative"]["above_threshold"] = _mask_count(neg_threshold_mask)
            item["total"] = {
                "active": pos_count["active"] + neg_count["active"],
                "inactive": pos_count["inactive"] + neg_count["inactive"],
                "total": pos_count["total"] + neg_count["total"],
                "above_threshold": (
                    int(pos_threshold_mask.detach().bool().sum().item())
                    + int(neg_threshold_mask.detach().bool().sum().item())
                ),
            }
            total_before_threshold += item["total"]["total"]
            total_after_threshold += item["total"]["active"]
        elif hasattr(evidence_layer, "n_edges"):
            n_edges = int(evidence_layer.n_edges)
            item["total"] = {"active": n_edges, "inactive": 0, "total": n_edges}
            total_before_threshold += n_edges
            total_after_threshold += n_edges

        evidence_snapshots.append(item)

    if evidence_snapshots:
        snapshot["evidence_layers"] = evidence_snapshots
        snapshot["evidence_edges"] = {
            "active": int(total_after_threshold),
            "inactive": int(total_before_threshold - total_after_threshold),
            "total": int(total_before_threshold),
        }

    if hasattr(model, "structure_info"):
        try:
            snapshot["structure_info"] = model.structure_info()
        except Exception as exc:  # pragma: no cover - defensive reporting path.
            snapshot["structure_info_error"] = repr(exc)

    export_summary = _export_structure_summary(model)
    if export_summary is not None:
        snapshot["export_structure_info"] = export_summary

    return snapshot


def _export_structure_summary(model: nn.Module) -> dict[str, Any] | None:
    export_model_spec = getattr(model, "export_model_spec", None)
    if not callable(export_model_spec):
        return None

    try:
        spec = export_model_spec()
    except Exception as exc:  # pragma: no cover - diagnostic path only.
        return {"error": f"export_model_spec failed: {exc}"}

    result: dict[str, Any] = {
        "type": spec.__class__.__name__,
        "input_dim": int(getattr(spec, "input_dim", 0)),
        "n_basis": len(getattr(spec, "basis", ())),
        "active_input_indices": getattr(spec, "active_input_indices", None),
        "active_basis_indices": getattr(spec, "active_basis_indices", None),
        "structural_pruning": getattr(spec, "structural_pruning", None),
    }

    layers = tuple(getattr(spec, "rational_layers", ()) or ())
    if layers:
        evidence = getattr(layers[0], "evidence", None)
        result["first_layer"] = {
            "input_dim": int(getattr(layers[0], "input_dim", 0)),
            "evidence_edges": len(getattr(evidence, "edges", ()) or ()),
            "evidence_input_dim": int(
                getattr(evidence, "input_dim", 0)
            ),
            "evidence_n_basis": int(
                getattr(evidence, "input_dim", 0)
            ),
            "evidence_output_dim": int(
                getattr(evidence, "output_dim", 0)
            ),
            "evidence_n_biases": int(
                getattr(evidence, "n_biases", 0)
            ),
        }

    try:
        from export.crn_instance import CRNInstanceOptions, build_crn_instance

        instance = build_crn_instance(
            spec,
            CRNInstanceOptions(require_frozen_rates=False),
        )
        result["compiled_crn"] = {
            "species": int(instance.species_count),
            "reactions": int(instance.reaction_count),
        }
    except Exception as exc:  # pragma: no cover - diagnostic path only.
        result["compiled_crn_error"] = str(exc)

    return result


def _iter_evidence_layers(model: nn.Module):
    seen: set[int] = set()

    network = getattr(model, "network", None)
    layers = getattr(network, "layers", None)
    if layers is not None:
        for layer in layers:
            evidence_layer = getattr(layer, "evidence", None)
            if evidence_layer is None:
                evidence_layer = getattr(layer, "evidence_layer", None)
            if evidence_layer is None:
                continue
            ident = id(evidence_layer)
            if ident in seen:
                continue
            seen.add(ident)
            yield evidence_layer

    evidence_layer = getattr(model, "evidence", None)
    if evidence_layer is None:
        evidence_layer = getattr(model, "evidence_layer", None)
    if evidence_layer is not None and id(evidence_layer) not in seen:
        seen.add(id(evidence_layer))
        yield evidence_layer

    for module in model.modules():
        if id(module) in seen:
            continue
        if module.__class__.__name__ == "DenseEvidenceLayer":
            seen.add(id(module))
            yield module


def _mask_count(mask: torch.Tensor) -> dict[str, int]:
    active = int(mask.detach().bool().sum().item())
    total = int(mask.numel())
    return {
        "active": active,
        "inactive": total - active,
        "total": total,
    }
