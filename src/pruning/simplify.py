from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pruning.decisions import PruningDecision
from pruning.structural_state import StructuralPruningState


@dataclass(frozen=True)
class StructuralSimplifyConfig:
    """Configuration for deterministic structural closure after threshold pruning.

    The closure is deliberately stage-agnostic. A component removed by input,
    basis, or evidence pruning is allowed to remove its now-invalid dependents.
    This avoids coupling, for example, evidence pruning to an explicit basis
    pruning pass merely to delete orphan basis species.
    """

    enabled: bool = True
    remove_basis_with_pruned_inputs: bool = True
    remove_edges_from_pruned_basis: bool = True
    remove_orphan_basis: bool = True
    remove_orphan_inputs: bool = True
    keep_at_least_one_input: bool = True
    keep_at_least_one_basis: bool = True
    min_positive_edges_per_class: int = 1
    min_negative_edges_per_class: int = 0


def _basis_input_dependencies(model) -> list[set[int]]:
    deps: list[set[int]] = []
    for spec in model.basis.feature_specs():
        deps.append(set(int(i) for i in getattr(spec, "dependencies", ())))
    return deps


def _basis_supports_inputs(basis_deps: list[set[int]], active_inputs: torch.Tensor) -> torch.Tensor:
    active_input_set = set(torch.nonzero(active_inputs, as_tuple=False).flatten().tolist())
    return torch.tensor(
        [dep.issubset(active_input_set) for dep in basis_deps],
        dtype=torch.bool,
        device=active_inputs.device,
    )


def _used_input_mask(
    basis_deps: list[set[int]],
    active_basis: torch.Tensor,
    n_inputs: int,
) -> torch.Tensor:
    used = torch.zeros(n_inputs, dtype=torch.bool, device=active_basis.device)
    for basis_index in torch.nonzero(active_basis, as_tuple=False).flatten().tolist():
        for input_index in basis_deps[int(basis_index)]:
            if 0 <= input_index < n_inputs:
                used[input_index] = True
    return used


def _raw_evidence_weights(model) -> tuple[torch.Tensor, torch.Tensor]:
    evidence = model.network.layers[0].evidence
    if hasattr(evidence, "raw_positive_weights") and hasattr(evidence, "raw_negative_weights"):
        return evidence.raw_positive_weights().detach(), evidence.raw_negative_weights().detach()
    return evidence.effective_weights()


def _choose_best_active_basis(weights: torch.Tensor, active_basis: torch.Tensor, class_index: int) -> int | None:
    candidates = torch.nonzero(active_basis, as_tuple=False).flatten()
    if candidates.numel() == 0:
        return None
    row = weights[class_index, candidates]
    local = int(torch.argmax(row).item())
    return int(candidates[local].item())


def _ensure_nonempty_inputs(
    state: StructuralPruningState,
    before: StructuralPruningState,
) -> None:
    if bool(state.active_inputs.any()):
        return

    old_candidates = torch.nonzero(before.active_inputs, as_tuple=False).flatten()
    if old_candidates.numel() > 0:
        state.active_inputs[int(old_candidates[0].item())] = True
    elif state.active_inputs.numel() > 0:
        state.active_inputs[0] = True


def _ensure_nonempty_basis(
    model,
    state: StructuralPruningState,
    before: StructuralPruningState,
    basis_deps: list[set[int]],
) -> None:
    if bool(state.active_basis.any()):
        return

    active_inputs = set(torch.nonzero(state.active_inputs, as_tuple=False).flatten().tolist())

    old_basis = torch.nonzero(before.active_basis, as_tuple=False).flatten().tolist()
    candidates = old_basis or list(range(state.active_basis.numel()))

    chosen: int | None = None
    for idx in candidates:
        if basis_deps[int(idx)].issubset(active_inputs):
            chosen = int(idx)
            break

    if chosen is None and candidates:
        chosen = int(candidates[0])
        for input_index in basis_deps[chosen]:
            if 0 <= input_index < state.active_inputs.numel():
                state.active_inputs[input_index] = True

    if chosen is not None:
        state.active_basis[chosen] = True


def _ensure_min_edges(model, state: StructuralPruningState, cfg: StructuralSimplifyConfig) -> None:
    w_pos, w_neg = _raw_evidence_weights(model)
    n_classes = state.active_pos_edges.shape[0]

    for class_index in range(n_classes):
        while int(state.active_pos_edges[class_index].sum().item()) < cfg.min_positive_edges_per_class:
            basis_index = _choose_best_active_basis(w_pos, state.active_basis, class_index)
            if basis_index is None:
                break
            state.active_pos_edges[class_index, basis_index] = True

        while int(state.active_neg_edges[class_index].sum().item()) < cfg.min_negative_edges_per_class:
            basis_index = _choose_best_active_basis(w_neg, state.active_basis, class_index)
            if basis_index is None:
                break
            state.active_neg_edges[class_index, basis_index] = True


def _close_once(
    model,
    state: StructuralPruningState,
    before: StructuralPruningState,
    cfg: StructuralSimplifyConfig,
    basis_deps: list[set[int]],
) -> None:
    if cfg.remove_edges_from_pruned_basis:
        state.active_pos_edges &= state.active_basis.unsqueeze(0)
        state.active_neg_edges &= state.active_basis.unsqueeze(0)

    if cfg.remove_basis_with_pruned_inputs:
        supported = _basis_supports_inputs(basis_deps, state.active_inputs)
        state.active_basis &= supported

    if cfg.remove_edges_from_pruned_basis:
        state.active_pos_edges &= state.active_basis.unsqueeze(0)
        state.active_neg_edges &= state.active_basis.unsqueeze(0)

    if cfg.remove_orphan_basis:
        has_edge = (state.active_pos_edges | state.active_neg_edges).any(dim=0)
        state.active_basis &= has_edge

    if cfg.remove_orphan_inputs:
        used_inputs = _used_input_mask(basis_deps, state.active_basis, state.active_inputs.numel())
        state.active_inputs &= used_inputs

    if cfg.keep_at_least_one_input:
        _ensure_nonempty_inputs(state, before)

    if cfg.keep_at_least_one_basis:
        _ensure_nonempty_basis(model, state, before, basis_deps)

    _ensure_min_edges(model, state, cfg)

    if cfg.remove_edges_from_pruned_basis:
        state.active_pos_edges &= state.active_basis.unsqueeze(0)
        state.active_neg_edges &= state.active_basis.unsqueeze(0)


def _state_equal(a: StructuralPruningState, b: StructuralPruningState) -> bool:
    return (
        torch.equal(a.active_inputs, b.active_inputs)
        and torch.equal(a.active_basis, b.active_basis)
        and torch.equal(a.active_pos_edges, b.active_pos_edges)
        and torch.equal(a.active_neg_edges, b.active_neg_edges)
    )


def simplify_structural_state(
    model,
    state: StructuralPruningState,
    cfg: StructuralSimplifyConfig | None = None,
) -> tuple[StructuralPruningState, dict[str, Any]]:
    cfg = cfg if cfg is not None else StructuralSimplifyConfig()
    if not cfg.enabled:
        return state.clone(), {"enabled": False, "applied": False}

    before = state.clone()
    new_state = state.clone()
    basis_deps = _basis_input_dependencies(model)

    max_iter = max(4, new_state.active_inputs.numel() + new_state.active_basis.numel() + 2)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        previous = new_state.clone()
        _close_once(model, new_state, before, cfg, basis_deps)
        if _state_equal(previous, new_state):
            break

    removed_inputs = set(before.active_input_indices) - set(new_state.active_input_indices)
    removed_basis = set(before.active_basis_indices) - set(new_state.active_basis_indices)

    removed_pos_edges: set[tuple[int, int]] = set()
    removed_neg_edges: set[tuple[int, int]] = set()
    before_pos = before.active_pos_edges.detach().cpu().bool()
    after_pos = new_state.active_pos_edges.detach().cpu().bool()
    before_neg = before.active_neg_edges.detach().cpu().bool()
    after_neg = new_state.active_neg_edges.detach().cpu().bool()

    for cls, basis in torch.nonzero(before_pos & ~after_pos, as_tuple=False).tolist():
        removed_pos_edges.add((int(cls), int(basis)))
    for cls, basis in torch.nonzero(before_neg & ~after_neg, as_tuple=False).tolist():
        removed_neg_edges.add((int(cls), int(basis)))

    decision = PruningDecision(
        stage="structural_closure",
        remove_inputs=removed_inputs,
        remove_basis=removed_basis,
        remove_pos_edges=removed_pos_edges,
        remove_neg_edges=removed_neg_edges,
        reason="dependent_or_orphan_component",
    )

    return new_state, {
        "enabled": True,
        "applied": not _state_equal(before, new_state),
        "iterations": iterations,
        "before": before.to_dict(),
        "after": new_state.to_dict(),
        "decision": decision.to_dict(),
        "config": {
            "remove_basis_with_pruned_inputs": cfg.remove_basis_with_pruned_inputs,
            "remove_edges_from_pruned_basis": cfg.remove_edges_from_pruned_basis,
            "remove_orphan_basis": cfg.remove_orphan_basis,
            "remove_orphan_inputs": cfg.remove_orphan_inputs,
            "keep_at_least_one_input": cfg.keep_at_least_one_input,
            "keep_at_least_one_basis": cfg.keep_at_least_one_basis,
            "min_positive_edges_per_class": cfg.min_positive_edges_per_class,
            "min_negative_edges_per_class": cfg.min_negative_edges_per_class,
        },
    }
