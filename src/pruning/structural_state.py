from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class StructuralPruningState:
    active_inputs: torch.Tensor          # shape: [n_inputs], bool
    active_basis: torch.Tensor           # shape: [n_basis], bool
    active_pos_edges: torch.Tensor       # shape: [n_classes, n_basis], bool
    active_neg_edges: torch.Tensor       # shape: [n_classes, n_basis], bool

    def clone(self) -> "StructuralPruningState":
        return StructuralPruningState(
            active_inputs=self.active_inputs.clone(),
            active_basis=self.active_basis.clone(),
            active_pos_edges=self.active_pos_edges.clone(),
            active_neg_edges=self.active_neg_edges.clone(),
        )

    @property
    def active_input_indices(self) -> list[int]:
        return torch.nonzero(self.active_inputs, as_tuple=False).flatten().tolist()

    @property
    def active_basis_indices(self) -> list[int]:
        return torch.nonzero(self.active_basis, as_tuple=False).flatten().tolist()

    def input_index_map(self) -> dict[int, int]:
        return {old: new for new, old in enumerate(self.active_input_indices)}

    def basis_index_map(self) -> dict[int, int]:
        return {old: new for new, old in enumerate(self.active_basis_indices)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_input_indices": self.active_input_indices,
            "active_basis_indices": self.active_basis_indices,
            "n_active_inputs": int(self.active_inputs.sum().item()),
            "n_total_inputs": int(self.active_inputs.numel()),
            "n_active_basis": int(self.active_basis.sum().item()),
            "n_total_basis": int(self.active_basis.numel()),
            "n_active_pos_edges": int(self.active_pos_edges.sum().item()),
            "n_total_pos_edges": int(self.active_pos_edges.numel()),
            "n_active_neg_edges": int(self.active_neg_edges.sum().item()),
            "n_total_neg_edges": int(self.active_neg_edges.numel()),
        }


def _first_evidence(model):
    layers = getattr(getattr(model, "network", None), "layers", None)
    if not layers:
        raise ValueError(
            "Structural pruning requires model.network.layers."
        )
    first_layer = layers[0]
    evidence = getattr(first_layer, "evidence", None)
    if evidence is None:
        raise ValueError(
            "The first rational layer does not expose an evidence module."
        )
    return evidence


def _input_mask_from_model(model, *, device: torch.device) -> torch.Tensor:
    n_inputs = int(model.input_dim)
    input_gate = getattr(model, "input_gate", None)
    if input_gate is not None and hasattr(input_gate, "active_mask"):
        return input_gate.active_mask(0.0).detach().bool().to(device=device)
    return torch.ones(n_inputs, dtype=torch.bool, device=device)


def _basis_mask_from_model(model, *, device: torch.device) -> torch.Tensor:
    n_basis = int(model.basis.n_basis)
    basis_gate = getattr(model, "feature_gate", None)
    if basis_gate is None:
        basis_gate = getattr(model, "basis_gate", None)
    if basis_gate is not None and hasattr(basis_gate, "active_mask"):
        return basis_gate.active_mask(0.0).detach().bool().to(device=device)
    return torch.ones(n_basis, dtype=torch.bool, device=device)


def _edge_masks_from_model(model) -> tuple[torch.Tensor, torch.Tensor]:
    evidence = _first_evidence(model)
    if hasattr(evidence, "active_edge_masks"):
        pos_mask, neg_mask = evidence.active_edge_masks(0.0)
        return pos_mask.detach().bool(), neg_mask.detach().bool()

    n_classes = int(getattr(evidence, "output_dim"))
    n_basis = int(getattr(evidence, "input_dim"))
    device = next(evidence.parameters()).device
    pos = torch.ones((n_classes, n_basis), dtype=torch.bool, device=device)
    neg = torch.ones_like(pos)
    return pos, neg


def make_initial_structural_state(model) -> StructuralPruningState:
    first_evidence = _first_evidence(model)
    w_pos, w_neg = first_evidence.effective_weights()

    return StructuralPruningState(
        active_inputs=torch.ones(int(model.input_dim), dtype=torch.bool, device=w_pos.device),
        active_basis=torch.ones(int(model.basis.n_basis), dtype=torch.bool, device=w_pos.device),
        active_pos_edges=torch.ones_like(w_pos, dtype=torch.bool),
        active_neg_edges=torch.ones_like(w_neg, dtype=torch.bool),
    )


@torch.no_grad()
def make_current_structural_state(model) -> StructuralPruningState:
    """Read the currently active hard masks from the model.

    This is intentionally mask-based, not threshold-based. Thresholding remains
    the responsibility of ``apply_hard_pruning``; this function captures the
    post-threshold structural state so that closure rules can remove dependent
    objects independently from the original pruning stage.
    """

    pos_mask, neg_mask = _edge_masks_from_model(model)
    device = pos_mask.device

    return StructuralPruningState(
        active_inputs=_input_mask_from_model(model, device=device),
        active_basis=_basis_mask_from_model(model, device=device),
        active_pos_edges=pos_mask.to(device=device),
        active_neg_edges=neg_mask.to(device=device),
    )


@torch.no_grad()
def apply_structural_state_to_model(model, state: StructuralPruningState) -> dict[str, Any]:
    """Write a structural state back into hard masks when the modules support it."""

    applied: dict[str, Any] = {}

    input_gate = getattr(model, "input_gate", None)
    if input_gate is not None and hasattr(input_gate, "gate"):
        mask = state.active_inputs.to(dtype=input_gate.gate.hard_mask.dtype, device=input_gate.gate.hard_mask.device)
        before = int((input_gate.gate.hard_mask.detach() > 0.0).sum().item())
        input_gate.gate.hard_mask.copy_(mask)
        after = int(state.active_inputs.sum().item())
        applied["input_gate"] = {"before": before, "after": after, "pruned": before - after}

    basis_gate = getattr(model, "feature_gate", None)
    if basis_gate is None:
        basis_gate = getattr(model, "basis_gate", None)
    if basis_gate is not None and hasattr(basis_gate, "gate"):
        mask = state.active_basis.to(dtype=basis_gate.gate.hard_mask.dtype, device=basis_gate.gate.hard_mask.device)
        before = int((basis_gate.gate.hard_mask.detach() > 0.0).sum().item())
        basis_gate.gate.hard_mask.copy_(mask)
        after = int(state.active_basis.sum().item())
        applied["basis_gate"] = {"before": before, "after": after, "pruned": before - after}

    evidence = _first_evidence(model)
    if hasattr(evidence, "w_pos_hard_mask") and hasattr(evidence, "w_neg_hard_mask"):
        before_pos = int((evidence.w_pos_hard_mask.detach() > 0.0).sum().item())
        before_neg = int((evidence.w_neg_hard_mask.detach() > 0.0).sum().item())
        evidence.w_pos_hard_mask.copy_(
            state.active_pos_edges.to(dtype=evidence.w_pos_hard_mask.dtype, device=evidence.w_pos_hard_mask.device)
        )
        evidence.w_neg_hard_mask.copy_(
            state.active_neg_edges.to(dtype=evidence.w_neg_hard_mask.dtype, device=evidence.w_neg_hard_mask.device)
        )
        after_pos = int(state.active_pos_edges.sum().item())
        after_neg = int(state.active_neg_edges.sum().item())
        applied["first_evidence"] = {
            "positive": {"before": before_pos, "after": after_pos, "pruned": before_pos - after_pos},
            "negative": {"before": before_neg, "after": after_neg, "pruned": before_neg - after_neg},
            "total": {
                "before": before_pos + before_neg,
                "after": after_pos + after_neg,
                "pruned": (before_pos + before_neg) - (after_pos + after_neg),
            },
        }

    return applied
