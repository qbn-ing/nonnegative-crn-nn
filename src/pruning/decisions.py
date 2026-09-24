from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PruningDecision:
    stage: str
    remove_inputs: set[int] = field(default_factory=set)
    remove_basis: set[int] = field(default_factory=set)
    remove_pos_edges: set[tuple[int, int]] = field(default_factory=set)
    remove_neg_edges: set[tuple[int, int]] = field(default_factory=set)
    reason: str = "threshold"

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "reason": self.reason,
            "remove_inputs": sorted(self.remove_inputs),
            "remove_basis": sorted(self.remove_basis),
            "remove_pos_edges": sorted(list(self.remove_pos_edges)),
            "remove_neg_edges": sorted(list(self.remove_neg_edges)),
            "n_remove_inputs": len(self.remove_inputs),
            "n_remove_basis": len(self.remove_basis),
            "n_remove_pos_edges": len(self.remove_pos_edges),
            "n_remove_neg_edges": len(self.remove_neg_edges),
        }