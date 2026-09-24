from __future__ import annotations

import torch
import torch.nn as nn

from models.positive import PositiveParam


def check_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int. Got {type(value).__name__}.")

    if value <= 0:
        raise ValueError(f"{name} must be positive. Got {value}.")


def check_threshold(threshold: float) -> None:
    if threshold < 0.0:
        raise ValueError(f"threshold must be non-negative. Got {threshold}.")


class GateParam(nn.Module):
    """Positive gate with an optional hard-pruning mask.

    The trainable value is produced by ``PositiveParam``. Hard pruning is kept as
    a non-trainable 0/1 buffer so that a pruned gate is exactly zero in forward
    passes and in exported CRN specifications.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        *,
        init: float | list[float] | torch.Tensor = 1.0,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        self.param = PositiveParam(
            shape=shape,
            init=init,
            eps=eps,
            dtype=dtype,
            device=device,
        )
        self.register_buffer(
            "hard_mask",
            torch.ones(tuple(self.param.raw.shape), dtype=dtype, device=device),
        )

    def forward(self) -> torch.Tensor:
        return self.param() * self.hard_mask

    def regularization_loss(self) -> torch.Tensor:
        return self().sum()  # L1 over currently active gates.

    @torch.no_grad()
    def active_mask(self, threshold: float = 1e-3) -> torch.Tensor:
        check_threshold(threshold)
        return self().detach() > threshold

    @torch.no_grad()
    def apply_hard_pruning(
        self,
        *,
        threshold: float = 1e-3,
        keep_at_least_one: bool = True,
    ) -> dict[str, int]:
        """Threshold gate values and update the hard 0/1 mask in-place."""

        check_threshold(threshold)
        current = self().detach()
        old_active = self.hard_mask.detach() > 0.0
        new_active = current > threshold

        if keep_at_least_one and not bool(new_active.any()):
            flat_index = int(torch.argmax(current.reshape(-1)).item())
            new_active = torch.zeros_like(current, dtype=torch.bool)
            new_active.reshape(-1)[flat_index] = True

        self.hard_mask.copy_(new_active.to(dtype=self.hard_mask.dtype))

        return {
            "before": int(old_active.sum().item()),
            "after": int(new_active.sum().item()),
            "pruned": int(old_active.sum().item() - new_active.sum().item()),
            "total": int(new_active.numel()),
        }

    @torch.no_grad()
    def reset_hard_pruning(self) -> None:
        self.hard_mask.fill_(1.0)

    def extra_repr(self) -> str:
        return f"shape={tuple(self.param.raw.shape)}"


class InputGate(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        *,
        init: float | list[float] | torch.Tensor = 1.0,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        check_positive_int("n_inputs", n_inputs)

        self.n_inputs = n_inputs
        self.gate = GateParam(
            shape=(n_inputs,),
            init=init,
            eps=eps,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                "InputGate expects x with shape (batch, n_inputs). "
                f"Got shape {tuple(x.shape)}."
            )

        if x.shape[1] != self.n_inputs:
            raise ValueError(
                f"Input feature dimension mismatch: expected {self.n_inputs}, "
                f"got {x.shape[1]}."
            )

        return x * self.gate()

    def regularization_loss(self) -> torch.Tensor:
        return self.gate.regularization_loss()

    @torch.no_grad()
    def active_mask(self, threshold: float = 1e-3) -> torch.Tensor:
        return self.gate.active_mask(threshold)

    @torch.no_grad()
    def apply_hard_pruning(
        self,
        *,
        threshold: float = 1e-3,
        keep_at_least_one: bool = True,
    ) -> dict[str, int]:
        return self.gate.apply_hard_pruning(
            threshold=threshold,
            keep_at_least_one=keep_at_least_one,
        )

    @torch.no_grad()
    def reset_hard_pruning(self) -> None:
        self.gate.reset_hard_pruning()


class FeatureGate(nn.Module):
    def __init__(
        self,
        n_features: int,
        *,
        init: float | list[float] | torch.Tensor = 1.0,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        check_positive_int("n_features", n_features)

        self.n_features = n_features
        self.gate = GateParam(
            shape=(n_features,),
            init=init,
            eps=eps,
            dtype=dtype,
            device=device,
        )

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        if phi.ndim != 2:
            raise ValueError(
                "FeatureGate expects phi with shape (batch, n_features). "
                f"Got shape {tuple(phi.shape)}."
            )

        if phi.shape[1] != self.n_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.n_features}, "
                f"got {phi.shape[1]}."
            )

        return phi * self.gate()

    def regularization_loss(self) -> torch.Tensor:
        return self.gate.regularization_loss()

    @torch.no_grad()
    def active_mask(self, threshold: float = 1e-3) -> torch.Tensor:
        return self.gate.active_mask(threshold)

    @torch.no_grad()
    def apply_hard_pruning(
        self,
        *,
        threshold: float = 1e-3,
        keep_at_least_one: bool = True,
    ) -> dict[str, int]:
        return self.gate.apply_hard_pruning(
            threshold=threshold,
            keep_at_least_one=keep_at_least_one,
        )

    @torch.no_grad()
    def reset_hard_pruning(self) -> None:
        self.gate.reset_hard_pruning()


class EdgeGate(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        init: float | list[float] | torch.Tensor = 1.0,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        check_positive_int("n_in", n_in)
        check_positive_int("n_out", n_out)

        self.n_in = n_in
        self.n_out = n_out

        self.gate = GateParam(
            shape=(n_in, n_out),
            init=init,
            eps=eps,
            dtype=dtype,
            device=device,
        )

    def forward(self) -> torch.Tensor:
        return self.gate()

    def apply(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.shape != (self.n_in, self.n_out):
            raise ValueError(
                f"EdgeGate expected weight shape {(self.n_in, self.n_out)}, "
                f"got {tuple(weight.shape)}."
            )

        return weight * self.gate()

    def regularization_loss(self) -> torch.Tensor:
        return self.gate.regularization_loss()

    @torch.no_grad()
    def active_mask(self, threshold: float = 1e-3) -> torch.Tensor:
        return self.gate.active_mask(threshold)

    @torch.no_grad()
    def apply_hard_pruning(
        self,
        *,
        threshold: float = 1e-3,
        keep_at_least_one: bool = True,
    ) -> dict[str, int]:
        return self.gate.apply_hard_pruning(
            threshold=threshold,
            keep_at_least_one=keep_at_least_one,
        )

    @torch.no_grad()
    def reset_hard_pruning(self) -> None:
        self.gate.reset_hard_pruning()
