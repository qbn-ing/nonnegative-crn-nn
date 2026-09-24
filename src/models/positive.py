from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

def check_eps(eps: float) -> None:
    if eps < 0.0:
        raise ValueError(f"eps must be non-negative. Got {eps}.")
    
def positive_transform(
    raw: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    check_eps(eps)
    return F.softplus(raw) + eps     # softplus(x) = log(1 + exp(x))

def inverse_positive_transform(
    value: Any,
    *,
    eps: float = 1e-8,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    check_eps(eps)

    v = torch.as_tensor(value, dtype=dtype, device=device)
    shifted = v - eps

    # 确保都是正值
    if torch.any(shifted <= 0):
        raise ValueError(
            "Initial value must be greater than eps. "
            f"Got minimum shifted value {shifted.min().item()}."
        )

    return _softplus_inverse(shifted)


def _softplus_inverse(y: torch.Tensor) -> torch.Tensor:
    return y + torch.log(-torch.expm1(-y))


class PositiveParam(nn.Module):
    def __init__(
        self,
        shape: tuple[int, ...] | int | None = None,
        *,
        init: float | list[float] | torch.Tensor = 1.0,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        check_eps(eps)

        self.eps = eps

        init_tensor = _make_init_tensor(
            shape=shape,
            init=init,
            dtype=dtype,
            device=device,
        )

        raw_init = inverse_positive_transform(
            init_tensor,
            eps=eps,
            dtype=dtype,
            device=device,
        )

        self.raw = nn.Parameter(raw_init.clone())
    
    def forward(self) -> torch.Tensor:
        return positive_transform(self.raw, eps=self.eps)

    @torch.no_grad()
    def set_value_(
        self,
        value: float | list[float] | torch.Tensor,
    ) -> PositiveParam:
        """
        Initializers operate on the constrained value represented by this
        module.  This method performs the required inverse-softplus mapping so
        callers never need to write to ``raw`` directly.
        """

        value_tensor = _make_init_tensor(
            shape=tuple(self.raw.shape),
            init=value,
            dtype=self.raw.dtype,
            device=self.raw.device,
        )
        raw_value = inverse_positive_transform(
            value_tensor,
            eps=self.eps,
            dtype=self.raw.dtype,
            device=self.raw.device,
        )
        self.raw.copy_(raw_value)
        return self

    def extra_repr(self) -> str:
        return f"shape={tuple(self.raw.shape)}, eps={self.eps}"


class ExactNonnegativeParam(nn.Module):
    """A directly optimized parameter whose feasible set includes exact zero.

    Unlike :class:`PositiveParam`, this module does not reparameterize its
    trainable tensor. ``forward()`` therefore has unit derivative, including at
    zero. It also deliberately performs no forward-time clamping: feasibility
    must be restored explicitly after an optimizer step via ``project_()`` or
    the utilities in ``training.constraints``.

    This separation keeps parameter representation independent from the
    optimizer-side boundary rule. In particular, ``project_()`` is only the
    basic Euclidean projection and does not modify optimizer state.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int | None = None,
        *,
        init: float | list[float] | torch.Tensor = 0.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        init_tensor = _make_init_tensor(
            shape=shape,
            init=init,
            dtype=dtype,
            device=device,
        )
        if not torch.is_floating_point(init_tensor):
            raise TypeError(
                "ExactNonnegativeParam requires a floating-point dtype. "
                f"Got {init_tensor.dtype}."
            )
        _check_finite_nonnegative_init(init_tensor)

        self.value = nn.Parameter(init_tensor.clone())

    def forward(self) -> torch.Tensor:
        return self.value

    @torch.no_grad()
    def set_value_(
        self,
        value: float | list[float] | torch.Tensor,
    ) -> ExactNonnegativeParam:
        """Replace the directly optimized value after validating feasibility."""

        value_tensor = _make_init_tensor(
            shape=tuple(self.value.shape),
            init=value,
            dtype=self.value.dtype,
            device=self.value.device,
        )
        _check_finite_nonnegative_init(value_tensor)
        self.value.copy_(value_tensor)
        return self

    @torch.no_grad()
    def project_(self) -> ExactNonnegativeParam:
        """Apply the Euclidean projection onto the nonnegative orthant."""

        self.value.clamp_(min=0.0)
        return self

    def extra_repr(self) -> str:
        return f"shape={tuple(self.value.shape)}"


def _check_finite_nonnegative_init(value: torch.Tensor) -> None:
    if not torch.all(torch.isfinite(value)).item():
        raise ValueError("Initial value must contain only finite values.")
    if torch.any(value < 0).item():
        raise ValueError(
            "Initial value must be nonnegative. "
            f"Got minimum value {value.min().item()}."
        )



def _make_init_tensor(
    *,
    shape: tuple[int, ...] | int | None,
    init: float | list[float] | torch.Tensor,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    if isinstance(shape, int):
        shape = (shape,)
    if shape is None:
        return torch.as_tensor(init, dtype=dtype, device=device)
    if len(shape) == 0:
        raise ValueError("shape must not be an empty tuple.")
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"All shape dimensions must be positive. Got {shape}.")
    if isinstance(init, (float, int)):
        return torch.full(
            shape,
            float(init),
            dtype=dtype,
            device=device,
        )
    init_tensor = torch.as_tensor(init, dtype=dtype, device=device)

    try:
        return torch.broadcast_to(init_tensor, shape).clone()
    except RuntimeError as exc:
        raise ValueError(
            f"init with shape {tuple(init_tensor.shape)} cannot be broadcast "
            f"to requested shape {shape}."
        ) from exc
