from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .losses import concentration_probs, output_to_concentrations
from utils.validation import check_nonnegative_float, check_positive_float

@dataclass(frozen=True)
class ClassificationOutputDiagnostics:
    """Detached diagnostics for concentration-based classifier outputs."""

    batch_size: int
    n_outputs: int

    z_min: float
    z_max: float
    z_mean: float
    z_sum_min: float
    z_sum_max: float
    z_sum_mean: float

    prob_min: float
    prob_max: float
    entropy_mean: float

    predicted_margin_mean: float
    predicted_margin_min: float

    true_prob_mean: float | None = None
    true_prob_min: float | None = None
    true_margin_mean: float | None = None
    accuracy: float | None = None

    has_nonfinite_z: bool = False
    has_negative_z: bool = False

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return self.__dict__.copy()
    
@dataclass(frozen=True)
class GradientDiagnostics:
    """Detached diagnostics for gradients after backward()."""

    n_parameters: int
    n_trainable_parameters: int
    n_parameters_with_grad: int
    n_gradient_elements: int

    global_l2_norm: float
    max_abs_grad: float
    mean_abs_grad: float

    small_grad_fraction: float
    large_grad_fraction: float
    zero_grad_fraction: float

    has_nan_or_inf: bool
    has_missing_grad: bool

    small_threshold: float
    large_threshold: float

    def to_dict(self) -> dict[str, float | int | bool]:
        return self.__dict__.copy()
    
def classification_output_diagnostics(
    output: Any,
    target: torch.Tensor | None = None,
    *,
    eps: float = 1e-8,
) -> ClassificationOutputDiagnostics:
    """Compute detached diagnostics for concentration outputs.

    This function is intentionally side-effect free. It does not call backward,
    does not modify tensors, and does not require a concrete model class.
    """

    check_positive_float("eps", eps)

    Z = output_to_concentrations(output)

    if not isinstance(Z, torch.Tensor):
        raise TypeError(f"Z must be a torch.Tensor. Got {type(Z).__name__}.")

    if Z.ndim != 2:
        raise ValueError(
            "Z must be a 2D tensor with shape (batch_size, n_outputs). "
            f"Got shape {tuple(Z.shape)}."
        )

    if Z.shape[0] <= 0:
        raise ValueError("Z batch dimension must be positive.")

    if Z.shape[1] <= 1:
        raise ValueError("Z must contain at least two output channels.")

    with torch.no_grad():
        z_detached = Z.detach()

        has_nonfinite_z = not torch.all(torch.isfinite(z_detached)).item()
        has_negative_z = torch.any(z_detached < 0).item()

        z_for_probs = torch.clamp(z_detached, min=0.0)

        probs = concentration_probs(
            z_for_probs,
            eps=eps,
            check_nonnegative=True,
        )

        z_sum = z_detached.sum(dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1)

        top_values = probs.topk(k=2, dim=1).values
        predicted_margin = top_values[:, 0] - top_values[:, 1]

        true_prob_mean: float | None = None
        true_prob_min: float | None = None
        true_margin_mean: float | None = None
        accuracy: float | None = None

        if target is not None:
            target = _normalize_target_for_diagnostics(
                target,
                batch_size=Z.shape[0],
                n_outputs=Z.shape[1],
            )

            true_probs = probs.gather(dim=1, index=target.view(-1, 1)).squeeze(1)

            masked_probs = probs.clone()
            masked_probs.scatter_(dim=1, index=target.view(-1, 1), value=-1.0)
            best_other = masked_probs.max(dim=1).values
            true_margin = true_probs - best_other

            pred = probs.argmax(dim=1)

            true_prob_mean = _float(true_probs.mean())
            true_prob_min = _float(true_probs.min())
            true_margin_mean = _float(true_margin.mean())
            accuracy = _float((pred == target).float().mean())

        return ClassificationOutputDiagnostics(
            batch_size=int(Z.shape[0]),
            n_outputs=int(Z.shape[1]),
            z_min=_float(z_detached.min()),
            z_max=_float(z_detached.max()),
            z_mean=_float(z_detached.mean()),
            z_sum_min=_float(z_sum.min()),
            z_sum_max=_float(z_sum.max()),
            z_sum_mean=_float(z_sum.mean()),
            prob_min=_float(probs.min()),
            prob_max=_float(probs.max()),
            entropy_mean=_float(entropy.mean()),
            predicted_margin_mean=_float(predicted_margin.mean()),
            predicted_margin_min=_float(predicted_margin.min()),
            true_prob_mean=true_prob_mean,
            true_prob_min=true_prob_min,
            true_margin_mean=true_margin_mean,
            accuracy=accuracy,
            has_nonfinite_z=bool(has_nonfinite_z),
            has_negative_z=bool(has_negative_z),
        )
    
def gradient_diagnostics(
    model: torch.nn.Module,
    *,
    small_threshold: float = 1e-12,
    large_threshold: float = 1e3,
) -> GradientDiagnostics:
    """Compute detached gradient diagnostics after backward().

    This should be called after ``loss.backward()`` and before
    ``optimizer.step()``.

    The thresholds are diagnostic thresholds only. Gradient clipping policy
    should be handled by the training loop.
    """

    check_nonnegative_float("small_threshold", small_threshold)
    check_positive_float("large_threshold", large_threshold)

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"model must be a torch.nn.Module. Got {type(model).__name__}."
        )

    n_parameters = 0
    n_trainable_parameters = 0
    n_parameters_with_grad = 0
    n_gradient_elements = 0

    sum_sq = 0.0
    sum_abs = 0.0
    max_abs = 0.0

    n_small = 0
    n_large = 0
    n_zero = 0
    has_nan_or_inf = False
    has_missing_grad = False

    for param in model.parameters():
        n_parameters += param.numel()

        if not param.requires_grad:
            continue

        n_trainable_parameters += param.numel()

        if param.grad is None:
            has_missing_grad = True
            continue

        grad = param.grad.detach()
        flat = grad.reshape(-1)
        abs_grad = flat.abs()

        n_parameters_with_grad += param.numel()
        n_gradient_elements += flat.numel()

        finite_mask = torch.isfinite(flat)
        if not torch.all(finite_mask).item():
            has_nan_or_inf = True

        finite_abs = abs_grad[finite_mask]
        if finite_abs.numel() == 0:
            continue

        sum_sq += _float((finite_abs * finite_abs).sum())
        sum_abs += _float(finite_abs.sum())
        max_abs = max(max_abs, _float(finite_abs.max()))

        n_small += int((finite_abs < small_threshold).sum().item())
        n_large += int((finite_abs > large_threshold).sum().item())
        n_zero += int((finite_abs == 0.0).sum().item())

    if n_gradient_elements == 0:
        return GradientDiagnostics(
            n_parameters=n_parameters,
            n_trainable_parameters=n_trainable_parameters,
            n_parameters_with_grad=0,
            n_gradient_elements=0,
            global_l2_norm=0.0,
            max_abs_grad=0.0,
            mean_abs_grad=0.0,
            small_grad_fraction=0.0,
            large_grad_fraction=0.0,
            zero_grad_fraction=0.0,
            has_nan_or_inf=has_nan_or_inf,
            has_missing_grad=bool(n_trainable_parameters > 0),
            small_threshold=float(small_threshold),
            large_threshold=float(large_threshold),
        )

    return GradientDiagnostics(
        n_parameters=n_parameters,
        n_trainable_parameters=n_trainable_parameters,
        n_parameters_with_grad=n_parameters_with_grad,
        n_gradient_elements=n_gradient_elements,
        global_l2_norm=float(sum_sq ** 0.5),
        max_abs_grad=float(max_abs),
        mean_abs_grad=float(sum_abs / n_gradient_elements),
        small_grad_fraction=float(n_small / n_gradient_elements),
        large_grad_fraction=float(n_large / n_gradient_elements),
        zero_grad_fraction=float(n_zero / n_gradient_elements),
        has_nan_or_inf=bool(has_nan_or_inf),
        has_missing_grad=bool(has_missing_grad),
        small_threshold=float(small_threshold),
        large_threshold=float(large_threshold),
    )


def _normalize_target_for_diagnostics(
    target: torch.Tensor,
    *,
    batch_size: int,
    n_outputs: int,
) -> torch.Tensor:
    if not isinstance(target, torch.Tensor):
        raise TypeError(
            f"target must be a torch.Tensor. Got {type(target).__name__}."
        )

    if target.ndim == 2 and target.shape[1] == 1:
        target = target.reshape(-1)

    if target.ndim != 1:
        raise ValueError(
            "target must be a 1D tensor of class indices, or a 2D tensor "
            f"with shape (batch_size, 1). Got shape {tuple(target.shape)}."
        )

    if target.shape[0] != batch_size:
        raise ValueError(
            "target and Z must have the same batch size. "
            f"Got target batch_size={target.shape[0]}, Z batch_size={batch_size}."
        )

    if target.dtype.is_floating_point:
        raise TypeError(
            "target must contain integer class indices. "
            f"Got floating dtype {target.dtype}."
        )

    target = target.to(dtype=torch.long)

    if torch.any(target < 0).item() or torch.any(target >= n_outputs).item():
        raise ValueError(
            "target contains class indices outside valid range "
            f"[0, {n_outputs - 1}]."
        )

    return target


def _float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())