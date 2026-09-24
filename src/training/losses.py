from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from utils.validation import check_nonnegative_float, check_positive_float

@dataclass(frozen=True)
class RegularizationBreakdown:
    """Regularization terms used by a training objective.

    The raw terms are stored separately from their weights. This makes logs
    easier to interpret and avoids hiding whether a term is large or merely
    heavily weighted.
    """

    rate_l1: torch.Tensor
    gate_l1: torch.Tensor
    rate_l1_weight: float = 0.0
    gate_l1_weight: float = 0.0

    @property
    def weighted_rate_l1(self) -> torch.Tensor:
        return self.rate_l1_weight * self.rate_l1

    @property
    def weighted_gate_l1(self) -> torch.Tensor:
        return self.gate_l1_weight * self.gate_l1

    @property
    def total(self) -> torch.Tensor:
        return self.weighted_rate_l1 + self.weighted_gate_l1

    def to_detached_dict(self) -> dict[str, float]:
        return {
            "rate_l1": float(self.rate_l1.detach().cpu()),
            "gate_l1": float(self.gate_l1.detach().cpu()),
            "weighted_rate_l1": float(self.weighted_rate_l1.detach().cpu()),
            "weighted_gate_l1": float(self.weighted_gate_l1.detach().cpu()),
            "regularization": float(self.total.detach().cpu()),
            "rate_l1_weight": float(self.rate_l1_weight),
            "gate_l1_weight": float(self.gate_l1_weight),
        }
    

@dataclass(frozen=True)
class LossBreakdown:
    """Classification loss plus optional regularization terms."""

    total: torch.Tensor
    classification: torch.Tensor
    regularization: RegularizationBreakdown

    @property
    def rate_l1(self) -> torch.Tensor:
        return self.regularization.rate_l1

    @property
    def gate_l1(self) -> torch.Tensor:
        return self.regularization.gate_l1

    def to_detached_dict(self) -> dict[str, float]:
        out = {
            "total": float(self.total.detach().cpu()),
            "classification": float(self.classification.detach().cpu()),
        }
        out.update(self.regularization.to_detached_dict())
        return out
    
# 可以训练不同的数据
def output_to_concentrations(output: Any) -> torch.Tensor:
    """Extract a concentration matrix from a model output.

    Accepted forms are intentionally lightweight:

    1. a torch.Tensor;
    2. an object with a ``Z`` attribute, such as RationalNetworkState;
    3. a dict containing ``"Z"``;
    4. an object or dict with ``scores`` only, used as a fallback.

    ``Z`` is preferred over ``scores`` because this project treats classifier
    scores as output-pool concentrations.
    """

    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, dict):
        if "Z" in output:
            return output["Z"]
        if "scores" in output:
            return output["scores"]
        raise KeyError(
            "Model output dict must contain key 'Z' or 'scores'. "
            f"Got keys: {sorted(output.keys())}."
        )
    
    if hasattr(output, "Z"):
        return getattr(output, "Z")

    if hasattr(output, "scores"):
        return getattr(output, "scores")

    raise TypeError(
        "Cannot extract concentration output. Expected a torch.Tensor, "
        "a dict with key 'Z' or 'scores', or an object with attribute "
        "'Z' or 'scores'. "
        f"Got {type(output).__name__}."
    )

def concentration_probs(
    Z: torch.Tensor,
    *,
    eps: float = 1e-8,
    check_nonnegative: bool = True,
) -> torch.Tensor:
    """Normalize non-negative output concentrations into probabilities.

    For a batch item b and class k,

        p[b, k] = (Z[b, k] + eps) / (sum_j Z[b, j] + K * eps)

    where K is the number of classes / output channels.
    """

    _check_eps(eps)
    _check_concentration_matrix(
        Z,
        name="Z",
        check_nonnegative=check_nonnegative,
    )

    n_classes = Z.shape[1]
    denom = Z.sum(dim=1, keepdim=True) + n_classes * eps
    return (Z + eps) / denom

def concentration_nll_loss(
    Z: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-8,
    reduction: str = "mean",
    check_nonnegative: bool = True,
) -> torch.Tensor:
    """Negative log-likelihood from normalized concentration outputs."""

    probs = concentration_probs(
        Z,
        eps=eps,
        check_nonnegative=check_nonnegative,
    )

    target = _normalize_target(
        target,
        batch_size=Z.shape[0],
        n_classes=Z.shape[1],
    )

    selected = probs.gather(dim=1, index=target.view(-1, 1)).squeeze(1)
    losses = -torch.log(selected)

    return _reduce_loss(losses, reduction=reduction)


def concentration_cross_entropy_loss(
    Z: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-8,
    reduction: str = "mean",
    check_nonnegative: bool = True,
) -> torch.Tensor:
    """Cross-entropy for hard labels or probability-distribution targets.

    Hard targets have shape ``(batch,)`` (or ``(batch, 1)``) and contain
    integer class indices.  Soft targets have shape ``(batch, n_classes)``,
    floating dtype, nonnegative entries, and rows summing to one.  This keeps
    task-specific curriculum construction outside the generic loss while
    giving direct and depth-continuation runs one shared objective.
    """

    probs = concentration_probs(
        Z,
        eps=eps,
        check_nonnegative=check_nonnegative,
    )
    if not isinstance(target, torch.Tensor):
        raise TypeError(
            f"target must be a torch.Tensor. Got {type(target).__name__}."
        )
    if target.ndim == 2 and target.shape == Z.shape:
        distribution = _normalize_distribution_target(
            target,
            batch_size=Z.shape[0],
            n_classes=Z.shape[1],
            dtype=probs.dtype,
            device=probs.device,
        )
        losses = -(distribution * torch.log(probs)).sum(dim=1)
    else:
        indices = _normalize_target(
            target,
            batch_size=Z.shape[0],
            n_classes=Z.shape[1],
        )
        selected = probs.gather(
            dim=1,
            index=indices.view(-1, 1),
        ).squeeze(1)
        losses = -torch.log(selected)
    return _reduce_loss(losses, reduction=reduction)


def classification_loss(
    output: Any,
    target: torch.Tensor,
    *,
    eps: float = 1e-8,
    reduction: str = "mean",
    check_nonnegative: bool = True,
) -> torch.Tensor:
    """Classification loss from a model output or direct concentration tensor."""

    Z = output_to_concentrations(output)
    return concentration_cross_entropy_loss(
        Z,
        target,
        eps=eps,
        reduction=reduction,
        check_nonnegative=check_nonnegative,
    )

def regularization_loss(
    model: Any | None,
    *,
    rate_l1_weight: float = 0.0,
    gate_l1_weight: float = 0.0,
    rate_l1_kwargs: dict[str, Any] | None = None,
    gate_l1_kwargs: dict[str, Any] | None = None,
    reference: torch.Tensor | None = None,
) -> RegularizationBreakdown:
    """Collect optional model regularizers without binding to a model class.

    The function uses duck typing only.

    Supported methods:

    - ``model.rate_l1_loss(**rate_l1_kwargs)``
    - ``model.gate_l1_loss(**gate_l1_kwargs)``
    - ``model.regularization_loss(**gate_l1_kwargs)`` as a fallback for gate
      modules that expose a generic L1 regularizer.

    No recursive traversal is performed here. This avoids double-counting
    nested gate modules. Composite models should expose their own top-level
    ``gate_l1_loss`` later if they need structured gate regularization.
    """

    check_nonnegative_float("rate_l1_weight", rate_l1_weight)
    check_nonnegative_float("gate_l1_weight", gate_l1_weight)

    rate_l1_kwargs = {} if rate_l1_kwargs is None else dict(rate_l1_kwargs)
    gate_l1_kwargs = {} if gate_l1_kwargs is None else dict(gate_l1_kwargs)

    zero = _zero_scalar(reference=reference, model=model)

    rate_l1 = zero
    gate_l1 = zero

    if model is not None and rate_l1_weight > 0.0:
        rate_fn = getattr(model, "rate_l1_loss", None)
        if callable(rate_fn):
            rate_l1 = _as_scalar_tensor(
                "rate_l1_loss",
                rate_fn(**rate_l1_kwargs),
                reference=zero,
            )

    if model is not None and gate_l1_weight > 0.0:
        gate_fn = getattr(model, "gate_l1_loss", None)

        if callable(gate_fn):
            gate_l1 = _as_scalar_tensor(
                "gate_l1_loss",
                gate_fn(**gate_l1_kwargs),
                reference=zero,
            )
        else:
            generic_fn = getattr(model, "regularization_loss", None)
            if callable(generic_fn):
                gate_l1 = _as_scalar_tensor(
                    "regularization_loss",
                    generic_fn(**gate_l1_kwargs),
                    reference=zero,
                )

    return RegularizationBreakdown(
        rate_l1=rate_l1,
        gate_l1=gate_l1,
        rate_l1_weight=rate_l1_weight,
        gate_l1_weight=gate_l1_weight,
    )

def total_classification_loss(
    output: Any,
    target: torch.Tensor,
    *,
    model: Any | None = None,
    eps: float = 1e-8,
    reduction: str = "mean",
    check_nonnegative: bool = True,
    rate_l1_weight: float = 0.0,
    gate_l1_weight: float = 0.0,
    rate_l1_kwargs: dict[str, Any] | None = None,
    gate_l1_kwargs: dict[str, Any] | None = None,
) -> LossBreakdown:
    """Compute classification loss plus optional regularization."""

    cls = classification_loss(
        output,
        target,
        eps=eps,
        reduction=reduction,
        check_nonnegative=check_nonnegative,
    )

    reg = regularization_loss(
        model,
        rate_l1_weight=rate_l1_weight,
        gate_l1_weight=gate_l1_weight,
        rate_l1_kwargs=rate_l1_kwargs,
        gate_l1_kwargs=gate_l1_kwargs,
        reference=cls,
    )

    total = cls + reg.total

    return LossBreakdown(
        total=total,
        classification=cls,
        regularization=reg,
    )

def _check_concentration_matrix(
    value: torch.Tensor,
    *,
    name: str,
    check_nonnegative: bool,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor. Got {type(value).__name__}.")

    if value.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D tensor with shape "
            f"(batch_size, n_classes). Got shape {tuple(value.shape)}."
        )

    if value.shape[0] <= 0:
        raise ValueError(f"{name} batch dimension must be positive.")

    if value.shape[1] <= 1:
        raise ValueError(
            f"{name} must contain at least two output channels for "
            f"classification. Got shape {tuple(value.shape)}."
        )

    if not torch.all(torch.isfinite(value)).item():
        raise ValueError(f"{name} must contain only finite values.")

    if check_nonnegative and torch.any(value < 0).item():
        raise ValueError(f"{name} must be non-negative.")


def _normalize_target(
    target: torch.Tensor,
    *,
    batch_size: int,
    n_classes: int,
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

    if torch.any(target < 0).item() or torch.any(target >= n_classes).item():
        raise ValueError(
            "target contains class indices outside valid range "
            f"[0, {n_classes - 1}]."
        )

    return target


def _normalize_distribution_target(
    target: torch.Tensor,
    *,
    batch_size: int,
    n_classes: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(target, torch.Tensor):
        raise TypeError(
            f"target must be a torch.Tensor. Got {type(target).__name__}."
        )
    expected = (batch_size, n_classes)
    if tuple(target.shape) != expected:
        raise ValueError(
            "soft target must have shape (batch_size, n_classes). "
            f"Expected {expected}, got {tuple(target.shape)}."
        )
    if not target.dtype.is_floating_point:
        raise TypeError("soft target must use a floating dtype.")
    value = target.to(device=device, dtype=dtype)
    if not torch.all(torch.isfinite(value)).item():
        raise ValueError("soft target must contain only finite values.")
    if torch.any(value < 0).item():
        raise ValueError("soft target must be nonnegative.")
    row_sums = value.sum(dim=1)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("soft target rows must sum to one.")
    return value


def _reduce_loss(losses: torch.Tensor, *, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return losses.mean()

    if reduction == "sum":
        return losses.sum()

    if reduction == "none":
        return losses

    raise ValueError(
        f"Unsupported reduction: {reduction!r}. "
        "Expected 'mean', 'sum', or 'none'."
    )


def _check_eps(eps: float) -> None:
    check_positive_float("eps", eps)


def _zero_scalar(
    *,
    reference: torch.Tensor | None = None,
    model: Any | None = None,
) -> torch.Tensor:
    if isinstance(reference, torch.Tensor):
        return reference.sum() * 0.0

    if model is not None and hasattr(model, "parameters"):
        try:
            for param in model.parameters():
                return param.sum() * 0.0
        except TypeError:
            pass

    return torch.zeros((), dtype=torch.float32)


def _as_scalar_tensor(
    name: str,
    value: Any,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(
            value,
            dtype=reference.dtype,
            device=reference.device,
        )

    if value.ndim != 0:
        raise ValueError(
            f"{name} must return a scalar tensor. "
            f"Got shape {tuple(value.shape)}."
        )

    return value
