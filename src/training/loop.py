from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from utils.validation import (
    check_nonnegative_float,
    check_nonnegative_int,
    check_positive_int,
)

from .diagnostics import (
    classification_output_diagnostics,
    gradient_diagnostics,
)
from .constraints import (
    apply_parameter_constraints_,
    iter_exact_nonnegative_parameters,
)
from .early_stopping import EarlyStoppingConfig, EarlyStoppingResult, EarlyStoppingTracker
from .losses import (
    concentration_probs,
    output_to_concentrations,
    total_classification_loss,
)
from .metrics import classification_metrics
from .progress import (
    ProgressSetting,
    progress_iter,
    set_progress_postfix,
)


@dataclass(frozen=True)
class EpochResult:
    """Aggregated metrics from one train or evaluation epoch."""

    n_batches: int
    n_samples: int
    metrics: dict[str, float | int | bool | None]

    @property
    def loss(self) -> float:
        return float(self.metrics["total"])

    @property
    def classification(self) -> float:
        return float(self.metrics["classification"])

    @property
    def regularization(self) -> float:
        return float(self.metrics.get("regularization", 0.0))

    @property
    def accuracy(self) -> float | None:
        value = self.metrics.get("accuracy", self.metrics.get("acc"))
        return None if value is None else float(value)


@dataclass(frozen=True)
class FitHistory:
    """Training and optional validation history."""

    train: list[EpochResult]
    validation: list[EpochResult | None]
    early_stopping: EarlyStoppingResult | None = None

    @property
    def best_epoch(self) -> int | None:
        if self.early_stopping is None:
            return None
        return self.early_stopping.best_epoch

    @property
    def stopped_early(self) -> bool:
        return bool(
            self.early_stopping is not None
            and self.early_stopping.stopped_early
        )


EpochCompletedCallback = Callable[
    [int, EpochResult, EpochResult | None],
    None,
]


@dataclass(frozen=True)
class OptimizerStepResult:
    """Audit payload emitted after one completed optimizer update."""

    global_step: int
    batch_size: int
    learning_rates: tuple[float, ...]
    loss_metrics: dict[str, float]


StepStartedCallback = Callable[
    [int, torch.optim.Optimizer],
    None,
]
TrainingTargetTransform = Callable[
    [Any, torch.Tensor, int],
    torch.Tensor,
]
StepCompletedCallback = Callable[
    [OptimizerStepResult, torch.nn.Module, torch.optim.Optimizer],
    None,
]
EvaluationBatchCallback = Callable[
    [Any, torch.Tensor, Any],
    None,
]


class CyclingBatchStream:
    """Keep one dataloader stream continuous across optimizer stages.

    Rebuilding AdamW after a depth insertion must not implicitly restart the
    minibatch order.  Passing one ``CyclingBatchStream`` to successive
    :func:`train_steps` calls preserves the iterator and starts a new loader
    epoch only when the current one is exhausted.
    """

    def __init__(self, dataloader: Any) -> None:
        self.dataloader = dataloader
        self._iterator = iter(dataloader)
        self.cycles_completed = 0
        self.batches_yielded = 0

    def __iter__(self) -> CyclingBatchStream:
        return self

    def __next__(self) -> Any:
        try:
            batch = next(self._iterator)
        except StopIteration:
            self.cycles_completed += 1
            self._iterator = iter(self.dataloader)
            try:
                batch = next(self._iterator)
            except StopIteration as exc:
                raise ValueError(
                    "dataloader must yield at least one batch."
                ) from exc
        self.batches_yielded += 1
        return batch


def unpack_batch(
    batch: Any,
    *,
    input_keys: Sequence[str] = ("x", "inputs", "features", "data"),
    target_keys: Sequence[str] = ("y", "target", "targets", "label", "labels"),
) -> tuple[Any, torch.Tensor]:
    """Extract ``(inputs, target)`` from a dataloader batch.

    Supported forms:

    - ``(x, y)`` or ``[x, y]``;
    - mapping with configurable input and target keys;
    - object with ``x``/``y`` or ``inputs``/``target`` attributes.

    The function intentionally does not assume a project-specific batch class.
    """

    if isinstance(batch, Mapping):
        x = _first_mapping_value(batch, input_keys)
        y = _first_mapping_value(batch, target_keys)

        if x is None:
            raise KeyError(
                "Could not find input tensor in batch mapping. "
                f"Expected one of {tuple(input_keys)}."
            )

        if y is None:
            raise KeyError(
                "Could not find target tensor in batch mapping. "
                f"Expected one of {tuple(target_keys)}."
            )

        return x, y

    if isinstance(batch, (tuple, list)):
        if len(batch) != 2:
            raise ValueError(
                "Tuple/list batch must have exactly two elements: (x, y). "
                f"Got length {len(batch)}."
            )

        return batch[0], batch[1]

    x = _first_attr_value(batch, input_keys)
    y = _first_attr_value(batch, target_keys)

    if x is not None and y is not None:
        return x, y

    raise TypeError(
        "Unsupported batch format. Expected (x, y), a mapping with input/target "
        "keys, or an object with input/target attributes."
    )


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: Any,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str | None = None,
    loss_kwargs: dict[str, Any] | None = None,
    max_grad_norm: float | None = None,
    collect_output_diagnostics: bool = True,
    collect_gradient_diagnostics: bool = True,
    collect_classification_metrics: bool = True,
    grad_small_threshold: float = 1e-12,
    grad_large_threshold: float = 1e3,
    progress: ProgressSetting = "auto",
    progress_desc: str | None = None,
) -> EpochResult:
    """Train ``model`` for one epoch.

    The loop is classification-oriented and uses concentration-based loss.
    It is deliberately model-agnostic: ``model(x)`` may return a tensor, a
    state object with ``Z``/``scores``, or a dict accepted by
    ``total_classification_loss``.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"model must be a torch.nn.Module. Got {type(model).__name__}.")

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError(
            "optimizer must be a torch.optim.Optimizer. "
            f"Got {type(optimizer).__name__}."
        )

    if max_grad_norm is not None:
        check_nonnegative_float("max_grad_norm", max_grad_norm)

    check_nonnegative_float("grad_small_threshold", grad_small_threshold)
    check_nonnegative_float("grad_large_threshold", grad_large_threshold)

    loss_kwargs = {} if loss_kwargs is None else dict(loss_kwargs)

    if device is not None:
        model.to(device)

    model.train()

    accumulator = _EpochAccumulator()
    metric_eps = float(loss_kwargs.get("eps", 1e-8))
    metric_check_nonnegative = bool(loss_kwargs.get("check_nonnegative", True))

    batches = progress_iter(
        dataloader,
        progress=progress,
        total=_safe_len(dataloader),
        desc=progress_desc or "Train",
        unit="batch",
        leave=False,
    )
    for batch in batches:
        x, target = unpack_batch(batch)
        x = _move_to_device(x, device)
        target = _move_to_device(target, device)

        batch_size = _infer_batch_size(target)

        optimizer.zero_grad(set_to_none=True)

        output = model(x)

        breakdown = total_classification_loss(
            output,
            target,
            model=model,
            **loss_kwargs,
        )

        breakdown.total.backward()

        if collect_gradient_diagnostics:
            grad_stats = gradient_diagnostics(
                model,
                small_threshold=grad_small_threshold,
                large_threshold=grad_large_threshold,
            )
            accumulator.update_gradient(grad_stats)

        if max_grad_norm is not None:
            clipped_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
            )
            accumulator.update_scalar(
                "pre_clip_grad_l2_norm",
                float(clipped_norm.detach().cpu()),
                weight=1,
            )

        optimizer.step()
        projection = _constraint_projection_diagnostics(model)
        apply_parameter_constraints_(model)
        _update_projection_metrics(accumulator, projection)

        accumulator.update_loss_dict(
            breakdown.to_detached_dict(),
            batch_size=batch_size,
        )

        with torch.no_grad():
            if collect_output_diagnostics:
                output_stats = classification_output_diagnostics(output, target)
                accumulator.update_output(output_stats, batch_size=batch_size)

            if collect_classification_metrics:
                accumulator.update_classification_metrics(
                    output,
                    target,
                    eps=metric_eps,
                    check_nonnegative=metric_check_nonnegative,
                )

    return accumulator.result()


def train_steps(
    model: torch.nn.Module,
    dataloader: Any,
    optimizer: torch.optim.Optimizer,
    *,
    steps: int,
    device: torch.device | str | None = None,
    loss_kwargs: dict[str, Any] | None = None,
    max_grad_norm: float | None = None,
    collect_output_diagnostics: bool = True,
    collect_gradient_diagnostics: bool = True,
    collect_classification_metrics: bool = True,
    grad_small_threshold: float = 1e-12,
    grad_large_threshold: float = 1e3,
    progress: ProgressSetting = "auto",
    progress_desc: str | None = None,
    global_step_start: int = 0,
    on_step_started: StepStartedCallback | None = None,
    training_target_transform: TrainingTargetTransform | None = None,
    on_step_completed: StepCompletedCallback | None = None,
) -> EpochResult:
    """Train for exactly ``steps`` optimizer updates.

    The dataloader is restarted whenever it is exhausted.  This gives
    continuation schedules a budget measured in optimizer steps rather than
    epochs, while retaining the same loss, diagnostics, gradient clipping, and
    nonnegative-constraint handling used by :func:`train_one_epoch`.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"model must be a torch.nn.Module. Got {type(model).__name__}."
        )
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError(
            "optimizer must be a torch.optim.Optimizer. "
            f"Got {type(optimizer).__name__}."
        )
    check_positive_int("steps", steps)
    check_nonnegative_int("global_step_start", global_step_start)
    for name, callback in (
        ("on_step_started", on_step_started),
        ("training_target_transform", training_target_transform),
        ("on_step_completed", on_step_completed),
    ):
        if callback is not None and not callable(callback):
            raise TypeError(f"{name} must be callable or None.")
    if max_grad_norm is not None:
        check_nonnegative_float("max_grad_norm", max_grad_norm)
    check_nonnegative_float("grad_small_threshold", grad_small_threshold)
    check_nonnegative_float("grad_large_threshold", grad_large_threshold)

    loss_kwargs = {} if loss_kwargs is None else dict(loss_kwargs)
    if device is not None:
        model.to(device)
    model.train()

    accumulator = _EpochAccumulator()
    metric_eps = float(loss_kwargs.get("eps", 1e-8))
    metric_check_nonnegative = bool(
        loss_kwargs.get("check_nonnegative", True)
    )
    iterator = iter(dataloader)

    step_iterator = progress_iter(
        range(steps),
        progress=progress,
        total=steps,
        desc=progress_desc or "Train",
        unit="step",
        leave=False,
    )
    for local_step in step_iterator:
        absolute_step = global_step_start + local_step
        if on_step_started is not None:
            on_step_started(absolute_step, optimizer)
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise ValueError(
                    "dataloader must yield at least one batch."
                ) from exc

        x, target = unpack_batch(batch)
        x = _move_to_device(x, device)
        target = _move_to_device(target, device)
        batch_size = _infer_batch_size(target)
        training_target = (
            target
            if training_target_transform is None
            else training_target_transform(x, target, absolute_step)
        )
        if not isinstance(training_target, torch.Tensor):
            raise TypeError(
                "training_target_transform must return a torch.Tensor."
            )
        training_target = _move_to_device(training_target, device)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        breakdown = total_classification_loss(
            output,
            training_target,
            model=model,
            **loss_kwargs,
        )
        breakdown.total.backward()

        if collect_gradient_diagnostics:
            grad_stats = gradient_diagnostics(
                model,
                small_threshold=grad_small_threshold,
                large_threshold=grad_large_threshold,
            )
            accumulator.update_gradient(grad_stats)

        if max_grad_norm is not None:
            clipped_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
            )
            accumulator.update_scalar(
                "pre_clip_grad_l2_norm",
                float(clipped_norm.detach().cpu()),
                weight=1,
            )

        optimizer.step()
        projection = _constraint_projection_diagnostics(model)
        apply_parameter_constraints_(model)
        _update_projection_metrics(accumulator, projection)

        detached_loss = breakdown.to_detached_dict()
        accumulator.update_loss_dict(
            detached_loss,
            batch_size=batch_size,
        )

        with torch.no_grad():
            if collect_output_diagnostics:
                output_stats = classification_output_diagnostics(
                    output,
                    target,
                )
                accumulator.update_output(
                    output_stats,
                    batch_size=batch_size,
                )
            if collect_classification_metrics:
                accumulator.update_classification_metrics(
                    output,
                    target,
                    eps=metric_eps,
                    check_nonnegative=metric_check_nonnegative,
                )

        if on_step_completed is not None:
            on_step_completed(
                OptimizerStepResult(
                    global_step=absolute_step + 1,
                    batch_size=batch_size,
                    learning_rates=tuple(
                        float(group["lr"])
                        for group in optimizer.param_groups
                    ),
                    loss_metrics=detached_loss,
                ),
                model,
                optimizer,
            )

    return accumulator.result()


@torch.no_grad()
def _constraint_projection_diagnostics(
    model: torch.nn.Module,
) -> tuple[int, int, float]:
    projected = 0
    total = 0
    most_negative = 0.0
    for parameter in iter_exact_nonnegative_parameters(model):
        values = parameter.detach()
        total += int(values.numel())
        negative = values < 0.0
        projected += int(negative.sum().item())
        if negative.any().item():
            most_negative = min(most_negative, float(values.min().item()))
    return projected, total, most_negative


def _update_projection_metrics(
    accumulator: _EpochAccumulator,
    projection: tuple[int, int, float],
) -> None:
    projected, total, most_negative = projection
    accumulator.update_scalar(
        "constraint_projection_trigger_rate",
        float(projected > 0),
        weight=1,
    )
    accumulator.update_scalar(
        "constraint_projected_elements_mean",
        float(projected),
        weight=1,
    )
    accumulator.update_max(
        "constraint_projected_elements_max",
        float(projected),
    )
    accumulator.update_scalar(
        "constraint_managed_elements",
        float(total),
        weight=1,
    )
    accumulator.update_min(
        "constraint_most_negative_pre_projection",
        most_negative,
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: Any,
    *,
    device: torch.device | str | None = None,
    loss_kwargs: dict[str, Any] | None = None,
    collect_output_diagnostics: bool = True,
    collect_classification_metrics: bool = True,
    on_batch_completed: EvaluationBatchCallback | None = None,
    progress: ProgressSetting = "auto",
    progress_desc: str | None = None,
) -> EpochResult:
    """Evaluate ``model`` for one epoch without gradient computation."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"model must be a torch.nn.Module. Got {type(model).__name__}.")

    loss_kwargs = {} if loss_kwargs is None else dict(loss_kwargs)
    if on_batch_completed is not None and not callable(on_batch_completed):
        raise TypeError("on_batch_completed must be callable or None.")

    if device is not None:
        model.to(device)

    model.eval()

    accumulator = _EpochAccumulator()
    metric_eps = float(loss_kwargs.get("eps", 1e-8))
    metric_check_nonnegative = bool(loss_kwargs.get("check_nonnegative", True))

    batches = progress_iter(
        dataloader,
        progress=progress,
        total=_safe_len(dataloader),
        desc=progress_desc or "Evaluate",
        unit="batch",
        leave=False,
    )
    for batch in batches:
        x, target = unpack_batch(batch)
        x = _move_to_device(x, device)
        target = _move_to_device(target, device)

        batch_size = _infer_batch_size(target)

        output = model(x)

        if on_batch_completed is not None:
            on_batch_completed(x, target, output)

        breakdown = total_classification_loss(
            output,
            target,
            model=model,
            **loss_kwargs,
        )

        accumulator.update_loss_dict(
            breakdown.to_detached_dict(),
            batch_size=batch_size,
        )

        if collect_output_diagnostics:
            output_stats = classification_output_diagnostics(output, target)
            accumulator.update_output(output_stats, batch_size=batch_size)

        if collect_classification_metrics:
            accumulator.update_classification_metrics(
                output,
                target,
                eps=metric_eps,
                check_nonnegative=metric_check_nonnegative,
            )

    return accumulator.result()


def fit(
    model: torch.nn.Module,
    train_loader: Any,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    val_loader: Any | None = None,
    device: torch.device | str | None = None,
    loss_kwargs: dict[str, Any] | None = None,
    max_grad_norm: float | None = None,
    collect_output_diagnostics: bool = True,
    collect_gradient_diagnostics: bool = True,
    collect_classification_metrics: bool = True,
    grad_small_threshold: float = 1e-12,
    grad_large_threshold: float = 1e3,
    early_stopping: EarlyStoppingConfig | None = None,
    progress: ProgressSetting = "auto",
    on_epoch_completed: EpochCompletedCallback | None = None,
) -> FitHistory:
    """Run a simple supervised classification training loop."""

    check_positive_int("epochs", epochs)
    if on_epoch_completed is not None and not callable(
        on_epoch_completed
    ):
        raise TypeError("on_epoch_completed must be callable or None.")

    train_history: list[EpochResult] = []
    val_history: list[EpochResult | None] = []
    early_config = early_stopping if early_stopping is not None else EarlyStoppingConfig(enabled=False)
    tracker = EarlyStoppingTracker(early_config)
    best_state_dict: dict[str, torch.Tensor] | None = None

    epoch_iterator = progress_iter(
        range(1, epochs + 1),
        progress=progress,
        total=epochs,
        desc="Training",
        unit="epoch",
        leave=True,
    )
    for epoch in epoch_iterator:
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            loss_kwargs=loss_kwargs,
            max_grad_norm=max_grad_norm,
            collect_output_diagnostics=collect_output_diagnostics,
            collect_gradient_diagnostics=collect_gradient_diagnostics,
            collect_classification_metrics=collect_classification_metrics,
            grad_small_threshold=grad_small_threshold,
            grad_large_threshold=grad_large_threshold,
            progress=False,
        )
        train_history.append(train_result)

        stop_now = False

        if val_loader is None:
            val_history.append(None)
            monitor_metrics = train_result.metrics
        else:
            val_result = evaluate(
                model,
                val_loader,
                device=device,
                loss_kwargs=loss_kwargs,
                collect_output_diagnostics=collect_output_diagnostics,
                collect_classification_metrics=collect_classification_metrics,
                progress=False,
            )
            val_history.append(val_result)
            monitor_metrics = val_result.metrics

        if early_config.enabled:
            previous_best_epoch = tracker.best_epoch
            stop_now = tracker.update(monitor_metrics, epoch=epoch)
            if (
                tracker.best_epoch != previous_best_epoch
                and early_config.restore_best
            ):
                best_state_dict = deepcopy(model.state_dict())

        postfix: dict[str, float] = {
            "train_loss": train_result.loss,
        }
        current_validation = val_history[-1]
        if current_validation is not None:
            postfix["val_loss"] = current_validation.loss
        set_progress_postfix(epoch_iterator, postfix)

        if on_epoch_completed is not None:
            on_epoch_completed(
                epoch,
                train_result,
                current_validation,
            )

        if stop_now:
            break

    if early_config.enabled and early_config.restore_best and best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return FitHistory(
        train=train_history,
        validation=val_history,
        early_stopping=tracker.result(),
    )


class _EpochAccumulator:
    def __init__(self) -> None:
        self.n_batches = 0
        self.n_samples = 0

        self._weighted_sums: dict[str, float] = {}
        self._weights: dict[str, int] = {}

        self._mins: dict[str, float] = {}
        self._maxs: dict[str, float] = {}
        self._flags: dict[str, bool] = {}

        self._y_true_chunks: list[torch.Tensor] = []
        self._y_score_chunks: list[torch.Tensor] = []

    def update_loss_dict(
        self,
        values: Mapping[str, float],
        *,
        batch_size: int,
    ) -> None:
        self.n_batches += 1
        self.n_samples += batch_size

        for key, value in values.items():
            if isinstance(value, bool):
                self.update_flag(key, value)
            elif isinstance(value, (int, float)):
                self.update_scalar(key, float(value), weight=batch_size)

    def update_output(self, stats: Any, *, batch_size: int) -> None:
        values = stats.to_dict()

        weighted_mean_keys = {
            "z_mean",
            "z_sum_mean",
            "prob_min",
            "prob_max",
            "entropy_mean",
            "predicted_margin_mean",
            "true_prob_mean",
            "true_prob_min",
            "true_margin_mean",
            "accuracy",
        }

        min_keys = {
            "z_min",
            "z_sum_min",
            "predicted_margin_min",
        }

        max_keys = {
            "z_max",
            "z_sum_max",
        }

        flag_keys = {
            "has_nonfinite_z",
            "has_negative_z",
        }

        for key in weighted_mean_keys:
            value = values.get(key)
            if value is not None:
                self.update_scalar(key, float(value), weight=batch_size)

        for key in min_keys:
            value = values.get(key)
            if value is not None:
                self.update_min(key, float(value))

        for key in max_keys:
            value = values.get(key)
            if value is not None:
                self.update_max(key, float(value))

        for key in flag_keys:
            value = values.get(key)
            if value is not None:
                self.update_flag(key, bool(value))

        self.update_scalar("output_n_outputs", int(values["n_outputs"]), weight=1)

    def update_gradient(self, stats: Any) -> None:
        values = stats.to_dict()

        mean_keys = {
            "global_l2_norm",
            "mean_abs_grad",
            "small_grad_fraction",
            "large_grad_fraction",
            "zero_grad_fraction",
        }

        max_keys = {
            "max_abs_grad",
        }

        flag_keys = {
            "has_nan_or_inf",
            "has_missing_grad",
        }

        for key in mean_keys:
            value = values.get(key)
            if value is not None:
                self.update_scalar(f"grad_{key}", float(value), weight=1)

        for key in max_keys:
            value = values.get(key)
            if value is not None:
                self.update_max(f"grad_{key}", float(value))

        for key in flag_keys:
            value = values.get(key)
            if value is not None:
                self.update_flag(f"grad_{key}", bool(value))

        self.update_scalar(
            "grad_n_trainable_parameters",
            int(values["n_trainable_parameters"]),
            weight=1,
        )
        self.update_scalar(
            "grad_n_parameters_with_grad",
            int(values["n_parameters_with_grad"]),
            weight=1,
        )

    def update_classification_metrics(
        self,
        output: Any,
        target: torch.Tensor,
        *,
        eps: float,
        check_nonnegative: bool,
    ) -> None:
        Z = output_to_concentrations(output)
        probs = concentration_probs(
            Z,
            eps=eps,
            check_nonnegative=check_nonnegative,
        )

        self._y_true_chunks.append(target.detach().cpu().reshape(-1))
        self._y_score_chunks.append(probs.detach().cpu())

    def update_scalar(self, key: str, value: float, *, weight: int) -> None:
        self._weighted_sums[key] = self._weighted_sums.get(key, 0.0) + value * weight
        self._weights[key] = self._weights.get(key, 0) + weight

    def update_min(self, key: str, value: float) -> None:
        old = self._mins.get(key)
        self._mins[key] = value if old is None else min(old, value)

    def update_max(self, key: str, value: float) -> None:
        old = self._maxs.get(key)
        self._maxs[key] = value if old is None else max(old, value)

    def update_flag(self, key: str, value: bool) -> None:
        self._flags[key] = self._flags.get(key, False) or value

    def result(self) -> EpochResult:
        if self.n_batches == 0:
            raise ValueError("Cannot build EpochResult from an empty dataloader.")

        metrics: dict[str, float | int | bool | None] = {}

        for key, total in self._weighted_sums.items():
            weight = self._weights[key]
            metrics[key] = total / weight

        metrics.update(self._mins)
        metrics.update(self._maxs)
        metrics.update(self._flags)

        if len(self._y_true_chunks) > 0:
            y_true = torch.cat(self._y_true_chunks, dim=0)
            y_score = torch.cat(self._y_score_chunks, dim=0)

            epoch_classification_metrics = classification_metrics(
                y_true=y_true,
                y_score=y_score,
            )

            for key, value in epoch_classification_metrics.items():
                metrics[key] = float(value)

            if "acc" in epoch_classification_metrics:
                metrics["accuracy"] = float(epoch_classification_metrics["acc"])

        return EpochResult(
            n_batches=self.n_batches,
            n_samples=self.n_samples,
            metrics=metrics,
        )


def _first_mapping_value(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
) -> Any | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _first_attr_value(obj: Any, names: Sequence[str]) -> Any | None:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _move_to_device(value: Any, device: torch.device | str | None) -> Any:
    if device is None:
        return value

    if isinstance(value, torch.Tensor):
        return value.to(device)

    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}

    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)

    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]

    return value


def _infer_batch_size(target: torch.Tensor) -> int:
    if not isinstance(target, torch.Tensor):
        raise TypeError(
            f"target must be a torch.Tensor after batch unpacking. "
            f"Got {type(target).__name__}."
        )

    if target.ndim == 0:
        raise ValueError("target must have a batch dimension.")

    return int(target.shape[0])


def _safe_len(value: Any) -> int | None:
    try:
        length = len(value)
    except (TypeError, AttributeError):
        return None
    return int(length)
