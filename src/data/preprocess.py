from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .dataset_spec import DatasetBundle


_VALID_METHODS = {
    "identity_nonnegative",
    "minmax",
    "robust_minmax",
}


@dataclass(frozen=True)
class PreprocessConfig:
    method: str = "robust_minmax"
    feature_range: tuple[float, float] = (0.0, 1.0)
    clip: bool = True
    eps: float = 1e-12
    q_low: float = 0.01
    q_high: float = 0.99
    allow_zero: bool = True

    def __post_init__(self) -> None:
        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"Unsupported preprocessing method {self.method!r}."
            )
        low, high = self.feature_range
        if not np.isfinite([low, high]).all() or high <= low or low < 0.0:
            raise ValueError(
                "feature_range must be finite, nonnegative and increasing."
            )
        if not self.allow_zero and low <= 0.0:
            raise ValueError(
                "allow_zero=False requires feature_range[0] > 0."
            )
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")
        if not 0.0 <= self.q_low < self.q_high <= 1.0:
            raise ValueError("Require 0 <= q_low < q_high <= 1.")
        if not isinstance(self.clip, bool):
            raise TypeError("clip must be a bool.")


@dataclass(frozen=True)
class PreprocessReport:
    input_min: np.ndarray
    input_max: np.ndarray
    output_min: np.ndarray
    output_max: np.ndarray
    clipped_low_count: np.ndarray
    clipped_high_count: np.ndarray
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_min": self.input_min.tolist(),
            "input_max": self.input_max.tolist(),
            "output_min": self.output_min.tolist(),
            "output_max": self.output_max.tolist(),
            "clipped_low_count": self.clipped_low_count.tolist(),
            "clipped_high_count": self.clipped_high_count.tolist(),
            "metadata": dict(self.metadata),
        }


class BasePreprocessor(ABC):
    method: str

    def __init__(self) -> None:
        self.is_fitted = False
        self.n_features_: int | None = None
        self.last_report: PreprocessReport | None = None

    @abstractmethod
    def fit(self, bundle: DatasetBundle) -> BasePreprocessor:
        raise NotImplementedError

    @abstractmethod
    def transform(self, bundle: DatasetBundle) -> DatasetBundle:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        raise NotImplementedError

    def fit_transform(self, bundle: DatasetBundle) -> DatasetBundle:
        return self.fit(bundle).transform(bundle)

    def _mark_fitted(self, X: np.ndarray) -> None:
        self.n_features_ = int(X.shape[1])
        self.is_fitted = True

    def _checked_X(self, bundle: DatasetBundle) -> np.ndarray:
        if not self.is_fitted or self.n_features_ is None:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before transform()."
            )
        X = _as_2d_float_array(bundle.X)
        if X.shape[1] != self.n_features_:
            raise ValueError(
                "Feature count differs from the fitted preprocessor."
            )
        return X

    def _copy(
        self,
        bundle: DatasetBundle,
        X: np.ndarray,
        metadata: Mapping[str, Any],
    ) -> DatasetBundle:
        merged = dict(bundle.metadata)
        merged.update(dict(metadata))
        return DatasetBundle(
            X=X,
            y=np.asarray(bundle.y),
            task=bundle.task,
            name=bundle.name,
            feature_names=bundle.feature_names,
            target_names=bundle.target_names,
            metadata=merged,
        )


class IdentityNonnegativePreprocessor(BasePreprocessor):
    method = "identity_nonnegative"

    def fit(self, bundle: DatasetBundle) -> IdentityNonnegativePreprocessor:
        X = _as_2d_float_array(bundle.X)
        _check_nonnegative(X)
        self._mark_fitted(X)
        return self

    def transform(self, bundle: DatasetBundle) -> DatasetBundle:
        X = self._checked_X(bundle)
        _check_nonnegative(X)
        zeros = np.zeros(X.shape[1], dtype=np.int64)
        self.last_report = _make_report(
            X,
            X,
            zeros,
            zeros,
            {"method": self.method},
        )
        return self._copy(
            bundle,
            X.copy(),
            {"preprocess_method": self.method},
        )

    def get_state(self) -> dict[str, Any]:
        self._checked_state()
        return {
            "schema_version": 1,
            "method": self.method,
            "n_features": self.n_features_,
        }

    def _checked_state(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor is not fitted.")


class _BoundedPreprocessor(BasePreprocessor):
    lower_: np.ndarray | None
    upper_: np.ndarray | None

    def __init__(self, config: PreprocessConfig) -> None:
        super().__init__()
        self.config = config
        self.lower_ = None
        self.upper_ = None

    def transform(self, bundle: DatasetBundle) -> DatasetBundle:
        X = self._checked_X(bundle)
        assert self.lower_ is not None
        assert self.upper_ is not None
        output, clipped_low, clipped_high = _scale_by_bounds(
            X,
            self.lower_,
            self.upper_,
            feature_range=self.config.feature_range,
            clip=self.config.clip,
            eps=self.config.eps,
        )
        _check_nonnegative(output)
        metadata = {
            "method": self.method,
            "feature_range": self.config.feature_range,
            "clip": self.config.clip,
            "eps": self.config.eps,
        }
        if self.method == "robust_minmax":
            metadata.update(
                {
                    "q_low": self.config.q_low,
                    "q_high": self.config.q_high,
                }
            )
        self.last_report = _make_report(
            X,
            output,
            clipped_low,
            clipped_high,
            metadata,
        )
        return self._copy(
            bundle,
            output,
            {
                "preprocess_method": self.method,
                "preprocess_feature_range": self.config.feature_range,
                "preprocess_clip": self.config.clip,
            },
        )

    def get_state(self) -> dict[str, Any]:
        if (
            not self.is_fitted
            or self.lower_ is None
            or self.upper_ is None
        ):
            raise RuntimeError("Preprocessor is not fitted.")
        return {
            "schema_version": 1,
            "method": self.method,
            "n_features": self.n_features_,
            "feature_range": list(self.config.feature_range),
            "clip": self.config.clip,
            "eps": self.config.eps,
            "q_low": self.config.q_low,
            "q_high": self.config.q_high,
            "lower": self.lower_.tolist(),
            "upper": self.upper_.tolist(),
        }


class MinMaxPreprocessor(_BoundedPreprocessor):
    method = "minmax"

    def __init__(
        self,
        *,
        feature_range: tuple[float, float] = (0.0, 1.0),
        clip: bool = True,
        eps: float = 1e-12,
        allow_zero: bool = True,
    ) -> None:
        super().__init__(
            PreprocessConfig(
                method=self.method,
                feature_range=feature_range,
                clip=clip,
                eps=eps,
                allow_zero=allow_zero,
            )
        )

    def fit(self, bundle: DatasetBundle) -> MinMaxPreprocessor:
        X = _as_2d_float_array(bundle.X)
        self.lower_ = X.min(axis=0)
        self.upper_ = X.max(axis=0)
        self._mark_fitted(X)
        return self


class RobustMinMaxPreprocessor(_BoundedPreprocessor):
    method = "robust_minmax"

    def __init__(
        self,
        *,
        q_low: float = 0.01,
        q_high: float = 0.99,
        feature_range: tuple[float, float] = (0.0, 1.0),
        clip: bool = True,
        eps: float = 1e-12,
        allow_zero: bool = True,
    ) -> None:
        super().__init__(
            PreprocessConfig(
                method=self.method,
                feature_range=feature_range,
                clip=clip,
                eps=eps,
                q_low=q_low,
                q_high=q_high,
                allow_zero=allow_zero,
            )
        )

    def fit(self, bundle: DatasetBundle) -> RobustMinMaxPreprocessor:
        X = _as_2d_float_array(bundle.X)
        self.lower_ = np.quantile(X, self.config.q_low, axis=0)
        self.upper_ = np.quantile(X, self.config.q_high, axis=0)
        self._mark_fitted(X)
        return self


def make_preprocessor(config: PreprocessConfig) -> BasePreprocessor:
    if not isinstance(config, PreprocessConfig):
        raise TypeError("config must be a PreprocessConfig.")
    if config.method == "identity_nonnegative":
        return IdentityNonnegativePreprocessor()
    common = {
        "feature_range": config.feature_range,
        "clip": config.clip,
        "eps": config.eps,
        "allow_zero": config.allow_zero,
    }
    if config.method == "minmax":
        return MinMaxPreprocessor(**common)
    return RobustMinMaxPreprocessor(
        q_low=config.q_low,
        q_high=config.q_high,
        **common,
    )


def preprocessor_from_state(
    state: Mapping[str, Any],
) -> BasePreprocessor:
    """Reconstruct a fitted preprocessor from a saved JSON-compatible state."""

    if not isinstance(state, Mapping):
        raise TypeError("state must be a mapping.")
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported preprocessor state schema.")
    method = str(state.get("method"))
    n_features = int(state["n_features"])
    if method == "identity_nonnegative":
        result: BasePreprocessor = IdentityNonnegativePreprocessor()
        result.n_features_ = n_features
        result.is_fitted = True
        return result
    config = PreprocessConfig(
        method=method,
        feature_range=tuple(float(v) for v in state["feature_range"]),
        clip=bool(state["clip"]),
        eps=float(state["eps"]),
        q_low=float(state.get("q_low", 0.01)),
        q_high=float(state.get("q_high", 0.99)),
    )
    result = make_preprocessor(config)
    assert isinstance(result, _BoundedPreprocessor)
    result.lower_ = np.asarray(state["lower"], dtype=np.float32)
    result.upper_ = np.asarray(state["upper"], dtype=np.float32)
    if result.lower_.shape != (n_features,) or result.upper_.shape != (
        n_features,
    ):
        raise ValueError("Saved preprocessing bounds have invalid shape.")
    result.n_features_ = n_features
    result.is_fitted = True
    return result


def _as_2d_float_array(value: Any) -> np.ndarray:
    X = np.asarray(value, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional array.")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or Inf.")
    return X


def _check_nonnegative(X: np.ndarray) -> None:
    if float(X.min()) < 0.0:
        raise ValueError(
            "CRN input concentrations must be nonnegative."
        )


def _scale_by_bounds(
    X: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    feature_range: tuple[float, float],
    clip: bool,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    range_low, range_high = feature_range
    denominator = np.maximum(upper - lower, eps)
    scaled = range_low + (X - lower) / denominator * (
        range_high - range_low
    )
    clipped_low = (scaled < range_low).sum(axis=0).astype(np.int64)
    clipped_high = (scaled > range_high).sum(axis=0).astype(np.int64)
    if clip:
        scaled = np.clip(scaled, range_low, range_high)
    return (
        scaled.astype(np.float32),
        clipped_low,
        clipped_high,
    )


def _make_report(
    X_input: np.ndarray,
    X_output: np.ndarray,
    clipped_low: np.ndarray,
    clipped_high: np.ndarray,
    metadata: Mapping[str, Any],
) -> PreprocessReport:
    return PreprocessReport(
        input_min=X_input.min(axis=0),
        input_max=X_input.max(axis=0),
        output_min=X_output.min(axis=0),
        output_max=X_output.max(axis=0),
        clipped_low_count=clipped_low,
        clipped_high_count=clipped_high,
        metadata=dict(metadata),
    )


__all__ = [
    "BasePreprocessor",
    "IdentityNonnegativePreprocessor",
    "MinMaxPreprocessor",
    "PreprocessConfig",
    "PreprocessReport",
    "RobustMinMaxPreprocessor",
    "make_preprocessor",
    "preprocessor_from_state",
]
