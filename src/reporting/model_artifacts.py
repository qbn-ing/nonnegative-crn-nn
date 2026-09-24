from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from utils.paths import normalize_path, validate_filename
from utils.serialization import to_jsonable, write_json
from utils.validation import check_bool

@dataclass(frozen=True)
class ModelArtifactPaths:
    """Paths produced by model artifact saving utilities."""

    output_dir: str
    state_dict_pt: str | None = None
    model_spec_json: str | None = None
    metadata_json: str | None = None
    preprocessor_state_json: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "output_dir": self.output_dir,
            "state_dict_pt": self.state_dict_pt,
            "model_spec_json": self.model_spec_json,
            "metadata_json": self.metadata_json,
            "preprocessor_state_json": self.preprocessor_state_json,
        }
    
def save_model_artifacts(
    model: torch.nn.Module,
    output_dir: str | Path,
    *,
    model_spec: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    preprocessor_state: Mapping[str, Any] | None = None,
    state_dict_filename: str = "model_state.pt",
    model_spec_filename: str = "model_spec.json",
    metadata_filename: str = "artifact_metadata.json",
    preprocessor_state_filename: str = "preprocessor_state.json",
    save_state_dict: bool = True,
    save_model_spec: bool = True,
    save_metadata: bool = True,
    save_preprocessor_state: bool = True,
) -> ModelArtifactPaths:
    """Save trained-model artifacts for later verification and export.

    This module only persists model-side artifacts. It does not lower the model
    to PySB, CRN DSL, or any reaction-level representation.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "model must be a torch.nn.Module. "
            f"Got {type(model).__name__}."
        )

    check_bool(save_state_dict, "save_state_dict")
    check_bool(save_model_spec, "save_model_spec")
    check_bool(save_metadata, "save_metadata")
    check_bool(save_preprocessor_state, "save_preprocessor_state")

    output_path = normalize_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    state_dict_filename = validate_filename(
        state_dict_filename,
        name="state_dict_filename",
    )
    model_spec_filename = validate_filename(
        model_spec_filename,
        name="model_spec_filename",
    )
    metadata_filename = validate_filename(
        metadata_filename,
        name="metadata_filename",
    )
    preprocessor_state_filename = validate_filename(
        preprocessor_state_filename,
        name="preprocessor_state_filename",
    )

    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError(
            "metadata must be a mapping or None. "
            f"Got {type(metadata).__name__}."
        )

    if preprocessor_state is not None and not isinstance(preprocessor_state, Mapping):
        raise TypeError(
            "preprocessor_state must be a mapping or None. "
            f"Got {type(preprocessor_state).__name__}."
        )

    state_dict_path: Path | None = None
    model_spec_path: Path | None = None
    metadata_path: Path | None = None
    preprocessor_state_path: Path | None = None

    if save_state_dict:
        state_dict_path = output_path / state_dict_filename
        torch.save(_cpu_state_dict(model), state_dict_path)

    if save_model_spec:
        if model_spec is None:
            model_spec = extract_model_spec(model)

        model_spec_path = write_json(
            model_spec,
            output_path / model_spec_filename,
        )
    
    if save_metadata:
        metadata_payload = make_model_artifact_metadata(
            model,
            extra=metadata,
        )
        metadata_path = write_json(
            metadata_payload,
            output_path / metadata_filename,
        )

    if save_preprocessor_state and preprocessor_state is not None:
        preprocessor_state_path = write_json(
            preprocessor_state,
            output_path / preprocessor_state_filename,
        )

    return ModelArtifactPaths(
        output_dir=output_path.as_posix(),
        state_dict_pt=None if state_dict_path is None else state_dict_path.as_posix(),
        model_spec_json=(
            None if model_spec_path is None else model_spec_path.as_posix()
        ),
        metadata_json=None if metadata_path is None else metadata_path.as_posix(),
        preprocessor_state_json=(
            None
            if preprocessor_state_path is None
            else preprocessor_state_path.as_posix()
        ),
    )

def load_model_state_dict_artifact(
    model: torch.nn.Module,
    path: str | Path,
    *,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> Any:
    """Load a saved state_dict artifact into a model.

    The return value is the object returned by ``model.load_state_dict``.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "model must be a torch.nn.Module. "
            f"Got {type(model).__name__}."
        )

    state_path = normalize_path(path)
    state = torch.load(
        state_path,
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(state, Mapping):
        raise TypeError(
            "Saved artifact must contain a state_dict mapping. "
            f"Got {type(state).__name__}."
        )

    return model.load_state_dict(state, strict=strict)

def extract_model_spec(model: Any) -> Any:
    """Extract a structured model specification from a model-like object.

    Preferred order:

    1. export_model_spec()
    2. compile_spec()
    3. export_layer_spec()
    4. structure_info()

    The ``structure_info`` fallback is weaker than a real compile spec, but it
    still preserves useful information for diagnostics.
    """

    if isinstance(model, Mapping):
        return model

    for method_name in (
        "export_model_spec",
        "compile_spec",
        "export_layer_spec",
        "structure_info",
    ):
        method = getattr(model, method_name, None)
        if callable(method):
            return method()

    raise AttributeError(
        "Could not extract a model specification. Expected one of: "
        "export_model_spec(), compile_spec(), export_layer_spec(), "
        "or structure_info()."
    )

def make_model_artifact_metadata(
    model: torch.nn.Module,
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build compact metadata for saved model artifacts."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "model must be a torch.nn.Module. "
            f"Got {type(model).__name__}."
        )

    if extra is not None and not isinstance(extra, Mapping):
        raise TypeError(
            "extra must be a mapping or None. "
            f"Got {type(extra).__name__}."
        )

    payload: dict[str, Any] = {
        "model_class": type(model).__name__,
        "model_module": type(model).__module__,
        "parameters": parameter_statistics(model),
        "state_dict": state_dict_statistics(model),
    }

    if extra is not None:
        payload["extra"] = to_jsonable(extra)

    return payload


def parameter_statistics(model: torch.nn.Module) -> dict[str, int]:
    """Count trainable and frozen parameter scalars."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "model must be a torch.nn.Module. "
            f"Got {type(model).__name__}."
        )

    total = 0
    trainable = 0
    frozen = 0
    n_parameter_tensors = 0
    n_trainable_parameter_tensors = 0
    n_frozen_parameter_tensors = 0

    for parameter in model.parameters():
        n_parameter_tensors += 1
        n_scalars = int(parameter.numel())
        total += n_scalars

        if parameter.requires_grad:
            trainable += n_scalars
            n_trainable_parameter_tensors += 1
        else:
            frozen += n_scalars
            n_frozen_parameter_tensors += 1

    return {
        "n_parameter_tensors": n_parameter_tensors,
        "n_trainable_parameter_tensors": n_trainable_parameter_tensors,
        "n_frozen_parameter_tensors": n_frozen_parameter_tensors,
        "total_parameter_scalars": total,
        "trainable_parameter_scalars": trainable,
        "frozen_parameter_scalars": frozen,
    }

def state_dict_statistics(model: torch.nn.Module) -> dict[str, Any]:
    """Summarize tensor entries in a model state_dict."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "model must be a torch.nn.Module. "
            f"Got {type(model).__name__}."
        )

    state = model.state_dict()

    entries: dict[str, dict[str, Any]] = {}
    total_numel = 0

    for key, tensor in state.items():
        if not torch.is_tensor(tensor):
            continue

        numel = int(tensor.numel())
        total_numel += numel

        entries[str(key)] = {
            "shape": [int(dim) for dim in tensor.shape],
            "dtype": str(tensor.dtype),
            "numel": numel,
        }

    return {
        "n_tensors": len(entries),
        "total_tensor_scalars": total_numel,
        "entries": entries,
    }

def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}

    for key, value in model.state_dict().items():
        if torch.is_tensor(value):
            state[key] = value.detach().cpu()
        else:
            state[key] = value

    return state