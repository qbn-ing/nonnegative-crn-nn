from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

import torch
import torch.nn as nn

from models.basis.polynomial import PolynomialBasisLayer
from models.basis.specs import BasisFeatureSpec, ProductionTerm, SpeciesFactor
from models.composition.basis_rational import BasisRationalModel
from models.composition.rational_network import RationalNetwork
from models.evidence.dense import DenseEvidenceLayer, DenseRationalLayer
from models.positive import ExactNonnegativeParam, PositiveParam
from models.readout.competitive import CompetitiveOutputLayer
from pruning.structural_state import make_current_structural_state


@dataclass(frozen=True)
class CompactMaterializationInfo:
    """Metadata produced when a pruned model is rebuilt into a compact core."""

    enabled: bool
    model_type: str
    original_input_dim: int
    compact_input_dim: int
    original_n_basis: int
    compact_n_basis: int
    original_first_layer_edges: int
    compact_first_layer_edges: int
    active_input_indices: tuple[int, ...]
    active_basis_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "model_type": self.model_type,
            "original_input_dim": int(self.original_input_dim),
            "compact_input_dim": int(self.compact_input_dim),
            "original_n_basis": int(self.original_n_basis),
            "compact_n_basis": int(self.compact_n_basis),
            "original_first_layer_edges": int(self.original_first_layer_edges),
            "compact_first_layer_edges": int(self.compact_first_layer_edges),
            "active_input_indices": [int(i) for i in self.active_input_indices],
            "active_basis_indices": [int(i) for i in self.active_basis_indices],
        }


class CompactBasisRationalAdapter(nn.Module):
    """Run a compact BasisRationalModel from the original full input columns.

    The compact core is built with only the surviving input coordinates and
    basis features. ``forward`` always accepts the original full-width input.
    Call ``forward_compact`` only when the caller already holds the scaled,
    compact coordinates. Export/model specs are delegated to the compact core,
    so structural counts describe the compact CRN rather than pre-pruning
    tensor shapes.
    """

    def __init__(
        self,
        *,
        core: BasisRationalModel,
        active_input_indices: Iterable[int],
        original_input_dim: int,
        original_model_name: str,
        materialization_info: CompactMaterializationInfo,
    ) -> None:
        super().__init__()
        self.core = core
        self.original_input_dim = int(original_input_dim)
        self.original_model_name = str(original_model_name)
        indices = tuple(int(i) for i in active_input_indices)
        if not indices:
            raise ValueError("active_input_indices must not be empty.")
        self.register_buffer(
            "active_input_indices_tensor",
            torch.tensor(indices, dtype=torch.long),
        )
        self.register_buffer(
            "active_input_scale_tensor",
            getattr(core, "_compact_input_scale", torch.ones(len(indices), dtype=torch.float32)).detach().clone(),
        )
        self.materialization_info = materialization_info

    @property
    def input_dim(self) -> int:
        return self.original_input_dim

    @property
    def compact_input_dim(self) -> int:
        return self.core.input_dim

    @property
    def n_basis(self) -> int:
        return self.core.n_basis

    @property
    def n_classes(self) -> int:
        return self.core.n_classes

    @property
    def n_outputs(self) -> int:
        return self.core.n_outputs

    def forward(self, x: torch.Tensor) -> Any:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError(f"Expected a 2D input tensor. Got shape {tuple(x.shape)}.")
        if x.shape[1] != self.original_input_dim:
            raise ValueError(
                "forward expects the original full input dimension "
                f"({self.original_input_dim}). Got {x.shape[1]}. "
                "Use forward_compact for preprojected compact inputs."
            )
        compact_x = self.project_full_input_tensor(x)
        return self.core(compact_x)

    def forward_compact(self, compact_x: torch.Tensor) -> Any:
        """Run the compact core on coordinates that are already projected."""

        if not isinstance(compact_x, torch.Tensor):
            raise TypeError("compact_x must be a torch.Tensor.")
        if compact_x.ndim != 2 or compact_x.shape[1] != self.core.input_dim:
            raise ValueError(
                "compact_x must have shape "
                f"(batch_size, {self.core.input_dim}). "
                f"Got {tuple(compact_x.shape)}."
            )
        return self.core(compact_x)

    def project_full_input_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Select active coordinates and fold the trained input-gate scale."""

        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor.")
        if x.ndim != 2 or x.shape[1] != self.original_input_dim:
            raise ValueError(
                "x must have shape "
                f"(batch_size, {self.original_input_dim}). "
                f"Got {tuple(x.shape)}."
            )
        indices = self.active_input_indices_tensor.to(device=x.device)
        scale = self.active_input_scale_tensor.to(
            device=x.device,
            dtype=x.dtype,
        )
        return x.index_select(dim=1, index=indices) * scale

    def rate_l1_loss(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        fn = getattr(self.core, "rate_l1_loss", None)
        if callable(fn):
            return fn(*args, **kwargs)
        return torch.zeros((), device=self.active_input_indices_tensor.device)

    def gate_l1_loss(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.zeros((), device=self.active_input_indices_tensor.device)

    def compile_spec(self) -> Any:
        return self.core.compile_spec()

    def export_model_spec(self) -> Any:
        return self.core.export_model_spec()

    def structure_info(self) -> dict[str, Any]:
        info = self.core.structure_info()
        spec = self.core.export_model_spec()
        info["n_species"] = int(spec.n_species)
        info["n_reactions"] = int(spec.n_reactions)
        info["n_evidence_species"] = int(getattr(spec, "n_evidence_species", info.get("n_evidence_species", 0)))
        info["n_response_species"] = int(getattr(spec, "n_response_species", info.get("n_response_species", 0)))
        info["n_output_species"] = int(getattr(spec, "n_output_species", info.get("n_output_species", 0)))
        info["compact_adapter"] = {
            "enabled": True,
            "original_input_dim": self.original_input_dim,
            "compact_input_dim": self.core.input_dim,
            "active_input_indices": [
                int(i) for i in self.active_input_indices_tensor.detach().cpu().tolist()
            ],
            "original_model_name": self.original_model_name,
        }
        info["materialization"] = self.materialization_info.to_dict()
        return info

    def project_full_input_values_for_export(
        self,
        input_values: Iterable[float],
    ) -> tuple[float, ...]:
        """Project raw full-width values into compact export coordinates."""

        values = tuple(float(v) for v in input_values)
        if len(values) != self.original_input_dim:
            raise ValueError(
                "input_values must have the original full input dimension "
                f"({self.original_input_dim}). Got {len(values)}."
            )
        scale = tuple(
            float(v)
            for v in self.active_input_scale_tensor.detach().cpu().tolist()
        )
        active = tuple(
            int(i)
            for i in self.active_input_indices_tensor.detach().cpu().tolist()
        )
        return tuple(values[i] * scale[j] for j, i in enumerate(active))

    def compact_input_values_for_export(
        self,
        input_values: Iterable[float],
    ) -> tuple[float, ...]:
        """Validate and return already projected compact export coordinates."""

        values = tuple(float(v) for v in input_values)
        if len(values) != self.core.input_dim:
            raise ValueError(
                "input_values must have the compact input dimension "
                f"({self.core.input_dim}). Got {len(values)}."
            )
        return values

    def project_input_values_for_export(
        self,
        input_values: Iterable[float],
    ) -> tuple[float, ...]:
        """Compatibility entry point with an explicit full-input contract."""

        return self.project_full_input_values_for_export(input_values)

    def active_input_indices_for_export(self) -> tuple[int, ...]:
        return tuple(range(self.core.input_dim))

    def active_basis_indices_for_export(self) -> tuple[int, ...]:
        return tuple(range(self.core.n_basis))


def materialize_compact_model(model: nn.Module) -> tuple[nn.Module, CompactMaterializationInfo | None]:
    """Return a compact, export-consistent model when the input is BasisRationalModel.

    The current pruning implementation uses hard masks and export-time remapping.
    That is correct for DSD export, but it leaves the PyTorch module with the
    original dense basis/evidence tensor shapes.  This function rebuilds the
    single-layer BasisRationalModel used by the current tabular experiments so summary,
    saved model spec, and CRN export agree on the compact structure.
    """

    if isinstance(model, CompactBasisRationalAdapter):
        return model, model.materialization_info

    if not isinstance(model, BasisRationalModel):
        return model, None

    if len(model.network.layers) != 1:
        raise NotImplementedError(
            "Compact materialization currently supports single-layer "
            "tabular experiments only."
        )

    layer = model.network.layers[0]
    if not isinstance(layer.evidence, DenseEvidenceLayer):
        raise NotImplementedError(
            "Compact materialization currently supports DenseEvidenceLayer only."
        )
    structural_state = make_current_structural_state(model)
    original_feature_specs = tuple(model.basis.feature_specs())

    edge_basis_indices: set[int] = set()
    if structural_state.active_pos_edges.size:
        _, pos_basis_indices = torch.nonzero(
            structural_state.active_pos_edges,
            as_tuple=True,
        )
        edge_basis_indices.update(int(j) for j in pos_basis_indices.tolist())
    if structural_state.active_neg_edges.size:
        _, neg_basis_indices = torch.nonzero(
            structural_state.active_neg_edges,
            as_tuple=True,
        )
        edge_basis_indices.update(int(j) for j in neg_basis_indices.tolist())

    gate_basis_indices = set(
        int(i) for i in torch.nonzero(
            structural_state.active_basis,
            as_tuple=True,
        )[0].tolist()
    )
    if edge_basis_indices:
        active_basis = tuple(sorted(edge_basis_indices))
    elif gate_basis_indices:
        active_basis = tuple(sorted(gate_basis_indices))
    else:
        active_basis = (0,)

    dependency_inputs = set(_basis_dependencies(original_feature_specs, active_basis))
    gate_input_indices = set(
        int(i) for i in torch.nonzero(
            structural_state.active_inputs,
            as_tuple=True,
        )[0].tolist()
    )
    active_inputs = tuple(sorted(dependency_inputs | gate_input_indices))
    if not active_inputs:
        active_inputs = (0,)

    original_input_to_new = {old: new for new, old in enumerate(active_inputs)}
    compact_feature_specs = tuple(
        _remap_feature_spec(
            original_feature_specs[old_basis_index],
            old_input_to_new=original_input_to_new,
            new_index=new_basis_index,
        )
        for new_basis_index, old_basis_index in enumerate(active_basis)
    )

    compact_basis = PolynomialBasisLayer(
        n_inputs=len(active_inputs),
        specs=compact_feature_specs,
        check_nonnegative=getattr(model.basis, "check_nonnegative", True),
    )

    with torch.no_grad():
        old_w_pos, old_w_neg = layer.evidence.effective_weights()
        old_b_pos, old_b_neg = layer.evidence.effective_biases()
        old_w_pos = old_w_pos.detach().cpu()
        old_w_neg = old_w_neg.detach().cpu()
        old_b_pos = old_b_pos.detach().cpu()
        old_b_neg = old_b_neg.detach().cpu()
        old_pos_mask = layer.evidence.w_pos_hard_mask.detach().cpu().bool()
        old_neg_mask = layer.evidence.w_neg_hard_mask.detach().cpu().bool()

        input_scale = torch.ones(len(active_inputs), dtype=old_w_pos.dtype)
        if model.input_gate is not None:
            input_scale = model.input_gate.gate().detach().cpu()[list(active_inputs)].to(dtype=old_w_pos.dtype)

        feature_scale = torch.ones(len(active_basis), dtype=old_w_pos.dtype)
        if model.feature_gate is not None:
            feature_scale = model.feature_gate.gate().detach().cpu()[list(active_basis)].to(dtype=old_w_pos.dtype)

        pos_values = old_w_pos[:, list(active_basis)] * feature_scale.reshape(1, -1)
        neg_values = old_w_neg[:, list(active_basis)] * feature_scale.reshape(1, -1)
        if layer.evidence.weight_parameterization == "positive":
            pos_values = pos_values.clamp_min(layer.evidence.w_pos.eps * 2.0)
            neg_values = neg_values.clamp_min(layer.evidence.w_neg.eps * 2.0)
        pos_mask = old_pos_mask[:, list(active_basis)].to(dtype=torch.float32)
        neg_mask = old_neg_mask[:, list(active_basis)].to(dtype=torch.float32)

    evidence_eps = float(getattr(layer.evidence.w_pos, "eps", 0.0))
    compact_evidence = DenseEvidenceLayer(
        input_dim=len(active_basis),
        output_dim=layer.evidence.output_dim,
        init_rate=0.1,
        include_bias=layer.include_bias,
        bias_init_rate=0.1,
        eps=evidence_eps,
        weight_parameterization=layer.evidence.weight_parameterization,
        name=getattr(layer.evidence, "name", getattr(layer.evidence, "_name", "dense_evidence")),
        check_nonnegative_phi=layer.evidence.check_nonnegative_phi,
        dtype=pos_values.dtype,
        device="cpu",
    )
    compact_evidence.w_pos = _nonnegative_param_like(
        layer.evidence.w_pos,
        pos_values,
    )
    compact_evidence.w_neg = _nonnegative_param_like(
        layer.evidence.w_neg,
        neg_values,
    )
    if layer.include_bias:
        assert layer.evidence.b_pos is not None
        assert layer.evidence.b_neg is not None
        compact_evidence.b_pos = _nonnegative_param_like(
            layer.evidence.b_pos,
            old_b_pos,
        )
        compact_evidence.b_neg = _nonnegative_param_like(
            layer.evidence.b_neg,
            old_b_neg,
        )
    _set_registered_buffer(
        compact_evidence,
        "w_pos_hard_mask",
        pos_mask.to(dtype=pos_values.dtype),
    )
    _set_registered_buffer(
        compact_evidence,
        "w_neg_hard_mask",
        neg_mask.to(dtype=neg_values.dtype),
    )

    rates = {
        name: value.detach().cpu()
        for name, value in layer.response_rates().items()
    }
    compact_layer = DenseRationalLayer(
        in_features=len(active_basis),
        out_features=layer.out_features,
        init_rate=0.1,
        weight_parameterization=layer.evidence.weight_parameterization,
        eta_init=rates["eta"],
        gamma_init=rates["gamma"],
        chi_init=rates["chi"],
        evidence_eps=evidence_eps,
        response_eps=layer.eta.eps,
        denom_eps=layer.denom_eps,
        name=layer.name,
        response_prefix=layer.response_prefix,
        check_nonnegative_input=layer.check_nonnegative_input,
        include_bias=layer.include_bias,
        dtype=rates["eta"].dtype,
        device="cpu",
    )
    compact_layer.evidence = compact_evidence
    compact_network = RationalNetwork(
        layers=[compact_layer],
        name=model.network.name,
        output=CompetitiveOutputLayer(
            n_classes=model.network.output_layer.n_classes,
            total=model.network.output_layer.total,
            name=model.network.output_layer.name,
            inactive_name=model.network.output_layer.inactive_name,
            active_prefix=model.network.output_layer.active_prefix,
            response_prefix=compact_layer.response_prefix,
        ),
        check_finite_input=model.network.check_finite_input,
        check_nonnegative_input=model.network.check_nonnegative_input,
    )
    compact_core = BasisRationalModel(
        basis=compact_basis,
        network=compact_network,
        name=f"{model.name}_compact",
        input_gate=None,
        feature_gate=None,
        check_finite_input=model.check_finite_input,
    )

    compact_edges = int(pos_mask.sum().item() + neg_mask.sum().item())
    original_edges = int(
        layer.evidence.output_dim * layer.evidence.input_dim * 2
    )
    compact_core.register_buffer("_compact_input_scale", input_scale.detach().clone())

    info = CompactMaterializationInfo(
        enabled=True,
        model_type=type(model).__name__,
        original_input_dim=model.input_dim,
        compact_input_dim=len(active_inputs),
        original_n_basis=model.n_basis,
        compact_n_basis=len(active_basis),
        original_first_layer_edges=original_edges,
        compact_first_layer_edges=compact_edges,
        active_input_indices=active_inputs,
        active_basis_indices=active_basis,
    )

    return CompactBasisRationalAdapter(
        core=compact_core,
        active_input_indices=active_inputs,
        original_input_dim=model.input_dim,
        original_model_name=model.name,
        materialization_info=info,
    ), info


def _set_registered_buffer(
    module: torch.nn.Module,
    name: str,
    value: torch.Tensor,
) -> None:
    """Set a tensor as a registered buffer, preserving device migration semantics."""

    tensor = value.detach().clone()
    module.register_buffer(name, tensor)


def _nonnegative_param_like(
    source: PositiveParam | ExactNonnegativeParam,
    values: torch.Tensor,
) -> PositiveParam | ExactNonnegativeParam:
    if isinstance(source, PositiveParam):
        return PositiveParam(
            shape=tuple(values.shape),
            init=values,
            eps=source.eps,
            dtype=values.dtype,
            device="cpu",
        )
    if isinstance(source, ExactNonnegativeParam):
        return ExactNonnegativeParam(
            shape=tuple(values.shape),
            init=values,
            dtype=values.dtype,
            device="cpu",
        )
    raise TypeError("Unsupported nonnegative parameter type.")


def _basis_dependencies(
    feature_specs: tuple[BasisFeatureSpec, ...],
    active_basis: tuple[int, ...],
) -> tuple[int, ...]:
    deps: set[int] = set()
    for basis_index in active_basis:
        if 0 <= basis_index < len(feature_specs):
            for term in feature_specs[basis_index].production_terms:
                for factor in term.factors:
                    if factor.role == "input":
                        deps.add(int(factor.index))
    return tuple(sorted(deps))


def _remap_feature_spec(
    feature: BasisFeatureSpec,
    *,
    old_input_to_new: dict[int, int],
    new_index: int,
) -> BasisFeatureSpec:
    new_terms: list[ProductionTerm] = []
    for term in feature.production_terms:
        new_factors: list[SpeciesFactor] = []
        for factor in term.factors:
            if factor.role == "input":
                old_index = int(factor.index)
                if old_index not in old_input_to_new:
                    raise ValueError(
                        "Cannot materialize compact basis because a surviving "
                        f"feature depends on pruned input index {old_index}."
                    )
                new_factors.append(replace(factor, index=int(old_input_to_new[old_index])))
            else:
                new_factors.append(factor)
        new_terms.append(replace(term, factors=tuple(new_factors)))

    metadata = getattr(feature, "metadata", None)
    if isinstance(metadata, dict):
        new_metadata = dict(metadata)
    elif metadata is not None:
        new_metadata = dict(metadata)
    else:
        new_metadata = None

    if new_metadata is not None and "input_index" in new_metadata:
        old_index = int(new_metadata["input_index"])
        if old_index not in old_input_to_new:
            raise ValueError(
                "Cannot materialize compact basis because a surviving feature "
                f"metadata depends on pruned input index {old_index}."
            )
        new_input_index = int(old_input_to_new[old_index])
        new_metadata["input_index"] = new_input_index
        if new_metadata.get("basis_family") == "simplex":
            new_metadata["input_species"] = f"X_{new_input_index}"
            if new_metadata.get("simplex_kind") == "complement":
                new_metadata["reservoir_species"] = f"R_simplex_{new_input_index}"

    return replace(
        feature,
        output_species=f"Phi_{new_index}",
        production_terms=tuple(new_terms),
        metadata=new_metadata,
    )
