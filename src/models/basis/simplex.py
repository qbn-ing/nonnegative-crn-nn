from __future__ import annotations

from typing import Any, Literal, Sequence

import torch

from .base import BaseBasisLayer
from .specs import (
    BasisFeatureSpec,
    feature,
    ma_term,
    rate,
    x,
)
from utils.validation import (
    check_nonnegative_float,
    check_positive_float,
    check_positive_int,
)

SimplexFeatureKind = Literal["complement", "identity"]


class SimplexBasisLayer(BaseBasisLayer):
    """CRN-compatible simplex/complement input basis.

    For every selected input dimension ``i`` this layer returns the pair

        [total - x_i, x_i]

    by default.  This matches the input encoding used in the v7.5 synthetic
    experiments while exposing explicit compile-time metadata for later CRN
    lowering.

    The complement feature is not represented as an ordinary polynomial
    production/decay feature.  Its ``BasisFeatureSpec.metadata`` marks it as
    a simplex complement motif that can be lowered to the autocatalytic CRN

        R_i + Xbar_i -> R_i + 2 Xbar_i
        X_i + Xbar_i -> X_i
        2 Xbar_i -> Xbar_i

    whose positive steady state is ``Xbar_i = total - X_i`` when the input
    concentration is externally fixed and ``0 <= X_i <= total``.

    The identity feature is marked as an alias to the corresponding input
    species.  Exporters that understand the metadata can feed ``X_i`` directly
    into the following rational network without introducing a copy species.

    Parameters
    ----------
    n_inputs:
        Number of raw input concentrations.

    total:
        Simplex total/reservoir concentration.  Defaults to ``1.0``.

    input_indices:
        Optional subset/order of input dimensions.  ``None`` means all inputs
        in natural order.

    include_complement / include_identity:
        Which features to emit for every selected input.  At least one must be
        true.  The default order is ``complement_first``: ``[total-x_i, x_i]``.

    complement_rho:
        Fixed rate parameter for the complement CRN motif metadata.

    complement_initial:
        Positive initial value used by CRN lowering for complement species.
        The autocatalytic complement motif needs a positive seed when the
        target complement steady state is positive.
    """

    def __init__(
        self,
        n_inputs: int,
        *,
        total: float = 1.0,
        input_indices: Sequence[int] | None = None,
        include_complement: bool = True,
        include_identity: bool = True,
        complement_first: bool = True,
        check_bounds: bool = True,
        complement_rho: float = 1.0,
        complement_initial: float = 1e-6,
        species_prefix: str = "Phi",
    ) -> None:
        super().__init__()

        check_positive_int("n_inputs", n_inputs)
        check_positive_float("total", total)
        check_positive_float("complement_rho", complement_rho)
        check_nonnegative_float("complement_initial", complement_initial)

        if not isinstance(include_complement, bool):
            raise TypeError(
                "include_complement must be a bool. "
                f"Got {type(include_complement).__name__}."
            )
        if not isinstance(include_identity, bool):
            raise TypeError(
                "include_identity must be a bool. "
                f"Got {type(include_identity).__name__}."
            )
        if not include_complement and not include_identity:
            raise ValueError(
                "SimplexBasisLayer must include at least one of complement or identity."
            )
        if not isinstance(complement_first, bool):
            raise TypeError(
                "complement_first must be a bool. "
                f"Got {type(complement_first).__name__}."
            )
        if not isinstance(check_bounds, bool):
            raise TypeError(
                "check_bounds must be a bool. "
                f"Got {type(check_bounds).__name__}."
            )
        if not isinstance(species_prefix, str) or not species_prefix:
            raise ValueError("species_prefix must be a non-empty string.")

        clean_indices = _normalize_input_indices(input_indices, n_inputs=n_inputs)

        self.n_inputs = int(n_inputs)
        self.total = float(total)
        self.input_indices = clean_indices
        self.include_complement = include_complement
        self.include_identity = include_identity
        self.complement_first = complement_first
        self.check_bounds = check_bounds
        self.complement_rho = float(complement_rho)
        self.complement_initial = float(complement_initial)
        self.species_prefix = species_prefix

        self._features = _build_simplex_feature_specs(
            n_inputs=n_inputs,
            input_indices=clean_indices,
            total=self.total,
            include_complement=include_complement,
            include_identity=include_identity,
            complement_first=complement_first,
            complement_rho=self.complement_rho,
            complement_initial=self.complement_initial,
            species_prefix=species_prefix,
        )
        self.validate_metadata()

    def feature_specs(self) -> list[BasisFeatureSpec]:
        return list(self._features)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        if x_in.ndim != 2:
            raise ValueError(
                "SimplexBasisLayer expects x_in with shape "
                f"(batch, n_inputs). Got shape {tuple(x_in.shape)}."
            )

        if x_in.shape[1] != self.n_inputs:
            raise ValueError(
                f"Input dimension mismatch: expected {self.n_inputs}, "
                f"got {x_in.shape[1]}."
            )

        if self.check_bounds:
            if torch.any(x_in < 0).item() or torch.any(x_in > self.total).item():
                raise ValueError(
                    "SimplexBasisLayer expected inputs in [0, total]. "
                    f"Got total={self.total}. Set check_bounds=False if this "
                    "is intentional."
                )

        cols: list[torch.Tensor] = []
        for input_index in self.input_indices:
            complement = torch.as_tensor(
                self.total,
                dtype=x_in.dtype,
                device=x_in.device,
            ) - x_in[:, input_index]
            identity = x_in[:, input_index]

            if self.include_complement and self.include_identity:
                if self.complement_first:
                    cols.extend([complement, identity])
                else:
                    cols.extend([identity, complement])
            elif self.include_complement:
                cols.append(complement)
            elif self.include_identity:
                cols.append(identity)

        return torch.stack(cols, dim=1)

    def structure_info(self) -> dict[str, Any]:
        info = super().structure_info()
        info.update(
            {
                "n_inputs": self.n_inputs,
                "input_indices": self.input_indices,
                "total": self.total,
                "include_complement": self.include_complement,
                "include_identity": self.include_identity,
                "complement_first": self.complement_first,
                "check_bounds": self.check_bounds,
                "complement_rho": self.complement_rho,
                "complement_initial": self.complement_initial,
                "basis_family": "simplex",
            }
        )
        return info

    def extra_repr(self) -> str:
        return (
            f"n_inputs={self.n_inputs}, n_basis={self.n_basis}, "
            f"total={self.total}, input_indices={self.input_indices}, "
            f"complement_first={self.complement_first}"
        )


def _normalize_input_indices(
    input_indices: Sequence[int] | None,
    *,
    n_inputs: int,
) -> tuple[int, ...]:
    if input_indices is None:
        return tuple(range(n_inputs))

    if isinstance(input_indices, (str, bytes)):
        raise TypeError("input_indices must be a sequence of integer indices.")

    try:
        items = tuple(input_indices)
    except TypeError as exc:
        raise TypeError("input_indices must be a sequence of integer indices.") from exc

    if len(items) == 0:
        raise ValueError("input_indices must be non-empty when provided.")

    out: list[int] = []
    for position, item in enumerate(items):
        if not isinstance(item, int) or isinstance(item, bool):
            raise TypeError(
                f"input_indices[{position}] must be an int. "
                f"Got {type(item).__name__}."
            )
        if item < 0 or item >= n_inputs:
            raise ValueError(
                f"input_indices[{position}]={item} is outside valid range "
                f"[0, {n_inputs})."
            )
        if item in out:
            raise ValueError(f"Duplicate input index {item} in input_indices.")
        out.append(int(item))

    return tuple(out)


def _build_simplex_feature_specs(
    *,
    n_inputs: int,
    input_indices: tuple[int, ...],
    total: float,
    include_complement: bool,
    include_identity: bool,
    complement_first: bool,
    complement_rho: float,
    complement_initial: float,
    species_prefix: str,
) -> list[BasisFeatureSpec]:
    specs: list[BasisFeatureSpec] = []

    def add(kind: SimplexFeatureKind, input_index: int) -> None:
        idx = len(specs)
        if kind == "complement":
            name = f"xbar{input_index}"
            output_species = f"Xbar_{input_index}"
            group = "simplex_complement"
            metadata = {
                "basis_family": "simplex",
                "simplex_kind": "complement",
                "compile_mode": "simplex_complement_motif",
                "input_index": input_index,
                "input_species": f"X_{input_index}",
                "reservoir_species": f"R_simplex_{input_index}",
                "output_alias": False,
                "total": total,
                "rho": complement_rho,
                "initial": complement_initial,
                "steady_state": "total_minus_input",
                "crn_reactions": [
                    "R_i + Xbar_i -> R_i + 2 Xbar_i",
                    "X_i + Xbar_i -> X_i",
                    "2 Xbar_i -> Xbar_i",
                ],
            }
            # Placeholder term only keeps BasisFeatureSpec compatible with the
            # existing metadata contract. Exporters should use compile_mode and
            # skip ordinary production/decay lowering for this feature.
            production_terms = (
                ma_term(
                    rate_ref=rate(
                        f"simplex_xbar{input_index}_placeholder_prod",
                        value=0.0,
                        trainable=False,
                        shared=False,
                    )
                ),
            )
        elif kind == "identity":
            name = f"x{input_index}"
            output_species = f"{species_prefix}_{idx}"
            group = "simplex_identity"
            metadata = {
                "basis_family": "simplex",
                "simplex_kind": "identity",
                "compile_mode": "input_alias",
                "input_index": input_index,
                "input_species": f"X_{input_index}",
                "output_alias": True,
                "total": total,
            }
            production_terms = (
                ma_term(
                    x(input_index),
                    rate_ref=rate(
                        f"simplex_x{input_index}_identity_prod",
                        value=1.0,
                        trainable=False,
                        shared=False,
                    ),
                ),
            )
        else:
            raise AssertionError(f"Unhandled simplex feature kind {kind!r}.")

        specs.append(
            feature(
                name=name,
                output_species=output_species,
                production_terms=production_terms,
                decay=rate(
                    f"simplex_{name}_decay",
                    value=1.0,
                    trainable=False,
                    shared=False,
                ),
                group=group,
                metadata=metadata,
            )
        )

    for input_index in input_indices:
        if include_complement and include_identity:
            if complement_first:
                add("complement", input_index)
                add("identity", input_index)
            else:
                add("identity", input_index)
                add("complement", input_index)
        elif include_complement:
            add("complement", input_index)
        elif include_identity:
            add("identity", input_index)

    return specs
