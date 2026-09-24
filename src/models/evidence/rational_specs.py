from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from models.basis.specs import RateRef
from utils.validation import (
    check_non_empty_str,
    check_positive_float,
    check_positive_int,
)

from .specs import EvidenceLayerSpec


@dataclass(frozen=True)
class RationalResponseSpec:
    """CRN specification of an E/I-to-r response."""

    width: int
    eta: tuple[RateRef, ...]
    gamma: tuple[RateRef, ...]
    chi: tuple[RateRef, ...]
    response_prefix: str = "R"
    positive_evidence_prefix: str = "E"
    negative_evidence_prefix: str = "I"

    def __post_init__(self) -> None:
        check_positive_int("width", self.width)
        check_non_empty_str("response_prefix", self.response_prefix)
        check_non_empty_str(
            "positive_evidence_prefix",
            self.positive_evidence_prefix,
        )
        check_non_empty_str(
            "negative_evidence_prefix",
            self.negative_evidence_prefix,
        )

        for name, values in (
            ("eta", self.eta),
            ("gamma", self.gamma),
            ("chi", self.chi),
        ):
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be a tuple.")
            if len(values) != self.width:
                raise ValueError(
                    f"{name} length must equal width. "
                    f"Got {len(values)} and {self.width}."
                )
            if not all(isinstance(value, RateRef) for value in values):
                raise TypeError(f"{name} must contain RateRef objects.")

    @property
    def n_species(self) -> int:
        return self.width

    @property
    def n_reactions(self) -> int:
        return 3 * self.width

    def species_name(self, index: int) -> str:
        return f"{self.response_prefix}_{index}"

    def to_dict(self) -> dict[str, Any]:
        reactions: list[dict[str, Any]] = []

        for index in range(self.width):
            e_name = f"{self.positive_evidence_prefix}_{index}"
            i_name = f"{self.negative_evidence_prefix}_{index}"
            r_name = self.species_name(index)
            reactions.extend(
                [
                    {
                        "kind": "activation",
                        "output_index": index,
                        "reactants": [e_name],
                        "products": [e_name, r_name],
                        "catalysts": [e_name],
                        "rate": self.eta[index].to_dict(),
                    },
                    {
                        "kind": "decay",
                        "output_index": index,
                        "reactants": [r_name],
                        "products": [],
                        "catalysts": [],
                        "rate": self.gamma[index].to_dict(),
                    },
                    {
                        "kind": "inhibition",
                        "output_index": index,
                        "reactants": [i_name, r_name],
                        "products": [i_name],
                        "catalysts": [i_name],
                        "rate": self.chi[index].to_dict(),
                    },
                ]
            )

        return {
            "width": self.width,
            "n_species": self.n_species,
            "n_reactions": self.n_reactions,
            "response_prefix": self.response_prefix,
            "positive_evidence_prefix": self.positive_evidence_prefix,
            "negative_evidence_prefix": self.negative_evidence_prefix,
            "species": [
                {
                    "name": self.species_name(index),
                    "output_index": index,
                }
                for index in range(self.width)
            ],
            "reactions": reactions,
        }


@dataclass(frozen=True, init=False)
class RationalLayerSpec:
    """Specification of one affine E/I-to-r layer.

    Bias parameters live in ``evidence.biases``.  A conserved unit catalyst is
    introduced only by the CRN lowering step and therefore does not contribute
    to ``input_dim``.
    """

    name: str
    input_dim: int
    width: int
    evidence: EvidenceLayerSpec
    response: RationalResponseSpec
    include_bias: bool = False
    bias_catalyst_initial: float = 1.0

    def __init__(
        self,
        *,
        name: str,
        input_dim: int,
        width: int,
        evidence: EvidenceLayerSpec,
        response: RationalResponseSpec,
        include_bias: bool | None = None,
        bias_catalyst_initial: float = 1.0,
        include_constant_channel: bool | None = None,
        constant_value: float | None = None,
    ) -> None:
        resolved_bias = _resolve_bias_flag(
            include_bias,
            include_constant_channel,
        )
        if constant_value is not None:
            if (
                bias_catalyst_initial != 1.0
                and bias_catalyst_initial != constant_value
            ):
                raise ValueError(
                    "bias_catalyst_initial and constant_value must agree."
                )
            bias_catalyst_initial = constant_value
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "input_dim", input_dim)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "include_bias", resolved_bias)
        object.__setattr__(
            self,
            "bias_catalyst_initial",
            bias_catalyst_initial,
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        check_non_empty_str("name", self.name)
        check_positive_int("input_dim", self.input_dim)
        check_positive_int("width", self.width)

        if not isinstance(self.evidence, EvidenceLayerSpec):
            raise TypeError("evidence must be an EvidenceLayerSpec.")
        if not isinstance(self.response, RationalResponseSpec):
            raise TypeError("response must be a RationalResponseSpec.")
        if not isinstance(self.include_bias, bool):
            raise TypeError("include_bias must be a bool.")
        check_positive_float(
            "bias_catalyst_initial",
            self.bias_catalyst_initial,
        )
        if self.bias_catalyst_initial != 1.0:
            raise ValueError(
                "bias_catalyst_initial must equal 1.0. "
                "Affine bias values are encoded by reaction-rate constants, "
                "not by rescaling the conserved catalyst."
            )
        if self.evidence.input_dim != self.input_dim:
            raise ValueError("evidence.input_dim must equal input_dim.")
        if self.evidence.output_dim != self.width:
            raise ValueError("evidence.output_dim must equal width.")
        if self.response.width != self.width:
            raise ValueError("response.width must equal width.")
        if self.evidence.biases and not self.include_bias:
            raise ValueError(
                "evidence.biases requires include_bias=True."
            )

    @property
    def include_constant_channel(self) -> bool:
        return self.include_bias

    @property
    def constant_value(self) -> float:
        return self.bias_catalyst_initial

    @property
    def n_species(self) -> int:
        return (
            self.n_constant_species
            + self.evidence.n_species
            + self.response.n_species
        )

    @property
    def n_constant_species(self) -> int:
        return int(bool(self.evidence.biases))

    @property
    def n_reactions(self) -> int:
        return (
            self.evidence.n_edges
            + self.evidence.n_biases
            + self.evidence.n_species
            + self.response.n_reactions
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "input_dim": self.input_dim,
            "evidence_input_dim": self.evidence.input_dim,
            "width": self.width,
            "include_bias": self.include_bias,
            "bias_catalyst_initial": self.bias_catalyst_initial,
            "include_constant_channel": self.include_bias,
            "constant_value": self.bias_catalyst_initial,
            "n_constant_species": self.n_constant_species,
            "n_species": self.n_species,
            "n_reactions": self.n_reactions,
            "evidence": self.evidence.to_dict(),
            "response": self.response.to_dict(),
        }


def _resolve_bias_flag(
    include_bias: bool | None,
    include_constant_channel: bool | None,
) -> bool:
    if include_bias is not None and not isinstance(include_bias, bool):
        raise TypeError("include_bias must be a bool or None.")
    if (
        include_constant_channel is not None
        and not isinstance(include_constant_channel, bool)
    ):
        raise TypeError("include_constant_channel must be a bool or None.")
    if (
        include_bias is not None
        and include_constant_channel is not None
        and include_bias != include_constant_channel
    ):
        raise ValueError(
            "include_bias and include_constant_channel must agree."
        )
    if include_bias is not None:
        return include_bias
    if include_constant_channel is not None:
        return include_constant_channel
    return False


def rational_response_spec(
    *,
    eta: Sequence[RateRef],
    gamma: Sequence[RateRef],
    chi: Sequence[RateRef],
    response_prefix: str = "R",
    positive_evidence_prefix: str = "E",
    negative_evidence_prefix: str = "I",
) -> RationalResponseSpec:
    values = tuple(eta)
    return RationalResponseSpec(
        width=len(values),
        eta=values,
        gamma=tuple(gamma),
        chi=tuple(chi),
        response_prefix=response_prefix,
        positive_evidence_prefix=positive_evidence_prefix,
        negative_evidence_prefix=negative_evidence_prefix,
    )


__all__ = [
    "RationalLayerSpec",
    "RationalResponseSpec",
    "rational_response_spec",
]
