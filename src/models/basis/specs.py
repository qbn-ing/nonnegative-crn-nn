from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Literal, Sequence
from utils.validation import (
    check_nonnegative_int,
    check_positive_int,
    is_int_not_bool
)

SpeciesRole = Literal["input", "basis", "buffer"] # 输入、基函数、缓冲

_VALID_SPECIES_ROLES = {"input", "basis", "buffer"}

@dataclass(frozen=True)
class SpeciesFactor:

    role: SpeciesRole
    index: int
    stoich: int = 1 # 化学计量系数

    def __post_init__(self) -> None:
        if self.role not in _VALID_SPECIES_ROLES:
            raise ValueError(
                f"Unsupported species role: {self.role!r}. "
                f"Expected one of {sorted(_VALID_SPECIES_ROLES)}."
            )

        check_nonnegative_int("index", self.index)
        check_positive_int("stoich", self.stoich)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "index": self.index,
            "stoich": self.stoich,
        }


@dataclass(frozen=True)
class RateRef: # 速率常数
    key: str # 导出名称
    value: float | None = None
    trainable: bool = False
    shared: bool = False
    param_ref: str | None = None # pytorch参数引用

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("RateRef.key must be a non-empty string.")

        if self.value is not None:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise TypeError(
                    "RateRef.value must be int, float, or None. "
                    f"Got {type(self.value).__name__}."
                )

            if self.value < 0:
                raise ValueError(
                    f"RateRef.value must be non-negative. Got {self.value}."
                )

        if not isinstance(self.trainable, bool):
            raise TypeError("RateRef.trainable must be bool.")

        if not isinstance(self.shared, bool):
            raise TypeError("RateRef.shared must be bool.")

        if self.param_ref is not None and not isinstance(self.param_ref, str):
            raise TypeError("RateRef.param_ref must be str or None.")
        
    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "trainable": self.trainable,
            "shared": self.shared,
            "param_ref": self.param_ref,
        }
    

@dataclass(frozen=True)
class ProductionTerm: # 质量作用定理生成项
    factors: tuple[SpeciesFactor, ...]
    rate: RateRef
    def __post_init__(self) -> None:
        if not isinstance(self.factors, tuple):
            raise TypeError("ProductionTerm.factors must be a tuple.")

        for factor in self.factors:
            if not isinstance(factor, SpeciesFactor):
                raise TypeError(
                    "ProductionTerm.factors must contain SpeciesFactor objects. "
                    f"Got {type(factor).__name__}."
                )

        if not isinstance(self.rate, RateRef):
            raise TypeError(
                "ProductionTerm.rate must be a RateRef. "
                f"Got {type(self.rate).__name__}."
            )
        
    @property    
    def order(self) -> int:
        return sum(factor.stoich for factor in self.factors)
    
    @property
    def dependencies(self) -> tuple[int, ...]:
        deps = {
            factor.index
            for factor in self.factors
            if factor.role == "input"
        }
        return tuple(sorted(deps))

    def to_dict(self) -> dict[str, Any]:
        return {
            "factors": [factor.to_dict() for factor in self.factors],
            "rate": self.rate.to_dict(),
            "order": self.order,
            "dependencies": self.dependencies,
        }
    
@dataclass(frozen=True)
class BasisFeatureSpec: # 基函数元数据
    name: str
    output_species: str # 输出物质名
    production_terms: tuple[ProductionTerm, ...]
    decay: RateRef # 讲解速率
    group: str | None = None
    metadata: Mapping[str, Any] | None = None
    
    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("BasisFeatureSpec.name must be a non-empty string.")

        if not isinstance(self.output_species, str) or not self.output_species:
            raise ValueError(
                "BasisFeatureSpec.output_species must be a non-empty string."
            )
        
        if not isinstance(self.production_terms, tuple):
            raise TypeError("BasisFeatureSpec.production_terms must be a tuple.")

        if len(self.production_terms) == 0:
            raise ValueError(
                "BasisFeatureSpec requires at least one production term."
            )

        for prod in self.production_terms:
            if not isinstance(prod, ProductionTerm):
                raise TypeError(
                    "BasisFeatureSpec.production_terms must contain "
                    f"ProductionTerm objects. Got {type(prod).__name__}."
                )

        if not isinstance(self.decay, RateRef):
            raise TypeError(
                "BasisFeatureSpec.decay must be a RateRef. "
                f"Got {type(self.decay).__name__}."
            )

        if self.decay.value is not None and self.decay.value <= 0:
            raise ValueError(
                "BasisFeatureSpec.decay.value must be positive when provided. "
                f"Got {self.decay.value}."
            )

        if self.group is not None and not isinstance(self.group, str):
            raise TypeError("BasisFeatureSpec.group must be str or None.")

        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            raise TypeError("BasisFeatureSpec.metadata must be a mapping or None.")


    @property
    def dependencies(self) -> tuple[int, ...]:
        deps: set[int] = set()

        for prod in self.production_terms:
            deps.update(prod.dependencies)

        return tuple(sorted(deps))
    
    @property
    def max_order(self) -> int:
        return max(prod.order for prod in self.production_terms)

    @property
    def n_production_terms(self) -> int:
        return len(self.production_terms)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "output_species": self.output_species,
            "dependencies": self.dependencies,
            "max_order": self.max_order,
            "n_production_terms": self.n_production_terms,
            "production_terms": [
                prod.to_dict() for prod in self.production_terms
            ],
            "decay": self.decay.to_dict(),
            "group": self.group,
            "metadata": None if self.metadata is None else dict(self.metadata),
        }
    
def x(index: int) -> SpeciesFactor: # 声明输入物质x_i
    return SpeciesFactor(role="input", index=index, stoich=1)


def x_pow(index: int, power: int) -> SpeciesFactor: # 声明输入物质x_i^p
    return SpeciesFactor(role="input", index=index, stoich=power)


def basis_species(index: int, *, stoich: int = 1) -> SpeciesFactor: # 声明基函数物质
    return SpeciesFactor(role="basis", index=index, stoich=stoich)


def buffer_species(index: int, *, stoich: int = 1) -> SpeciesFactor: # 声明缓冲物质
    return SpeciesFactor(role="buffer", index=index, stoich=stoich)

# 声明速率
def rate( 
    key: str,
    *,
    value: float | None = None,
    trainable: bool = False,
    shared: bool = False,
    param_ref: str | None = None,
) -> RateRef:
    return RateRef(
        key=key,
        value=value,
        trainable=trainable,
        shared=shared,
        param_ref=param_ref,
    )


# 声明质量作用定理项
def ma_term(
    *factors: SpeciesFactor, # 物质，可以没有
    rate_ref: RateRef,
) -> ProductionTerm:
    return ProductionTerm(factors=tuple(factors), rate=rate_ref)

# 声明基函数特征
def feature(
    *,
    name: str,
    output_species: str,
    production_terms: Sequence[ProductionTerm],
    decay: RateRef,
    group: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> BasisFeatureSpec:
    return BasisFeatureSpec(
        name=name,
        output_species=output_species,
        production_terms=tuple(production_terms),
        decay=decay,
        group=group,
        metadata=None if metadata is None else dict(metadata),
    )
