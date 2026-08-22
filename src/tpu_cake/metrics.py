from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from enum import StrEnum
from functools import partial

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Unit(StrEnum):
    COUNT = "count"
    NANOSECOND = "ns"
    MICROSECOND = "us"
    MILLISECOND = "ms"
    SECOND = "s"
    BYTE = "byte"
    BYTE_PER_SECOND = "byte/s"
    FLOP = "flop"
    FLOP_PER_SECOND = "flop/s"
    FLOP_PER_BYTE = "flop/byte"
    TOKEN_PER_SECOND = "token/s"
    RATIO = "ratio"
    PERCENT = "percent"


class MeasurementKind(StrEnum):
    MEASURED = "measured"
    DERIVED = "derived"
    ESTIMATED = "estimated"


class Quantity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    value: Decimal
    unit: Unit


class MeasurementInterval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scope: str = Field(min_length=1)
    start_ns: int | None = Field(default=None, ge=0)
    end_ns: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def interval_is_ordered(self) -> MeasurementInterval:
        if (self.start_ns is None) != (self.end_ns is None):
            raise ValueError("measurement intervals need both start_ns and end_ns")
        if self.start_ns is not None and self.end_ns is not None and self.end_ns < self.start_ns:
            raise ValueError("measurement interval ends before it starts")
        return self


class MetricSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_path: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    field: str = Field(min_length=1)


class FormulaIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    expression: str = Field(min_length=1)


class Metric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1)
    quantity: Quantity
    kind: MeasurementKind
    interval: MeasurementInterval
    sources: tuple[MetricSource, ...]
    formula: FormulaIdentity | None = None
    numerator: Quantity | None = None
    denominator: Quantity | None = None

    @model_validator(mode="after")
    def evidence_is_complete(self) -> Metric:
        if not self.sources:
            raise ValueError("every metric needs at least one evidence source")
        if (
            self.kind in (MeasurementKind.DERIVED, MeasurementKind.ESTIMATED)
            and self.formula is None
        ):
            raise ValueError("derived and estimated metrics need a formula identity")
        if self.quantity.unit in (Unit.RATIO, Unit.PERCENT):
            if self.numerator is None or self.denominator is None:
                raise ValueError("ratio and percent metrics need numerator and denominator")
            if self.denominator.value == 0:
                raise ValueError("metric denominator cannot be zero")
        elif self.numerator is not None or self.denominator is not None:
            raise ValueError(
                "numerator and denominator are only valid for ratio or percent metrics"
            )
        return self


def estimated_metric(
    name: str,
    value: Decimal | int,
    unit: Unit,
    source: MetricSource,
    formula_name: str,
    expression: str,
    *,
    scope: str,
    numerator: Quantity | None = None,
    denominator: Quantity | None = None,
) -> Metric:
    return Metric(
        name=name,
        quantity=Quantity(value=Decimal(value), unit=unit),
        kind=MeasurementKind.ESTIMATED,
        interval=MeasurementInterval(scope=scope),
        sources=(source,),
        formula=FormulaIdentity(name=formula_name, version="1", expression=expression),
        numerator=numerator,
        denominator=denominator,
    )


def estimated_metric_factory(default_scope: str) -> Callable[..., Metric]:
    return partial(estimated_metric, scope=default_scope)
