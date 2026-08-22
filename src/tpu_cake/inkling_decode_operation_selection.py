from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tpu_cake.artifacts import file_sha256
from tpu_cake.evidence import CaptureAssessment, ProgramEvidence
from tpu_cake.identity import model_identity_sha256
from tpu_cake.inkling_decode_operation_selection_verifier import (
    verify_report_independently,
)
from tpu_cake.inkling_decode_profile import (
    InklingDecodeProfileContract,
    _write_json_new,
    validate_inkling_decode_profile,
)
from tpu_cake.xprof_evidence import _gviz_rows
from tpu_cake.xprof_export import export_xprof_capture

INKLING_DECODE_OPERATION_SELECTION_SCHEMA = "inkling-decode-operation-selection-v1"
_ATTRIBUTION_SCOPE = "xprof-summed-device-op-self-time"
_SELECTION_RULE = "largest-xprof-device-op-self-time-partition"
_PRODUCER_PATH = Path(__file__).resolve()
_REPOSITORY_ROOT = _PRODUCER_PATH.parents[2]
_VERIFIER_PATH = _PRODUCER_PATH.with_name("inkling_decode_operation_selection_verifier.py")


@dataclass
class _HloFamilyAccumulator:
    raw_time_ps: int = 0
    occurrences: int = 0
    hlo_rows: int = 0
    operational_intensities: list[Decimal] = field(default_factory=list)
    dma_stall_percents: list[Decimal] = field(default_factory=list)
    bound_by: set[str] = field(default_factory=set)


@dataclass
class _PartitionAccumulator:
    raw_time_ps: int = 0
    occurrences: int = 0
    member_names: list[str] = field(default_factory=list)


class InklingDecodeExpectedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    main_program_raw_time_ps: int = Field(gt=0)
    winner_raw_time_ps: int = Field(gt=0)
    candidate_raw_time_ps: int = Field(gt=0)
    candidate_occurrences: int = Field(gt=0)
    candidate_hlo_rows: int = Field(gt=0)
    candidate_operational_intensity_min: Decimal = Field(ge=0)
    candidate_operational_intensity_max: Decimal = Field(ge=0)


class InklingDecodeOperationSelectionContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["inkling-decode-operation-selection-v1"] = (
        INKLING_DECODE_OPERATION_SELECTION_SCHEMA
    )
    profile_contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    xplane_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    op_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hlo_stats_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    producer_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_rule: Literal["largest-xprof-device-op-self-time-partition"] = _SELECTION_RULE
    required_winner_partition_key: str = Field(min_length=1)
    maximum_unattributed_device_op_share_of_main_program: Decimal = Field(ge=0, lt=1)
    candidate_hlo_prefix: str = Field(min_length=1)
    expected_candidate_kernel_families: tuple[str, ...]
    minimum_candidate_device_op_share_of_main_program: Decimal = Field(gt=0, lt=1)
    expected_bound_by: tuple[str, ...]
    expected_claim: InklingDecodeExpectedClaim

    @property
    def contract_id(self) -> str:
        return model_identity_sha256(self)

    @model_validator(mode="after")
    def inventories_are_canonical(self) -> InklingDecodeOperationSelectionContract:
        families = self.expected_candidate_kernel_families
        if not families or families != tuple(sorted(set(families))):
            raise ValueError("operation-selection candidate families must be sorted and unique")
        if any(not family.startswith(self.candidate_hlo_prefix) for family in families):
            raise ValueError("operation-selection candidate family prefix mismatch")
        if not self.expected_bound_by or self.expected_bound_by != tuple(
            sorted(set(self.expected_bound_by))
        ):
            raise ValueError("operation-selection bound classes must be sorted and unique")
        claim = self.expected_claim
        if claim.candidate_operational_intensity_min > claim.candidate_operational_intensity_max:
            raise ValueError("operation-selection expected intensity range is inverted")
        return self


class InklingDecodeOperationPartitionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(gt=0)
    key: str = Field(min_length=1)
    source: Literal["xprof-category", "custom-call-family"]
    raw_time_ps: int = Field(gt=0)
    occurrences: int = Field(gt=0)
    device_op_share_of_main_program: Decimal = Field(gt=0, lt=1)
    member_names: tuple[str, ...]

    @model_validator(mode="after")
    def members_are_canonical(self) -> InklingDecodeOperationPartitionEvidence:
        if not self.member_names or self.member_names != tuple(sorted(set(self.member_names))):
            raise ValueError("operation partition members must be sorted and unique")
        return self


class InklingDecodeKernelFamilyEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    raw_time_ps: int = Field(gt=0)
    occurrences: int = Field(gt=0)
    hlo_rows: int = Field(gt=0)
    device_op_share_of_main_program: Decimal = Field(gt=0, lt=1)
    operational_intensity_min: Decimal = Field(ge=0)
    operational_intensity_max: Decimal = Field(ge=0)
    dma_stall_percent_min: Decimal = Field(ge=0)
    dma_stall_percent_max: Decimal = Field(ge=0)
    bound_by: tuple[str, ...]


class InklingDecodeOperationSelectionReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["inkling-decode-operation-selection-v1"] = (
        INKLING_DECODE_OPERATION_SELECTION_SCHEMA
    )
    report_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    xplane_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    op_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hlo_stats_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    producer_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    main_program_name: str
    main_program_id: str
    main_program_semantic_hlo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    main_program_raw_time_ps: int = Field(gt=0)
    attributed_raw_time_ps: int = Field(gt=0)
    unattributed_raw_time_ps: int = Field(ge=0)
    unattributed_device_op_share_of_main_program: Decimal = Field(ge=0, lt=1)
    selection_rule: Literal["largest-xprof-device-op-self-time-partition"] = _SELECTION_RULE
    operation_ranking: tuple[InklingDecodeOperationPartitionEvidence, ...]
    winner_partition_key: str
    winner_raw_time_ps: int = Field(gt=0)
    winner_device_op_share_of_main_program: Decimal = Field(gt=0, lt=1)
    candidate_hlo_prefix: str
    candidate_raw_time_ps: int = Field(gt=0)
    candidate_occurrences: int = Field(gt=0)
    candidate_hlo_rows: int = Field(gt=0)
    candidate_device_op_share_of_main_program: Decimal = Field(gt=0, lt=1)
    candidate_kernel_families: tuple[InklingDecodeKernelFamilyEvidence, ...]
    attribution_scope: Literal["xprof-summed-device-op-self-time"] = _ATTRIBUTION_SCOPE
    is_end_to_end_latency: Literal[False] = False
    bound_classification_source: Literal["xprof-hlo-stats"] = "xprof-hlo-stats"

    @model_validator(mode="after")
    def arithmetic_and_identity_are_consistent(self) -> InklingDecodeOperationSelectionReport:
        ranking = self.operation_ranking
        if not ranking or ranking != tuple(
            sorted(ranking, key=lambda item: (-item.raw_time_ps, item.key))
        ):
            raise ValueError("operation-selection ranking order mismatch")
        if tuple(item.rank for item in ranking) != tuple(range(1, len(ranking) + 1)):
            raise ValueError("operation-selection ranks must be contiguous")
        if len({item.key for item in ranking}) != len(ranking):
            raise ValueError("operation-selection partition keys must be unique")
        if sum(item.raw_time_ps for item in ranking) != self.attributed_raw_time_ps:
            raise ValueError("operation-selection attributed time mismatch")
        if (
            self.attributed_raw_time_ps + self.unattributed_raw_time_ps
            != self.main_program_raw_time_ps
        ):
            raise ValueError("operation-selection main time partition mismatch")
        for item in ranking:
            if item.device_op_share_of_main_program != _share(
                item.raw_time_ps, self.main_program_raw_time_ps
            ):
                raise ValueError("operation-selection partition share mismatch")
        winner = ranking[0]
        if (
            self.winner_partition_key != winner.key
            or self.winner_raw_time_ps != winner.raw_time_ps
            or self.winner_device_op_share_of_main_program != winner.device_op_share_of_main_program
        ):
            raise ValueError("operation-selection winner mismatch")
        if self.unattributed_device_op_share_of_main_program != _share(
            self.unattributed_raw_time_ps, self.main_program_raw_time_ps
        ):
            raise ValueError("operation-selection unattributed share mismatch")
        if self.unattributed_raw_time_ps >= self.winner_raw_time_ps:
            raise ValueError("operation-selection unattributed time can change winner")
        candidate_time = sum(item.raw_time_ps for item in self.candidate_kernel_families)
        candidate_occurrences = sum(item.occurrences for item in self.candidate_kernel_families)
        candidate_rows = sum(item.hlo_rows for item in self.candidate_kernel_families)
        if (
            candidate_time != self.candidate_raw_time_ps
            or candidate_occurrences != self.candidate_occurrences
            or candidate_rows != self.candidate_hlo_rows
            or self.candidate_device_op_share_of_main_program
            != _share(candidate_time, self.main_program_raw_time_ps)
        ):
            raise ValueError("operation-selection candidate aggregate mismatch")
        if model_identity_sha256(self, exclude={"report_id"}) != self.report_id:
            raise ValueError("operation-selection report identity mismatch")
        return self


def _mapping(value: object, *, error: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(error)
    return value


def _children(value: dict[str, Any], *, error: str) -> tuple[dict[str, Any], ...]:
    children = value.get("children")
    if not isinstance(children, list) or any(not isinstance(child, dict) for child in children):
        raise ValueError(error)
    return tuple(children)


def _positive_integer(value: object, *, error: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(error)
    return value


def _decimal(value: object, *, error: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exception:
        raise ValueError(error) from exception
    if not result.is_finite() or result < 0:
        raise ValueError(error)
    return result


def _integer_decimal(value: object, *, error: str) -> int:
    result = _decimal(value, error=error)
    if result != result.to_integral_value() or result <= 0:
        raise ValueError(error)
    return int(result)


def _family_name(name: str, prefix: str) -> str | None:
    match = re.fullmatch(
        rf"(?P<family>{re.escape(prefix)}[^.]+)\.\d+(?: and its duplicate\(s\))?",
        name,
    )
    return None if match is None else match.group("family")


def _stable_custom_call_family(name: str) -> str:
    match = re.fullmatch(
        r"(?P<family>[^.]+)\.\d+(?:\.[^. ]+)*(?: and its duplicate\(s\))?",
        name,
    )
    if match is None:
        raise ValueError("INKLING_OP_PROFILE_CUSTOM_CALL_FAMILY_INVALID")
    return match.group("family")


def _share(numerator: int, denominator: int) -> Decimal:
    with localcontext() as context:
        context.prec = 50
        return Decimal(numerator) / Decimal(denominator)


def _profile_main_program_contract(profile_contract: InklingDecodeProfileContract):
    matches = tuple(
        program
        for program in profile_contract.programs
        if program.name_prefix == profile_contract.main_program_prefix
    )
    if len(matches) != 1:
        raise ValueError("INKLING_OPERATION_SELECTION_PROFILE_MAIN_CONTRACT_MISMATCH")
    return matches[0]


def _main_program(
    assessment: CaptureAssessment,
    semantic_hlo_sha256_by_program: dict[str, str],
    profile_contract: InklingDecodeProfileContract,
) -> ProgramEvidence:
    expected = _profile_main_program_contract(profile_contract)
    matches = tuple(
        program
        for program in assessment.capture.programs
        if program.name.startswith(f"{profile_contract.main_program_prefix}(")
        and program.name.endswith(")")
    )
    if len(matches) != 1:
        raise ValueError("INKLING_OPERATION_SELECTION_MAIN_PROGRAM_INVENTORY_MISMATCH")
    program = matches[0]
    if semantic_hlo_sha256_by_program.get(program.name) != expected.semantic_hlo_sha256:
        raise ValueError("INKLING_OPERATION_SELECTION_MAIN_HLO_MISMATCH")
    return program


def _operation_partitions(
    path: Path,
    *,
    main_program_name: str,
) -> tuple[int, int, tuple[InklingDecodeOperationPartitionEvidence, ...]]:
    payload = _mapping(json.loads(path.read_text()), error="INKLING_OP_PROFILE_INVALID")
    if payload.get("deviceType") != "TPU":
        raise ValueError("INKLING_OP_PROFILE_DEVICE_TYPE_MISMATCH")
    root = _mapping(payload.get("byProgram"), error="INKLING_OP_PROFILE_PROGRAM_ROOT_MISSING")
    main_nodes = tuple(
        child
        for child in _children(root, error="INKLING_OP_PROFILE_PROGRAM_CHILDREN_INVALID")
        if child.get("name") == main_program_name
    )
    if len(main_nodes) != 1:
        raise ValueError("INKLING_OP_PROFILE_MAIN_PROGRAM_INVENTORY_MISMATCH")
    main = main_nodes[0]
    main_metrics = _mapping(main.get("metrics"), error="INKLING_OP_PROFILE_MAIN_METRICS_MISSING")
    main_time = _positive_integer(
        main_metrics.get("rawTime"), error="INKLING_OP_PROFILE_MAIN_TIME_INVALID"
    )
    custom_calls = 0
    accumulators: dict[tuple[str, str], _PartitionAccumulator] = {}
    for child in _children(main, error="INKLING_OP_PROFILE_MAIN_CHILDREN_INVALID"):
        name = child.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("INKLING_OP_PROFILE_CATEGORY_NAME_INVALID")
        metrics = _mapping(
            child.get("metrics"), error="INKLING_OP_PROFILE_CATEGORY_METRICS_MISSING"
        )
        raw_time = _positive_integer(
            metrics.get("rawTime"), error="INKLING_OP_PROFILE_CATEGORY_TIME_INVALID"
        )
        occurrences = _positive_integer(
            metrics.get("occurrences"), error="INKLING_OP_PROFILE_CATEGORY_OCCURRENCES_INVALID"
        )
        if name != "custom-call":
            key = ("xprof-category", name)
            if key in accumulators:
                raise ValueError("INKLING_OP_PROFILE_CATEGORY_DUPLICATE")
            accumulators[key] = _PartitionAccumulator(raw_time, occurrences, [name])
            continue
        custom_calls += 1
        child_time = 0
        child_occurrences = 0
        for operation in _children(child, error="INKLING_OP_PROFILE_CUSTOM_CALL_CHILDREN_INVALID"):
            operation_name = operation.get("name")
            if not isinstance(operation_name, str):
                raise TypeError("INKLING_OP_PROFILE_CUSTOM_CALL_NAME_INVALID")
            family = _stable_custom_call_family(operation_name)
            operation_metrics = _mapping(
                operation.get("metrics"), error="INKLING_OP_PROFILE_CUSTOM_CALL_METRICS_MISSING"
            )
            operation_time = _positive_integer(
                operation_metrics.get("rawTime"),
                error="INKLING_OP_PROFILE_CUSTOM_CALL_TIME_INVALID",
            )
            operation_occurrences = _positive_integer(
                operation_metrics.get("occurrences"),
                error="INKLING_OP_PROFILE_CUSTOM_CALL_OCCURRENCES_INVALID",
            )
            child_time += operation_time
            child_occurrences += operation_occurrences
            entry = accumulators.setdefault(("custom-call-family", family), _PartitionAccumulator())
            entry.raw_time_ps += operation_time
            entry.occurrences += operation_occurrences
            entry.member_names.append(operation_name)
        if child_time != raw_time or child_occurrences != occurrences:
            raise ValueError("INKLING_OP_PROFILE_CUSTOM_CALL_CONTAINER_MISMATCH")
    if custom_calls != 1:
        raise ValueError("INKLING_OP_PROFILE_CUSTOM_CALL_INVENTORY_MISMATCH")

    ordered = sorted(
        accumulators.items(),
        key=lambda item: (-item[1].raw_time_ps, f"{item[0][0]}/{item[0][1]}"),
    )
    partitions = tuple(
        InklingDecodeOperationPartitionEvidence(
            rank=rank,
            key=f"{source}/{name}",
            source=source,
            raw_time_ps=entry.raw_time_ps,
            occurrences=entry.occurrences,
            device_op_share_of_main_program=_share(entry.raw_time_ps, main_time),
            member_names=tuple(sorted(entry.member_names)),
        )
        for rank, ((source, name), entry) in enumerate(ordered, start=1)
    )
    attributed_time = sum(item.raw_time_ps for item in partitions)
    if attributed_time > main_time:
        raise ValueError("INKLING_OP_PROFILE_MAIN_TIME_UNDERFLOW")
    return main_time, main_time - attributed_time, partitions


def _candidate_op_profile_families(
    ranking: tuple[InklingDecodeOperationPartitionEvidence, ...],
    *,
    candidate_prefix: str,
) -> dict[str, tuple[int, int]]:
    families: dict[str, tuple[int, int]] = {}
    for partition in ranking:
        if partition.source != "custom-call-family":
            continue
        family = partition.key.removeprefix("custom-call-family/")
        if not family.startswith(candidate_prefix):
            continue
        member_families = {
            _family_name(member, candidate_prefix) for member in partition.member_names
        }
        if member_families != {family} or family in families:
            raise ValueError("INKLING_OP_PROFILE_CANDIDATE_FAMILY_INVALID")
        families[family] = (partition.raw_time_ps, partition.occurrences)
    return families


def _hlo_stats_families(
    path: Path,
    *,
    main_program_id: str,
    candidate_prefix: str,
) -> dict[str, _HloFamilyAccumulator]:
    families: dict[str, _HloFamilyAccumulator] = {}
    for row in _gviz_rows(path):
        name = str(row.get("hlo_op_name") or "")
        if str(row.get("program_id")) != main_program_id or not name.startswith(candidate_prefix):
            continue
        family = _family_name(name, candidate_prefix)
        if family is None:
            raise ValueError("INKLING_HLO_STATS_CANDIDATE_FAMILY_INVALID")
        if row.get("category") != "custom-call":
            raise ValueError("INKLING_HLO_STATS_CANDIDATE_CATEGORY_MISMATCH")
        time_us = _decimal(row.get("total_self_time"), error="INKLING_HLO_STATS_SELF_TIME_INVALID")
        time_ps = time_us * Decimal(1_000_000)
        if time_ps != time_ps.to_integral_value() or time_ps <= 0:
            raise ValueError("INKLING_HLO_STATS_SELF_TIME_NOT_EXACT_PS")
        entry = families.setdefault(family, _HloFamilyAccumulator())
        entry.raw_time_ps += int(time_ps)
        entry.occurrences += _integer_decimal(
            row.get("occurrences"), error="INKLING_HLO_STATS_OCCURRENCES_INVALID"
        )
        entry.hlo_rows += 1
        entry.operational_intensities.append(
            _decimal(
                row.get("operational_intensity"),
                error="INKLING_HLO_STATS_OPERATIONAL_INTENSITY_INVALID",
            )
        )
        entry.dma_stall_percents.append(
            _decimal(row.get("dma_stall_percent"), error="INKLING_HLO_STATS_DMA_STALL_INVALID")
        )
        bound = row.get("bound_by")
        if not isinstance(bound, str) or not bound:
            raise ValueError("INKLING_HLO_STATS_BOUND_CLASS_MISSING")
        entry.bound_by.add(bound)
    return families


def _report_with_identity(**payload: object) -> InklingDecodeOperationSelectionReport:
    provisional = InklingDecodeOperationSelectionReport.model_construct(
        report_id="0" * 64, **payload
    )
    report_id = model_identity_sha256(provisional, exclude={"report_id"})
    return InklingDecodeOperationSelectionReport.model_validate({**payload, "report_id": report_id})


def derive_inkling_decode_operation_selection(
    *,
    assessment: CaptureAssessment,
    semantic_hlo_sha256_by_program: dict[str, str],
    profile_contract: InklingDecodeProfileContract,
    profile_request_sha256: str,
    prompt_corpus_sha256: str,
    producer_source_sha256: str,
    verifier_source_sha256: str,
    uv_lock_sha256: str,
    op_profile_path: Path,
    hlo_stats_path: Path,
    contract: InklingDecodeOperationSelectionContract,
) -> InklingDecodeOperationSelectionReport:
    if not assessment.accepted:
        raise ValueError("INKLING_OPERATION_SELECTION_PROFILE_NOT_ACCEPTED")
    if profile_contract.contract_id != contract.profile_contract_id:
        raise ValueError("INKLING_OPERATION_SELECTION_PROFILE_CONTRACT_MISMATCH")
    checks = (
        (profile_request_sha256, contract.profile_request_sha256, "PROFILE_REQUEST"),
        (prompt_corpus_sha256, contract.prompt_corpus_sha256, "PROMPT_CORPUS"),
        (assessment.capture.xplane.sha256, contract.xplane_sha256, "XPLANE"),
        (file_sha256(op_profile_path), contract.op_profile_sha256, "OP_PROFILE_HASH"),
        (file_sha256(hlo_stats_path), contract.hlo_stats_sha256, "HLO_STATS_HASH"),
        (producer_source_sha256, contract.producer_source_sha256, "PRODUCER_SOURCE"),
        (verifier_source_sha256, contract.verifier_source_sha256, "VERIFIER_SOURCE"),
        (uv_lock_sha256, contract.uv_lock_sha256, "UV_LOCK"),
    )
    for observed, expected, name in checks:
        if observed != expected:
            raise ValueError(f"INKLING_OPERATION_SELECTION_{name}_MISMATCH")

    main = _main_program(assessment, semantic_hlo_sha256_by_program, profile_contract)
    main_time, unattributed_time, ranking = _operation_partitions(
        op_profile_path, main_program_name=main.name
    )
    if not ranking or ranking[0].key != contract.required_winner_partition_key:
        raise ValueError("INKLING_OPERATION_SELECTION_REQUIRED_WINNER_MISMATCH")
    unattributed_share = _share(unattributed_time, main_time)
    if unattributed_share > contract.maximum_unattributed_device_op_share_of_main_program:
        raise ValueError("INKLING_OPERATION_SELECTION_UNATTRIBUTED_SHARE_ABOVE_GATE")
    op_families = _candidate_op_profile_families(
        ranking, candidate_prefix=contract.candidate_hlo_prefix
    )
    hlo_families = _hlo_stats_families(
        hlo_stats_path,
        main_program_id=main.program_id,
        candidate_prefix=contract.candidate_hlo_prefix,
    )
    expected = contract.expected_candidate_kernel_families
    if tuple(sorted(op_families)) != expected or tuple(sorted(hlo_families)) != expected:
        raise ValueError("INKLING_OPERATION_SELECTION_CANDIDATE_INVENTORY_MISMATCH")

    family_evidence = []
    for name in expected:
        op_time, op_occurrences = op_families[name]
        hlo = hlo_families[name]
        if op_time != hlo.raw_time_ps:
            raise ValueError("INKLING_OPERATION_SELECTION_TIME_MISMATCH")
        if op_occurrences != hlo.occurrences:
            raise ValueError("INKLING_OPERATION_SELECTION_OCCURRENCES_MISMATCH")
        bounds = tuple(sorted(hlo.bound_by))
        if bounds != contract.expected_bound_by:
            raise ValueError("INKLING_OPERATION_SELECTION_BOUND_CLASS_MISMATCH")
        family_evidence.append(
            InklingDecodeKernelFamilyEvidence(
                name=name,
                raw_time_ps=op_time,
                occurrences=op_occurrences,
                hlo_rows=hlo.hlo_rows,
                device_op_share_of_main_program=_share(op_time, main_time),
                operational_intensity_min=min(hlo.operational_intensities),
                operational_intensity_max=max(hlo.operational_intensities),
                dma_stall_percent_min=min(hlo.dma_stall_percents),
                dma_stall_percent_max=max(hlo.dma_stall_percents),
                bound_by=bounds,
            )
        )

    candidate_time = sum(family.raw_time_ps for family in family_evidence)
    candidate_share = _share(candidate_time, main_time)
    if candidate_share < contract.minimum_candidate_device_op_share_of_main_program:
        raise ValueError("INKLING_OPERATION_SELECTION_CANDIDATE_SHARE_BELOW_GATE")
    intensity_min = min(family.operational_intensity_min for family in family_evidence)
    intensity_max = max(family.operational_intensity_max for family in family_evidence)
    claim = contract.expected_claim
    if (
        main_time,
        ranking[0].raw_time_ps,
        candidate_time,
        sum(family.occurrences for family in family_evidence),
        sum(family.hlo_rows for family in family_evidence),
        intensity_min,
        intensity_max,
    ) != (
        claim.main_program_raw_time_ps,
        claim.winner_raw_time_ps,
        claim.candidate_raw_time_ps,
        claim.candidate_occurrences,
        claim.candidate_hlo_rows,
        claim.candidate_operational_intensity_min,
        claim.candidate_operational_intensity_max,
    ):
        raise ValueError("INKLING_OPERATION_SELECTION_EXPECTED_CLAIM_MISMATCH")
    main_contract = _profile_main_program_contract(profile_contract)
    return _report_with_identity(
        contract_id=contract.contract_id,
        profile_contract_id=profile_contract.contract_id,
        profile_request_sha256=profile_request_sha256,
        prompt_corpus_sha256=prompt_corpus_sha256,
        xplane_sha256=assessment.capture.xplane.sha256,
        op_profile_sha256=contract.op_profile_sha256,
        hlo_stats_sha256=contract.hlo_stats_sha256,
        producer_source_sha256=producer_source_sha256,
        verifier_source_sha256=verifier_source_sha256,
        uv_lock_sha256=uv_lock_sha256,
        main_program_name=main.name,
        main_program_id=main.program_id,
        main_program_semantic_hlo_sha256=main_contract.semantic_hlo_sha256,
        main_program_raw_time_ps=main_time,
        attributed_raw_time_ps=sum(item.raw_time_ps for item in ranking),
        unattributed_raw_time_ps=unattributed_time,
        unattributed_device_op_share_of_main_program=unattributed_share,
        operation_ranking=ranking,
        winner_partition_key=ranking[0].key,
        winner_raw_time_ps=ranking[0].raw_time_ps,
        winner_device_op_share_of_main_program=ranking[0].device_op_share_of_main_program,
        candidate_hlo_prefix=contract.candidate_hlo_prefix,
        candidate_raw_time_ps=candidate_time,
        candidate_occurrences=sum(family.occurrences for family in family_evidence),
        candidate_hlo_rows=sum(family.hlo_rows for family in family_evidence),
        candidate_device_op_share_of_main_program=candidate_share,
        candidate_kernel_families=tuple(family_evidence),
    )


def _selection_provenance() -> tuple[str, str, str]:
    return (
        file_sha256(_PRODUCER_PATH),
        file_sha256(_VERIFIER_PATH),
        file_sha256(_REPOSITORY_ROOT / "uv.lock"),
    )


def select_inkling_decode_operation(
    *,
    capture_root: Path,
    request_path: Path,
    prompt_cases_path: Path,
    profile_contract: InklingDecodeProfileContract,
    selection_contract: InklingDecodeOperationSelectionContract,
) -> InklingDecodeOperationSelectionReport:
    assessment = validate_inkling_decode_profile(
        capture_root=capture_root,
        request_path=request_path,
        prompt_cases_path=prompt_cases_path,
        contract=profile_contract,
    )
    producer_sha256, verifier_sha256, uv_lock_sha256 = _selection_provenance()
    with TemporaryDirectory(prefix="tpu-cake-inkling-operation-selection-") as temporary:
        output_root = Path(temporary)
        manifest = export_xprof_capture(
            capture_root,
            output_root,
            tools=("op_profile", "hlo_stats"),
        )
        exports = {export.tool: export.output for export in manifest.exports}
        if set(exports) != {"op_profile", "hlo_stats"}:
            raise ValueError("INKLING_OPERATION_SELECTION_XPROF_EXPORT_INVENTORY_MISMATCH")
        return derive_inkling_decode_operation_selection(
            assessment=assessment.capture,
            semantic_hlo_sha256_by_program=assessment.semantic_hlo_sha256_by_program,
            profile_contract=profile_contract,
            profile_request_sha256=assessment.request_sha256,
            prompt_corpus_sha256=assessment.prompt_corpus_sha256,
            producer_source_sha256=producer_sha256,
            verifier_source_sha256=verifier_sha256,
            uv_lock_sha256=uv_lock_sha256,
            op_profile_path=exports["op_profile"],
            hlo_stats_path=exports["hlo_stats"],
            contract=selection_contract,
        )


def validate_inkling_decode_operation_selection(
    report_path: Path,
    *,
    capture_root: Path,
    request_path: Path,
    prompt_cases_path: Path,
    profile_contract: InklingDecodeProfileContract,
    selection_contract: InklingDecodeOperationSelectionContract,
) -> InklingDecodeOperationSelectionReport:
    saved = InklingDecodeOperationSelectionReport.model_validate_json(report_path.read_text())
    replayed = select_inkling_decode_operation(
        capture_root=capture_root,
        request_path=request_path,
        prompt_cases_path=prompt_cases_path,
        profile_contract=profile_contract,
        selection_contract=selection_contract,
    )
    if saved != replayed:
        raise ValueError("INKLING_OPERATION_SELECTION_REPORT_REPLAY_MISMATCH")
    verify_report_independently(
        report_path=report_path,
        capture_root=capture_root,
        profile_contract=profile_contract,
        selection_contract=selection_contract,
    )
    return saved


def write_inkling_decode_operation_selection(
    path: Path, report: InklingDecodeOperationSelectionReport
) -> None:
    _write_json_new(path, report)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tpu_cake.inkling_decode_operation_selection")
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("capture", type=Path)
    select.add_argument("--request", required=True, type=Path)
    select.add_argument("--prompt-cases", required=True, type=Path)
    select.add_argument("--profile-contract", required=True, type=Path)
    select.add_argument("--selection-contract", required=True, type=Path)
    select.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("report", type=Path)
    verify.add_argument("--capture", required=True, type=Path)
    verify.add_argument("--request", required=True, type=Path)
    verify.add_argument("--prompt-cases", required=True, type=Path)
    verify.add_argument("--profile-contract", required=True, type=Path)
    verify.add_argument("--selection-contract", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    profile_contract = InklingDecodeProfileContract.model_validate_json(
        args.profile_contract.read_text()
    )
    selection_contract = InklingDecodeOperationSelectionContract.model_validate_json(
        args.selection_contract.read_text()
    )
    if args.command == "select":
        report = select_inkling_decode_operation(
            capture_root=args.capture,
            request_path=args.request,
            prompt_cases_path=args.prompt_cases,
            profile_contract=profile_contract,
            selection_contract=selection_contract,
        )
        write_inkling_decode_operation_selection(args.output, report)
        verdict = "SELECTED"
    else:
        report = validate_inkling_decode_operation_selection(
            args.report,
            capture_root=args.capture,
            request_path=args.request,
            prompt_cases_path=args.prompt_cases,
            profile_contract=profile_contract,
            selection_contract=selection_contract,
        )
        verdict = "REPLAYED"
    print(
        f"INKLING_DECODE_OPERATION_{verdict} "
        f"report_id={report.report_id} "
        f"winner={report.winner_partition_key} "
        f"winner_share={report.winner_device_op_share_of_main_program} "
        f"candidate_prefix={report.candidate_hlo_prefix} "
        f"candidate_share={report.candidate_device_op_share_of_main_program} "
        "end_to_end_latency=false"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
