from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tpu_cake.matmul_collective_surface_correctness_evidence import (
    MatmulCollectiveSurfaceCorrectnessEvidence,
    SurfaceCompileContinuityEvidence,
    SurfaceCorrectnessCandidateExecution,
    SurfaceCorrectnessCaseEvidence,
    SurfaceCorrectnessInputCase,
    SurfaceCorrectnessSavedArray,
    SurfaceCorrectnessSentinel,
    SurfaceCorrectnessShardIdentity,
    SurfaceCorrectnessSlice,
    validate_surface_correctness_evidence,
)
from tpu_cake.matmul_collective_surface_correctness_protocol import (
    MatmulCollectiveSurfaceCorrectnessProtocol,
)
from tpu_cake.matmul_collective_surface_prediction import (
    MatmulCollectiveSurfaceDesignContract,
    MatmulCollectiveSurfaceSplit,
)

PROTOCOL = MatmulCollectiveSurfaceCorrectnessProtocol.model_validate_json(
    Path("contracts/matmul-collective-surface-correctness-v1.json").read_text()
)
DESIGN = MatmulCollectiveSurfaceDesignContract.model_validate_json(
    Path("contracts/matmul-collective-surface-design-v1.json").read_text()
)


def _saved(
    label: str,
    shape: tuple[int, int],
    *,
    array_label: str | None = None,
) -> SurfaceCorrectnessSavedArray:
    file_digit = format(sum(label.encode()) % 16, "x")
    array_digit = format(sum((array_label or label).encode()) % 16, "x")
    return SurfaceCorrectnessSavedArray(
        path=f"outputs/{label}.npy",
        file_sha256=file_digit * 64,
        array_sha256=array_digit * 64,
        shape=shape,
    )


def _shard(role: str, device: int, m: int, k: int, n: int):
    local_k = k // 8
    slices = (
        (
            SurfaceCorrectnessSlice(start=0, stop=m),
            SurfaceCorrectnessSlice(start=device * local_k, stop=(device + 1) * local_k),
        )
        if role == "lhs"
        else (
            SurfaceCorrectnessSlice(start=device * local_k, stop=(device + 1) * local_k),
            SurfaceCorrectnessSlice(start=0, stop=n),
        )
    )
    coordinates = tuple(
        sorted(
            {
                (
                    slices[0].start + index % (slices[0].stop - slices[0].start),
                    slices[1].start + (index * 17) % (slices[1].stop - slices[1].start),
                )
                for index in range(32)
            }
        )
    )
    assert len(coordinates) == 32
    return SurfaceCorrectnessShardIdentity(
        role=role,
        shard_index=device,
        device_id=device,
        global_shape=(m, k) if role == "lhs" else (k, n),
        sharding=("PartitionSpec(None, 't')" if role == "lhs" else "PartitionSpec('t', None)"),
        global_slice=slices,
        local_shape=tuple(value.stop - value.start for value in slices),
        host_callback_payload_nbytes=(slices[0].stop - slices[0].start)
        * (slices[1].stop - slices[1].start)
        * 2,
        host_callback_payload_sha256=format(device + 1, "x") * 64,
        sentinels=tuple(
            SurfaceCorrectnessSentinel(
                ordinal=ordinal,
                global_coordinate=value,
                local_coordinate=tuple(
                    coordinate - bound.start
                    for coordinate, bound in zip(value, slices, strict=True)
                ),
                expected_bfloat16_hex="803f",
                observed_bfloat16_hex="803f",
            )
            for ordinal, value in enumerate(coordinates)
        ),
    )


def _evidence() -> MatmulCollectiveSurfaceCorrectnessEvidence:
    scenarios = {value.name: value for value in DESIGN.scenarios}
    continuity = []
    cases = []
    execution_sequence = 0
    for scenario_name in PROTOCOL.calibration_scenarios:
        scenario = scenarios[scenario_name]
        for strategy in PROTOCOL.strategies:
            continuity.append(
                SurfaceCompileContinuityEvidence(
                    scenario_name=scenario_name,
                    strategy=strategy,
                    stablehlo_path=f"continuity/{scenario_name}/{strategy}/stablehlo.txt",
                    stablehlo_file_sha256="1" * 64,
                    compiler_hlo_path=f"continuity/{scenario_name}/{strategy}/compiler_hlo.txt",
                    compiler_hlo_file_sha256="2" * 64,
                    parent_distributed_schedule_sha256="3" * 64,
                    observed_distributed_schedule_sha256="3" * 64,
                    parent_physical_schedule_sha256="4" * 64,
                    observed_physical_schedule_sha256="4" * 64,
                    parent_pallas_source_sha256="5" * 64,
                    observed_pallas_source_sha256="5" * 64,
                    parent_semantic_stablehlo_sha256="6" * 64,
                    observed_semantic_stablehlo_sha256="6" * 64,
                    parent_semantic_compiler_hlo_sha256="7" * 64,
                    observed_semantic_compiler_hlo_sha256="7" * 64,
                )
            )
        for pattern_index, pattern in enumerate(PROTOCOL.patterns.ordered_patterns):
            inputs = SurfaceCorrectnessInputCase(
                scenario_name=scenario_name,
                pattern=pattern,
                protocol_id=PROTOCOL.protocol_id,
                pattern_contract_sha256=PROTOCOL.patterns.contract_sha256,
                lhs_shards=tuple(
                    _shard("lhs", device, scenario.m, scenario.k, scenario.n) for device in range(8)
                ),
                rhs_shards=tuple(
                    _shard("rhs", device, scenario.m, scenario.k, scenario.n) for device in range(8)
                ),
            )
            first, second = PROTOCOL.strategies
            order = (first, second, second, first)
            if pattern_index % 2:
                order = (second, first, first, second)
            repetitions = {first: 0, second: 0}
            executions = []
            oracle = _saved(f"{scenario_name}-{pattern}-oracle", (scenario.m, scenario.n))
            for position, strategy in enumerate(order, start=1):
                repetitions[strategy] += 1
                execution_sequence += 1
                compile_record = next(
                    value
                    for value in continuity
                    if value.scenario_name == scenario_name and value.strategy is strategy
                )
                executions.append(
                    SurfaceCorrectnessCandidateExecution(
                        sequence=execution_sequence,
                        position=position,
                        strategy=strategy,
                        strategy_repetition=repetitions[strategy],
                        invocation_nonce="a" * 64,
                        worker_pid=123,
                        fresh_compile_record_sha256=compile_record.compile_record_sha256,
                        lhs_identity_set_sha256=inputs.lhs_identity_set_sha256,
                        rhs_identity_set_sha256=inputs.rhs_identity_set_sha256,
                        oracle_array_sha256=oracle.array_sha256,
                        output=_saved(
                            f"{scenario_name}-{pattern}-{strategy}-{repetitions[strategy]}",
                            (scenario.m, scenario.n),
                            array_label=f"{scenario_name}-{pattern}-{strategy}",
                        ),
                        mismatched_element_count=0,
                        maximum_absolute_error=0.0,
                        maximum_normalized_error=0.0,
                    )
                )
            cases.append(
                SurfaceCorrectnessCaseEvidence(
                    input=inputs,
                    oracle=oracle,
                    executions=tuple(executions),
                )
            )
    return MatmulCollectiveSurfaceCorrectnessEvidence(
        protocol_id=PROTOCOL.protocol_id,
        protocol_file_sha256="8" * 64,
        split=MatmulCollectiveSurfaceSplit.CALIBRATION,
        parent_compile_manifest_file_sha256=PROTOCOL.parent_compile.manifest_file_sha256,
        correctness_execution_authority_sha256="9" * 64,
        continuity=tuple(continuity),
        cases=tuple(cases),
    )


def test_correctness_evidence_binds_inventory_shards_repeats_and_parent() -> None:
    evidence = _evidence()
    validate_surface_correctness_evidence(
        evidence,
        PROTOCOL,
        DESIGN,
        expected_protocol_file_sha256="8" * 64,
        expected_execution_authority_sha256="9" * 64,
        expected_invocation_nonce="a" * 64,
        expected_worker_pid=123,
    )

    assert len(evidence.continuity) == 32
    assert len(evidence.cases) == 80
    assert sum(len(value.executions) for value in evidence.cases) == 320


def test_correctness_evidence_rejects_wrong_parent_and_noncanonical_shard() -> None:
    evidence = _evidence()
    wrong_parent = evidence.model_copy(update={"parent_compile_manifest_file_sha256": "0" * 64})
    with pytest.raises(ValueError, match="AUTHORITY_MISMATCH"):
        validate_surface_correctness_evidence(
            wrong_parent,
            PROTOCOL,
            DESIGN,
            expected_protocol_file_sha256="8" * 64,
            expected_execution_authority_sha256="9" * 64,
            expected_invocation_nonce="a" * 64,
            expected_worker_pid=123,
        )
    first_case = evidence.cases[0]
    first_shard = first_case.input.lhs_shards[0].model_copy(
        update={
            "global_slice": (
                SurfaceCorrectnessSlice(start=0, stop=16),
                SurfaceCorrectnessSlice(start=1, stop=2049),
            )
        }
    )
    inputs = first_case.input.model_copy(
        update={"lhs_shards": (first_shard, *first_case.input.lhs_shards[1:])}
    )
    broken_case = first_case.model_copy(update={"input": inputs})
    broken = evidence.model_copy(update={"cases": (broken_case, *evidence.cases[1:])})
    with pytest.raises(ValueError, match="SHARD_IDENTITY_INVALID"):
        validate_surface_correctness_evidence(
            broken,
            PROTOCOL,
            DESIGN,
            expected_protocol_file_sha256="8" * 64,
            expected_execution_authority_sha256="9" * 64,
            expected_invocation_nonce="a" * 64,
            expected_worker_pid=123,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("protocol_file_sha256", "0" * 64),
        ("correctness_execution_authority_sha256", "0" * 64),
    ),
)
def test_correctness_evidence_rejects_rebound_authority(field: str, value: str) -> None:
    evidence = _evidence().model_copy(update={field: value})

    with pytest.raises(ValueError, match="AUTHORITY_MISMATCH"):
        validate_surface_correctness_evidence(
            evidence,
            PROTOCOL,
            DESIGN,
            expected_protocol_file_sha256="8" * 64,
            expected_execution_authority_sha256="9" * 64,
            expected_invocation_nonce="a" * 64,
            expected_worker_pid=123,
        )


@pytest.mark.parametrize(
    "execution_update",
    ({"invocation_nonce": "f" * 64}, {"worker_pid": 999999}),
)
def test_correctness_evidence_rejects_spliced_execution(execution_update) -> None:
    evidence = _evidence()
    first_case = evidence.cases[0]
    first_execution = first_case.executions[0].model_copy(update=execution_update)
    broken_case = first_case.model_copy(
        update={"executions": (first_execution, *first_case.executions[1:])}
    )
    broken = evidence.model_copy(update={"cases": (broken_case, *evidence.cases[1:])})

    with pytest.raises(ValueError, match="EXECUTION_SEQUENCE_MISMATCH"):
        validate_surface_correctness_evidence(
            broken,
            PROTOCOL,
            DESIGN,
            expected_protocol_file_sha256="8" * 64,
            expected_execution_authority_sha256="9" * 64,
            expected_invocation_nonce="a" * 64,
            expected_worker_pid=123,
        )


def test_compile_continuity_rejects_path_traversal() -> None:
    continuity = _evidence().continuity[0]

    with pytest.raises(ValueError, match="COMPILE_CONTINUITY_FAILED"):
        SurfaceCompileContinuityEvidence.model_validate(
            continuity.model_copy(update={"stablehlo_path": "../outside.txt"}).model_dump(
                mode="python", exclude_computed_fields=True
            )
        )


@pytest.mark.parametrize(
    "updates",
    (
        {"output": _saved("failed-nan", (16, 16)).model_copy(update={"nan_count": 1})},
        {"mismatched_element_count": 1},
        {"maximum_normalized_error": 1.1},
    ),
)
def test_candidate_execution_rejects_reported_failure(updates) -> None:
    payload = {
        "sequence": 1,
        "position": 1,
        "strategy": PROTOCOL.strategies[0],
        "strategy_repetition": 1,
        "invocation_nonce": "a" * 64,
        "worker_pid": 123,
        "fresh_compile_record_sha256": "b" * 64,
        "lhs_identity_set_sha256": "c" * 64,
        "rhs_identity_set_sha256": "d" * 64,
        "oracle_array_sha256": "e" * 64,
        "output": _saved("failed", (16, 16)),
        "mismatched_element_count": 0,
        "maximum_absolute_error": 0.0,
        "maximum_normalized_error": 0.0,
    }
    payload.update(updates)
    with pytest.raises((ValidationError, ValueError)):
        SurfaceCorrectnessCandidateExecution(**payload)
