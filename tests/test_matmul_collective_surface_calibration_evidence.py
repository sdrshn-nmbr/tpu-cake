from __future__ import annotations

from collections.abc import Iterator

import pytest

from tpu_cake.matmul_collective_surface_calibration_evidence import (
    MatmulCollectiveSurfaceCalibrationEvidence,
    SurfaceCalibrationCallSample,
    SurfaceCalibrationOutputGate,
    SurfaceCalibrationResidentPair,
    SurfaceCalibrationTimingInput,
    SurfaceCalibrationWarmupExecution,
    validate_surface_calibration_evidence,
)
from tpu_cake.matmul_collective_surface_calibration_protocol import (
    default_matmul_collective_surface_calibration_protocol,
)
from tpu_cake.matmul_collective_surface_correctness_evidence import (
    SurfaceCompileContinuityEvidence,
    SurfaceCorrectnessInputCase,
    SurfaceCorrectnessSavedArray,
    SurfaceCorrectnessSentinel,
    SurfaceCorrectnessShardIdentity,
    SurfaceCorrectnessSlice,
)
from tpu_cake.matmul_collective_surface_prediction import (
    default_matmul_collective_surface_design_contract,
)

PROTOCOL_FILE_SHA256 = "1" * 64
DESIGN_FILE_SHA256 = "2" * 64
AUTHORITY_SHA256 = "3" * 64
NONCE = "4" * 64
WORKER_PID = 4242


def _saved(path: str, shape: tuple[int, int], array_sha256: str) -> SurfaceCorrectnessSavedArray:
    return SurfaceCorrectnessSavedArray(
        path=path,
        file_sha256="5" * 64,
        array_sha256=array_sha256,
        shape=shape,
        dtype="float32",
        numpy_dtype_str="<f4",
        nan_count=0,
        positive_infinity_count=0,
        negative_infinity_count=0,
    )


def _shard(role: str, shard_index: int) -> SurfaceCorrectnessShardIdentity:
    sentinels = tuple(
        SurfaceCorrectnessSentinel(
            ordinal=index,
            global_coordinate=(0, index),
            local_coordinate=(0, index),
            expected_bfloat16_hex="0000",
            observed_bfloat16_hex="0000",
        )
        for index in range(32)
    )
    return SurfaceCorrectnessShardIdentity(
        role=role,
        shard_index=shard_index,
        device_id=shard_index,
        process_index=0,
        global_shape=(1, 32),
        sharding="PartitionSpec(None, 't')" if role == "lhs" else "PartitionSpec('t', None)",
        global_slice=(
            SurfaceCorrectnessSlice(start=0, stop=1),
            SurfaceCorrectnessSlice(start=0, stop=32),
        ),
        local_shape=(1, 32),
        logical_dtype="bfloat16",
        numpy_dtype_str="<V2",
        payload_byte_order="little",
        host_callback_payload_nbytes=64,
        host_callback_payload_sha256=f"{shard_index + 1:064x}",
        sentinels=sentinels,
    )


def _clock() -> Iterator[tuple[int, int]]:
    current = 100
    while True:
        yield current, current + 1
        current += 2


def _evidence() -> MatmulCollectiveSurfaceCalibrationEvidence:
    protocol = default_matmul_collective_surface_calibration_protocol()
    design = default_matmul_collective_surface_design_contract()
    shapes = {value.name: (value.m, value.n) for value in design.calibration_scenarios}
    lhs = tuple(_shard("lhs", index) for index in range(8))
    rhs = tuple(_shard("rhs", index) for index in range(8))
    continuity = []
    for scenario in protocol.scenarios:
        for strategy in protocol.strategies:
            continuity.append(
                SurfaceCompileContinuityEvidence(
                    scenario_name=scenario,
                    strategy=strategy,
                    stablehlo_path=f"continuity/{scenario}/{strategy.value}/stablehlo.txt",
                    stablehlo_file_sha256="6" * 64,
                    compiler_hlo_path=f"continuity/{scenario}/{strategy.value}/compiler_hlo.txt",
                    compiler_hlo_file_sha256="7" * 64,
                    parent_distributed_schedule_sha256="8" * 64,
                    observed_distributed_schedule_sha256="8" * 64,
                    parent_physical_schedule_sha256="9" * 64,
                    observed_physical_schedule_sha256="9" * 64,
                    parent_pallas_source_sha256="a" * 64,
                    observed_pallas_source_sha256="a" * 64,
                    parent_semantic_stablehlo_sha256="b" * 64,
                    observed_semantic_stablehlo_sha256="b" * 64,
                    parent_semantic_compiler_hlo_sha256="c" * 64,
                    observed_semantic_compiler_hlo_sha256="c" * 64,
                )
            )
    inputs = []
    for scenario in protocol.scenarios:
        array_sha256 = f"{protocol.scenarios.index(scenario) + 100:064x}"
        inputs.append(
            SurfaceCalibrationTimingInput(
                scenario_name=scenario,
                parent_case_sha256="d" * 64,
                parent_xla_array_sha256=array_sha256,
                parent_pallas_array_sha256=array_sha256,
                input=SurfaceCorrectnessInputCase(
                    scenario_name=scenario,
                    pattern="signed-periodic",
                    protocol_id=protocol.correctness_parent.protocol_id,
                    pattern_contract_sha256="e" * 64,
                    lhs_shards=lhs,
                    rhs_shards=rhs,
                ),
                oracle=_saved(f"oracles/{scenario}.npy", shapes[scenario], array_sha256),
            )
        )
    continuity_by_arm = {
        (value.scenario_name, value.strategy): value.compile_record_sha256 for value in continuity
    }
    pairs = tuple(
        SurfaceCalibrationResidentPair(
            scenario_name=scenario,
            xla_compile_record_sha256=continuity_by_arm[(scenario, protocol.strategies[0])],
            pallas_compile_record_sha256=continuity_by_arm[(scenario, protocol.strategies[1])],
            invocation_nonce=NONCE,
            worker_pid=WORKER_PID,
        )
        for scenario in protocol.scenarios
    )
    pair_by_scenario = {value.scenario_name: value for value in pairs}
    input_by_scenario = {value.scenario_name: value for value in inputs}
    clock = _clock()
    gates = []
    for phase in ("before_timing",):
        for scenario in protocol.scenarios:
            for strategy in protocol.strategies:
                start, stop = next(clock)
                array_sha256 = input_by_scenario[scenario].oracle.array_sha256
                gates.append(
                    SurfaceCalibrationOutputGate(
                        scenario_name=scenario,
                        strategy=strategy,
                        phase=phase,
                        resident_pair_sha256=pair_by_scenario[scenario].resident_pair_sha256,
                        invocation_nonce=NONCE,
                        worker_pid=WORKER_PID,
                        start_ns=start,
                        stop_ns=stop,
                        oracle_array_sha256=array_sha256,
                        output=_saved(
                            f"outputs/{scenario}/{strategy.value}-{phase}.npy",
                            shapes[scenario],
                            array_sha256,
                        ),
                        mismatched_element_count=0,
                        maximum_absolute_error=0.0,
                        maximum_normalized_error=0.0,
                    )
                )
    warmups = []
    warmup_sequence = 0
    repetitions = {}
    for scenario_index, scenario in enumerate(protocol.scenarios):
        for strategy in protocol.warmup_strategy_order(scenario_index):
            warmup_sequence += 1
            key = (scenario, strategy)
            repetitions[key] = repetitions.get(key, 0) + 1
            start, stop = next(clock)
            warmups.append(
                SurfaceCalibrationWarmupExecution(
                    sequence=warmup_sequence,
                    scenario_name=scenario,
                    scenario_position=scenario_index + 1,
                    strategy=strategy,
                    strategy_repetition=repetitions[key],
                    resident_pair_sha256=pair_by_scenario[scenario].resident_pair_sha256,
                    invocation_nonce=NONCE,
                    worker_pid=WORKER_PID,
                    start_ns=start,
                    stop_ns=stop,
                )
            )
    samples = []
    sample_sequence = 0
    for round_index in range(protocol.paired_rounds):
        for scenario_position, scenario in enumerate(protocol.scenario_order(round_index), start=1):
            for arm_position, strategy in enumerate(protocol.strategy_order(round_index), start=1):
                for call_index in range(protocol.calls_per_position):
                    sample_sequence += 1
                    start, stop = next(clock)
                    samples.append(
                        SurfaceCalibrationCallSample(
                            sequence=sample_sequence,
                            round_index=round_index,
                            scenario_name=scenario,
                            scenario_position=scenario_position,
                            strategy=strategy,
                            arm_position=arm_position,
                            call_index=call_index,
                            resident_pair_sha256=pair_by_scenario[scenario].resident_pair_sha256,
                            invocation_nonce=NONCE,
                            worker_pid=WORKER_PID,
                            start_ns=start,
                            stop_ns=stop,
                            duration_ns=stop - start,
                        )
                    )
    for phase in ("after_timing",):
        for scenario in protocol.scenarios:
            for strategy in protocol.strategies:
                start, stop = next(clock)
                array_sha256 = input_by_scenario[scenario].oracle.array_sha256
                gates.append(
                    SurfaceCalibrationOutputGate(
                        scenario_name=scenario,
                        strategy=strategy,
                        phase=phase,
                        resident_pair_sha256=pair_by_scenario[scenario].resident_pair_sha256,
                        invocation_nonce=NONCE,
                        worker_pid=WORKER_PID,
                        start_ns=start,
                        stop_ns=stop,
                        oracle_array_sha256=array_sha256,
                        output=_saved(
                            f"outputs/{scenario}/{strategy.value}-{phase}.npy",
                            shapes[scenario],
                            array_sha256,
                        ),
                        mismatched_element_count=0,
                        maximum_absolute_error=0.0,
                        maximum_normalized_error=0.0,
                    )
                )
    return MatmulCollectiveSurfaceCalibrationEvidence(
        protocol_id=protocol.protocol_id,
        protocol_file_sha256=PROTOCOL_FILE_SHA256,
        design_id=design.design_id,
        design_file_sha256=DESIGN_FILE_SHA256,
        correctness_parent_attempt_id=protocol.correctness_parent.attempt_id,
        correctness_parent_evidence_sha256=protocol.correctness_parent.evidence_sha256,
        correctness_parent_receipt_sha256=protocol.correctness_parent.receipt_sha256,
        calibration_execution_authority_sha256=AUTHORITY_SHA256,
        invocation_nonce=NONCE,
        worker_pid=WORKER_PID,
        continuity=tuple(continuity),
        inputs=tuple(inputs),
        resident_pairs=pairs,
        output_gates=tuple(gates),
        warmups=tuple(warmups),
        samples=tuple(samples),
    )


def _validate(evidence: MatmulCollectiveSurfaceCalibrationEvidence) -> None:
    validate_surface_calibration_evidence(
        evidence,
        default_matmul_collective_surface_calibration_protocol(),
        default_matmul_collective_surface_design_contract(),
        expected_protocol_file_sha256=PROTOCOL_FILE_SHA256,
        expected_design_file_sha256=DESIGN_FILE_SHA256,
        expected_execution_authority_sha256=AUTHORITY_SHA256,
        expected_invocation_nonce=NONCE,
        expected_worker_pid=WORKER_PID,
    )


def test_calibration_evidence_accepts_only_the_exact_raw_timeline() -> None:
    evidence = _evidence()

    _validate(evidence)
    assert len(evidence.continuity) == 32
    assert len(evidence.warmups) == 320
    assert len(evidence.samples) == 2560
    assert len(evidence.evidence_sha256) == 64


def test_calibration_evidence_rejects_reordered_timing_samples() -> None:
    evidence = _evidence()
    samples = list(evidence.samples)
    samples[0], samples[1] = samples[1], samples[0]

    with pytest.raises(ValueError, match="SAMPLE_SEQUENCE_MISMATCH"):
        _validate(evidence.model_copy(update={"samples": tuple(samples)}))


def test_calibration_evidence_rejects_parent_output_rebinding() -> None:
    evidence = _evidence()
    inputs = list(evidence.inputs)
    inputs[0] = inputs[0].model_copy(update={"parent_xla_array_sha256": "f" * 64})

    with pytest.raises(ValueError, match="OUTPUT_PARENT_MISMATCH"):
        _validate(evidence.model_copy(update={"inputs": tuple(inputs)}))


def test_calibration_evidence_rejects_overlapping_clock_intervals() -> None:
    evidence = _evidence()
    warmups = list(evidence.warmups)
    warmups[0] = warmups[0].model_copy(update={"start_ns": evidence.output_gates[31].stop_ns - 1})

    with pytest.raises(ValueError, match="CLOCK_ORDER_MISMATCH"):
        _validate(evidence.model_copy(update={"warmups": tuple(warmups)}))
