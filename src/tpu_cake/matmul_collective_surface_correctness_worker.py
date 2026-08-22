from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tpu_cake.frontend import schedule_sha256
from tpu_cake.identity import array_sha256
from tpu_cake.lowering import MatmulTile, lower_distributed_matmul
from tpu_cake.matmul_collective_surface_correctness import (
    correctness_sentinel_coordinates,
    make_correctness_operand_shard,
)
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
from tpu_cake.matmul_collective_surface_correctness_executor import (
    CORRECTNESS_EXECUTABLE_DEPENDENCIES,
    CORRECTNESS_EXECUTOR_SOURCE_PATH,
    CORRECTNESS_WORKER_SOURCE_PATH,
    SurfaceCorrectnessDevice,
    SurfaceCorrectnessExecutionAuthority,
    SurfaceCorrectnessRunIdentity,
    SurfaceCorrectnessWorkerRequest,
    SurfaceCorrectnessWorkerResult,
    _attempt_claim_path,
    _attempt_claim_payload,
    _compiler_environment,
    _metadata,
    _source_component_hashes,
    _write_model_exclusive,
    capture_correctness_source_authority,
    validate_correctness_execution_authority,
)
from tpu_cake.matmul_collective_surface_correctness_oracle import make_correctness_oracle
from tpu_cake.matmul_collective_surface_correctness_protocol import (
    MatmulCollectiveSurfaceCorrectnessProtocol,
)
from tpu_cake.matmul_collective_surface_prediction import (
    MatmulCollectiveSurfaceDesignContract,
    MatmulCollectiveSurfaceScenario,
)
from tpu_cake.matmul_collective_surface_runner import (
    CompileCaptureRecord,
    MatmulCollectiveSurfaceCompileReport,
    SurfaceCompileStatus,
    derive_surface_input_identities,
    make_compile_capture_record,
)
from tpu_cake.pallas_lowering import lower_physical_matmul_to_pallas
from tpu_cake.runner import MatmulCollectiveStrategy
from tpu_cake.workloads.distributed_matmul import distributed_matmul_schedule


@dataclass(frozen=True)
class _HostShardCapture:
    global_slice: tuple[slice, slice]
    local_shape: tuple[int, int]
    host_callback_payload_nbytes: int
    host_callback_payload_sha256: str
    sentinel_coordinates: tuple[tuple[int, int], ...]
    expected_sentinel_hex: tuple[str, ...]


@dataclass(frozen=True)
class _CompiledArm:
    strategy: MatmulCollectiveStrategy
    executable: Any
    mesh: Mesh
    continuity: SurfaceCompileContinuityEvidence


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _save_array_exclusive(
    root: Path, relative_path: str, value: np.ndarray
) -> SurfaceCorrectnessSavedArray:
    array = np.ascontiguousarray(value)
    if array.ndim != 2 or array.dtype != np.dtype("<f4") or not np.isfinite(array).all():
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_SAVED_ARRAY_INVALID")
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)
    return SurfaceCorrectnessSavedArray(
        path=relative_path,
        file_sha256=_file_sha256(path),
        array_sha256=array_sha256(array),
        shape=tuple(array.shape),
    )


def _canonical_slice(index: Any, shape: tuple[int, int]) -> tuple[slice, slice]:
    if not isinstance(index, tuple) or len(index) != 2:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_CALLBACK_INDEX_INVALID")
    canonical = []
    for value, size in zip(index, shape, strict=True):
        if not isinstance(value, slice):
            raise TypeError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_CALLBACK_INDEX_INVALID")
        start = 0 if value.start is None else value.start
        stop = size if value.stop is None else value.stop
        step = 1 if value.step is None else value.step
        if (
            not isinstance(start, int)
            or not isinstance(stop, int)
            or step != 1
            or not 0 <= start < stop <= size
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_CALLBACK_INDEX_INVALID")
        canonical.append(slice(start, stop, 1))
    return tuple(canonical)  # type: ignore[return-value]


def _bfloat16_hex(values: np.ndarray) -> tuple[str, ...]:
    array = np.ascontiguousarray(values)
    if array.dtype != np.dtype(ml_dtypes.bfloat16):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_SENTINEL_DTYPE_MISMATCH")
    return tuple(array[index : index + 1].tobytes().hex() for index in range(array.size))


class _OperandCallback:
    def __init__(
        self,
        *,
        request: SurfaceCorrectnessWorkerRequest,
        scenario: MatmulCollectiveSurfaceScenario,
        pattern: str,
        role: str,
    ) -> None:
        self.request = request
        self.scenario = scenario
        self.pattern = pattern
        self.role = role
        self.global_shape = (scenario.m, scenario.k) if role == "lhs" else (scenario.k, scenario.n)
        self.captures: dict[tuple[tuple[int, int], tuple[int, int]], _HostShardCapture] = {}

    def __call__(self, index: Any) -> np.ndarray:
        global_slice = _canonical_slice(index, self.global_shape)
        local_k = self.scenario.k // 8
        k_slice = global_slice[1] if self.role == "lhs" else global_slice[0]
        full_slice = global_slice[0] if self.role == "lhs" else global_slice[1]
        full_size = self.scenario.m if self.role == "lhs" else self.scenario.n
        if (
            full_slice.start != 0
            or full_slice.stop != full_size
            or k_slice.stop - k_slice.start != local_k
            or k_slice.start % local_k
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_CALLBACK_SHARDING_MISMATCH")
        shard_index = k_slice.start // local_k
        value = make_correctness_operand_shard(
            self.pattern,
            self.role,
            m=self.scenario.m,
            k=self.scenario.k,
            n=self.scenario.n,
            k_start=k_slice.start,
            k_stop=k_slice.stop,
        )
        expected_shape = tuple(item.stop - item.start for item in global_slice)
        if value.shape != expected_shape or value.dtype != np.dtype(ml_dtypes.bfloat16):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_CALLBACK_PAYLOAD_INVALID")
        coordinates = correctness_sentinel_coordinates(
            self.pattern,
            self.role,
            protocol_id=self.request.protocol.protocol_id,
            scenario_name=self.scenario.name,
            m=self.scenario.m,
            k=self.scenario.k,
            n=self.scenario.n,
            device_id=shard_index,
        )
        local_coordinates = tuple(
            (first - global_slice[0].start, second - global_slice[1].start)
            for first, second in coordinates
        )
        expected = np.asarray(
            value[
                np.asarray(tuple(item[0] for item in local_coordinates)),
                np.asarray(tuple(item[1] for item in local_coordinates)),
            ]
        )
        key = _slice_key(global_slice)
        capture = _HostShardCapture(
            global_slice=global_slice,
            local_shape=expected_shape,
            host_callback_payload_nbytes=value.nbytes,
            host_callback_payload_sha256=_sha256_bytes(
                np.ascontiguousarray(value).tobytes(order="C")
            ),
            sentinel_coordinates=coordinates,
            expected_sentinel_hex=_bfloat16_hex(expected),
        )
        if key in self.captures:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_CALLBACK_SHARD_REPEATED")
        self.captures[key] = capture
        return value


def _slice_key(value: tuple[slice, slice]) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple((item.start, item.stop) for item in value)  # type: ignore[return-value]


def _shard_index(value: Any, shape: tuple[int, int]) -> tuple[slice, slice]:
    return _canonical_slice(value, shape)


def _device_sentinel_hex(
    data: Any, local_coordinates: tuple[tuple[int, int], ...]
) -> tuple[str, ...]:
    first = np.asarray(tuple(value[0] for value in local_coordinates), dtype=np.int32)
    second = np.asarray(tuple(value[1] for value in local_coordinates), dtype=np.int32)
    selected = data[first, second]
    if hasattr(selected, "block_until_ready"):
        selected.block_until_ready()
    return _bfloat16_hex(np.asarray(jax.device_get(selected)))


def _materialize_operand(
    request: SurfaceCorrectnessWorkerRequest,
    scenario: MatmulCollectiveSurfaceScenario,
    pattern: str,
    role: str,
    mesh: Mesh,
) -> tuple[jax.Array, tuple[SurfaceCorrectnessShardIdentity, ...]]:
    spec = PartitionSpec(None, "t") if role == "lhs" else PartitionSpec("t", None)
    callback = _OperandCallback(
        request=request,
        scenario=scenario,
        pattern=pattern,
        role=role,
    )
    value = jax.make_array_from_callback(
        callback.global_shape,
        NamedSharding(mesh, spec),
        callback,
    )
    if len(callback.captures) != 8:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_CALLBACK_INVENTORY_MISMATCH")
    identities = []
    addressable = tuple(value.addressable_shards)
    if len(addressable) != 8:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_DEVICE_SHARD_INVENTORY_MISMATCH")
    for shard in addressable:
        device_id = int(shard.device.id)
        process_index = int(shard.device.process_index)
        global_slice = _shard_index(shard.index, callback.global_shape)
        capture = callback.captures.get(_slice_key(global_slice))
        if capture is None or process_index != 0:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_DEVICE_SHARD_MISMATCH")
        local_k = scenario.k // 8
        k_slice = global_slice[1] if role == "lhs" else global_slice[0]
        shard_index = k_slice.start // local_k
        if shard_index != device_id:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_DEVICE_SLICE_MISMATCH")
        local_coordinates = tuple(
            (first - global_slice[0].start, second - global_slice[1].start)
            for first, second in capture.sentinel_coordinates
        )
        observed_hex = _device_sentinel_hex(shard.data, local_coordinates)
        identities.append(
            SurfaceCorrectnessShardIdentity(
                role=role,
                shard_index=shard_index,
                device_id=device_id,
                process_index=process_index,
                global_shape=callback.global_shape,
                sharding=(
                    "PartitionSpec(None, 't')" if role == "lhs" else "PartitionSpec('t', None)"
                ),
                global_slice=tuple(
                    SurfaceCorrectnessSlice(start=item.start, stop=item.stop)
                    for item in global_slice
                ),
                local_shape=capture.local_shape,
                host_callback_payload_nbytes=capture.host_callback_payload_nbytes,
                host_callback_payload_sha256=capture.host_callback_payload_sha256,
                sentinels=tuple(
                    SurfaceCorrectnessSentinel(
                        ordinal=ordinal,
                        global_coordinate=coordinate,
                        local_coordinate=local_coordinate,
                        expected_bfloat16_hex=expected,
                        observed_bfloat16_hex=observed,
                    )
                    for ordinal, (coordinate, local_coordinate, expected, observed) in enumerate(
                        zip(
                            capture.sentinel_coordinates,
                            local_coordinates,
                            capture.expected_sentinel_hex,
                            observed_hex,
                            strict=True,
                        )
                    )
                ),
            )
        )
    identities.sort(key=lambda item: item.device_id)
    return value, tuple(identities)


def _verify_resident_sentinels(
    value: jax.Array,
    identities: tuple[SurfaceCorrectnessShardIdentity, ...],
) -> None:
    shards = tuple(value.addressable_shards)
    if len(shards) != len(identities):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_RESIDENT_INVENTORY_MISMATCH")
    by_device = {int(shard.device.id): shard for shard in shards}
    if tuple(sorted(by_device)) != tuple(range(8)):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_RESIDENT_DEVICE_MISMATCH")
    for identity in identities:
        shard = by_device[identity.device_id]
        global_slice = _shard_index(shard.index, identity.global_shape)
        expected_slice = tuple(
            slice(value.start, value.stop, value.step) for value in identity.global_slice
        )
        if (
            int(shard.device.process_index) != identity.process_index
            or global_slice != expected_slice
        ):
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_RESIDENT_SLICE_MISMATCH")
        observed = _device_sentinel_hex(
            shard.data,
            tuple(value.local_coordinate for value in identity.sentinels),
        )
        expected = tuple(value.observed_bfloat16_hex for value in identity.sentinels)
        if observed != expected:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_RESIDENT_SENTINEL_CHANGED")


def _parent_consensus(
    report: MatmulCollectiveSurfaceCompileReport,
    scenario_name: str,
    strategy: MatmulCollectiveStrategy,
) -> CompileCaptureRecord:
    captures = tuple(
        value
        for value in report.captures
        if value.scenario_name == scenario_name and value.strategy is strategy
    )
    if (
        len(captures) != 2
        or tuple(value.repetition for value in captures) != (1, 2)
        or any(value.status is not SurfaceCompileStatus.SUCCEEDED for value in captures)
        or len({value.distributed_schedule_sha256 for value in captures}) != 1
        or len({value.physical_schedule_sha256 for value in captures}) != 1
        or len({value.pallas_source_sha256 for value in captures}) != 1
        or len({value.semantic_stablehlo_sha256 for value in captures}) != 1
        or len({value.semantic_compiler_hlo_sha256 for value in captures}) != 1
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_PARENT_CONSENSUS_INVALID")
    return captures[0]


def _compile_arm(
    root: Path,
    request: SurfaceCorrectnessWorkerRequest,
    report: MatmulCollectiveSurfaceCompileReport,
    scenario: MatmulCollectiveSurfaceScenario,
    strategy: MatmulCollectiveStrategy,
) -> _CompiledArm:
    distributed = distributed_matmul_schedule(
        mesh_size=request.design.mesh_size,
        m=scenario.m,
        k=scenario.k,
        n=scenario.n,
    )
    distributed.verify()
    physical = lower_distributed_matmul(
        distributed,
        tile=MatmulTile(scenario.tile_m, scenario.tile_n),
        collective_implementation=strategy.lowering_implementation(),
    )
    plan = lower_physical_matmul_to_pallas(physical)
    executable, mesh = plan.build(interpret=False)
    if plan.mesh_axis != "t":
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_MESH_AXIS_MISMATCH")
    abstract_inputs = (
        jax.ShapeDtypeStruct(
            (scenario.m, scenario.k),
            jnp.bfloat16,
            sharding=NamedSharding(mesh, PartitionSpec(None, "t")),
        ),
        jax.ShapeDtypeStruct(
            (scenario.k, scenario.n),
            jnp.bfloat16,
            sharding=NamedSharding(mesh, PartitionSpec("t", None)),
        ),
    )
    lowered = executable.lower(*abstract_inputs)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    compiled = lowered.compile()
    compiler_hlo = compiled.as_text()
    if not isinstance(compiler_hlo, str) or not compiler_hlo:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_COMPILER_HLO_UNAVAILABLE")
    parent = _parent_consensus(report, scenario.name, strategy)
    input_identity = next(
        value
        for value in derive_surface_input_identities(request.design)
        if value.scenario_name == scenario.name
    )
    if input_identity.input_contract_sha256 != parent.input_contract_sha256:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_INPUT_IDENTITY_MISMATCH")
    fresh = make_compile_capture_record(
        scenario_name=scenario.name,
        strategy=strategy,
        repetition=1,
        input_contract_sha256=input_identity.input_contract_sha256,
        distributed_schedule_sha256=schedule_sha256(distributed),
        physical_schedule_sha256=plan.schedule_sha256,
        pallas_source_sha256=plan.source_sha256(),
        stablehlo=stablehlo,
        compiler_hlo=compiler_hlo,
    )
    stablehlo_path = f"continuity/{scenario.name}/{strategy.value}/stablehlo.txt"
    compiler_hlo_path = f"continuity/{scenario.name}/{strategy.value}/compiler_hlo.txt"
    _write_bytes_exclusive(root / stablehlo_path, fresh.stablehlo.encode())
    _write_bytes_exclusive(root / compiler_hlo_path, fresh.compiler_hlo.encode())
    continuity = SurfaceCompileContinuityEvidence(
        scenario_name=scenario.name,
        strategy=strategy,
        stablehlo_path=stablehlo_path,
        stablehlo_file_sha256=_file_sha256(root / stablehlo_path),
        compiler_hlo_path=compiler_hlo_path,
        compiler_hlo_file_sha256=_file_sha256(root / compiler_hlo_path),
        parent_distributed_schedule_sha256=parent.distributed_schedule_sha256,
        observed_distributed_schedule_sha256=fresh.distributed_schedule_sha256,
        parent_physical_schedule_sha256=parent.physical_schedule_sha256,
        observed_physical_schedule_sha256=fresh.physical_schedule_sha256,
        parent_pallas_source_sha256=parent.pallas_source_sha256,
        observed_pallas_source_sha256=fresh.pallas_source_sha256,
        parent_semantic_stablehlo_sha256=parent.semantic_stablehlo_sha256,
        observed_semantic_stablehlo_sha256=fresh.semantic_stablehlo_sha256,
        parent_semantic_compiler_hlo_sha256=parent.semantic_compiler_hlo_sha256,
        observed_semantic_compiler_hlo_sha256=fresh.semantic_compiler_hlo_sha256,
    )
    return _CompiledArm(strategy=strategy, executable=compiled, mesh=mesh, continuity=continuity)


def _validate_meshes(compiled: tuple[_CompiledArm, ...]) -> Mesh:
    first = compiled[0].mesh
    expected_devices = tuple(int(value.id) for value in first.devices.flat)
    if expected_devices != tuple(range(8)) or any(
        tuple(int(value.id) for value in arm.mesh.devices.flat) != expected_devices
        or arm.mesh.axis_names != first.axis_names
        for arm in compiled[1:]
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_COMPILED_MESH_MISMATCH")
    return first


def _execution_order(
    strategies: tuple[MatmulCollectiveStrategy, MatmulCollectiveStrategy],
    pattern_index: int,
) -> tuple[MatmulCollectiveStrategy, ...]:
    first, second = strategies
    return (
        (first, second, second, first) if pattern_index % 2 == 0 else (second, first, first, second)
    )


def _validate_output(value: jax.Array, scenario: MatmulCollectiveSurfaceScenario) -> None:
    sharding = getattr(value, "sharding", None)
    if (
        tuple(value.shape) != (scenario.m, scenario.n)
        or value.dtype != jnp.float32
        or not isinstance(sharding, NamedSharding)
        or sharding.spec != PartitionSpec(None, "t")
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_OUTPUT_ABI_MISMATCH")
    shards = tuple(sorted(value.addressable_shards, key=lambda shard: int(shard.device.id)))
    local_n = scenario.n // 8
    if len(shards) != 8 or any(
        int(shard.device.id) != index
        or int(shard.device.process_index) != 0
        or _shard_index(shard.index, (scenario.m, scenario.n))
        != (slice(0, scenario.m, 1), slice(index * local_n, (index + 1) * local_n, 1))
        for index, shard in enumerate(shards)
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_OUTPUT_SHARDS_MISMATCH")


def _error_metrics(
    candidate: np.ndarray,
    oracle: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[int, float, float]:
    if (
        candidate.shape != oracle.shape
        or candidate.dtype != np.dtype("<f4")
        or oracle.dtype != np.dtype("<f4")
        or not candidate.flags.c_contiguous
        or not oracle.flags.c_contiguous
        or not np.isfinite(candidate).all()
        or not np.isfinite(oracle).all()
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_COMPARISON_ABI_MISMATCH")
    absolute = np.abs(candidate - oracle)
    threshold = absolute_tolerance + relative_tolerance * np.abs(oracle)
    normalized = absolute / threshold
    return (
        int(np.count_nonzero(absolute > threshold)),
        float(absolute.max()),
        float(normalized.max()),
    )


def _run_case(
    root: Path,
    request: SurfaceCorrectnessWorkerRequest,
    scenario: MatmulCollectiveSurfaceScenario,
    pattern: str,
    pattern_index: int,
    compiled: dict[MatmulCollectiveStrategy, _CompiledArm],
    mesh: Mesh,
    sequence: int,
) -> tuple[SurfaceCorrectnessCaseEvidence, int]:
    lhs, lhs_shards = _materialize_operand(request, scenario, pattern, "lhs", mesh)
    rhs, rhs_shards = _materialize_operand(request, scenario, pattern, "rhs", mesh)
    inputs = SurfaceCorrectnessInputCase(
        scenario_name=scenario.name,
        pattern=pattern,
        protocol_id=request.protocol.protocol_id,
        pattern_contract_sha256=request.protocol.patterns.contract_sha256,
        lhs_shards=lhs_shards,
        rhs_shards=rhs_shards,
    )
    oracle_value = make_correctness_oracle(
        pattern,
        m=scenario.m,
        k=scenario.k,
        n=scenario.n,
    )
    oracle = _save_array_exclusive(
        root,
        f"outputs/{scenario.name}/{pattern}/oracle.npy",
        oracle_value,
    )
    repetitions = {strategy: 0 for strategy in request.protocol.strategies}
    executions = []
    for position, strategy in enumerate(
        _execution_order(request.protocol.strategies, pattern_index),
        start=1,
    ):
        repetitions[strategy] += 1
        sequence += 1
        device_output = compiled[strategy].executable(lhs, rhs)
        device_output.block_until_ready()
        _validate_output(device_output, scenario)
        _verify_resident_sentinels(lhs, lhs_shards)
        _verify_resident_sentinels(rhs, rhs_shards)
        candidate = np.ascontiguousarray(
            np.asarray(jax.device_get(device_output)), dtype=np.float32
        )
        output = _save_array_exclusive(
            root,
            (f"outputs/{scenario.name}/{pattern}/{strategy.value}-{repetitions[strategy]}.npy"),
            candidate,
        )
        mismatches, maximum_absolute, maximum_normalized = _error_metrics(
            candidate,
            oracle_value,
            absolute_tolerance=request.protocol.absolute_tolerance,
            relative_tolerance=request.protocol.relative_tolerance,
        )
        executions.append(
            SurfaceCorrectnessCandidateExecution(
                sequence=sequence,
                position=position,
                strategy=strategy,
                strategy_repetition=repetitions[strategy],
                invocation_nonce=request.invocation_nonce,
                worker_pid=os.getpid(),
                fresh_compile_record_sha256=compiled[strategy].continuity.compile_record_sha256,
                lhs_identity_set_sha256=inputs.lhs_identity_set_sha256,
                rhs_identity_set_sha256=inputs.rhs_identity_set_sha256,
                oracle_array_sha256=oracle.array_sha256,
                output=output,
                mismatched_element_count=mismatches,
                maximum_absolute_error=maximum_absolute,
                maximum_normalized_error=maximum_normalized,
            )
        )
    return SurfaceCorrectnessCaseEvidence(
        input=inputs, oracle=oracle, executions=tuple(executions)
    ), sequence


def execute_correctness_worker(
    root: Path,
    request: SurfaceCorrectnessWorkerRequest,
    parent_report: MatmulCollectiveSurfaceCompileReport,
) -> SurfaceCorrectnessWorkerResult:
    request = SurfaceCorrectnessWorkerRequest.model_validate(
        request.model_dump(mode="python", exclude_computed_fields=True)
    )
    if request.split.value != "calibration":
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_HOLDOUT_NOT_AUTHORIZED")
    parent_report = MatmulCollectiveSurfaceCompileReport.model_validate(
        parent_report.model_dump(mode="python", exclude_computed_fields=True)
    )
    if (
        parent_report.design_id != request.design.design_id
        or parent_report.report_sha256 != request.protocol.parent_compile.compile_report_sha256
        or parent_report.source_authority_sha256
        != request.protocol.parent_compile.source_authority_sha256
        or parent_report.execution_authority_sha256
        != request.protocol.parent_compile.execution_authority_sha256
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_PARENT_REPORT_MISMATCH")
    scenario_names = (
        request.protocol.calibration_scenarios
        if request.split.value == "calibration"
        else request.protocol.holdout_scenarios
    )
    scenarios = {value.name: value for value in request.design.scenarios}
    continuity = []
    cases = []
    sequence = 0
    for scenario_name in scenario_names:
        scenario = scenarios[scenario_name]
        compiled_arms = tuple(
            _compile_arm(root, request, parent_report, scenario, strategy)
            for strategy in request.protocol.strategies
        )
        mesh = _validate_meshes(compiled_arms)
        compiled = {value.strategy: value for value in compiled_arms}
        continuity.extend(value.continuity for value in compiled_arms)
        for pattern_index, pattern in enumerate(request.protocol.patterns.ordered_patterns):
            case, sequence = _run_case(
                root,
                request,
                scenario,
                pattern,
                pattern_index,
                compiled,
                mesh,
                sequence,
            )
            cases.append(case)
    evidence = MatmulCollectiveSurfaceCorrectnessEvidence(
        protocol_id=request.protocol.protocol_id,
        protocol_file_sha256=_file_sha256(
            Path(request.design.compilation_source_root)
            / "contracts/matmul-collective-surface-correctness-v1.json"
        ),
        split=request.split,
        parent_compile_manifest_file_sha256=request.protocol.parent_compile.manifest_file_sha256,
        correctness_execution_authority_sha256=request.execution_authority_sha256,
        continuity=tuple(continuity),
        cases=tuple(cases),
    )
    validate_surface_correctness_evidence(
        evidence,
        request.protocol,
        request.design,
        expected_protocol_file_sha256=evidence.protocol_file_sha256,
        expected_execution_authority_sha256=request.execution_authority_sha256,
        expected_invocation_nonce=request.invocation_nonce,
        expected_worker_pid=os.getpid(),
    )
    return SurfaceCorrectnessWorkerResult(
        attempt_id=request.attempt_id,
        split=request.split,
        invocation_nonce=request.invocation_nonce,
        worker_pid=os.getpid(),
        execution_authority_sha256=request.execution_authority_sha256,
        evidence=evidence,
    )


def _validate_loaded_tpu_cake_sources(
    repository_root: Path,
    source_blobs: dict[str, bytes],
) -> None:
    expected = {path: source_blobs[path] for path in CORRECTNESS_EXECUTABLE_DEPENDENCIES}
    source_root = (repository_root / "src").resolve()
    observed: set[str] = set()
    for name, module in tuple(sys.modules.items()):
        if name != "tpu_cake" and not name.startswith("tpu_cake."):
            continue
        module_file = getattr(module, "__file__", None)
        if not module_file:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_LOADED_SOURCE_MISSING")
        path = Path(module_file)
        try:
            relative = path.resolve().relative_to(source_root).as_posix()
        except ValueError as error:
            raise ValueError(
                "MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_LOADED_SOURCE_OUTSIDE_ROOT"
            ) from error
        if path.is_symlink() or relative not in expected or path.read_bytes() != expected[relative]:
            raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_LOADED_SOURCE_MISMATCH")
        observed.add(relative)
    worker_relative = CORRECTNESS_WORKER_SOURCE_PATH.removeprefix("src/")
    running_worker = Path(__file__)
    if (
        running_worker.is_symlink()
        or running_worker.resolve() != (repository_root / CORRECTNESS_WORKER_SOURCE_PATH).resolve()
        or running_worker.read_bytes() != expected[worker_relative]
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_LOADED_SOURCE_MISMATCH")
    observed.add(worker_relative)
    required = {
        "tpu_cake/__init__.py",
        CORRECTNESS_EXECUTOR_SOURCE_PATH.removeprefix("src/"),
        worker_relative,
    }
    if not required <= observed:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_LOADED_SOURCE_INCOMPLETE")


def capture_execution_authority(
    repository_root: Path,
    protocol: MatmulCollectiveSurfaceCorrectnessProtocol,
    design: MatmulCollectiveSurfaceDesignContract,
) -> tuple[SurfaceCorrectnessExecutionAuthority, dict[str, bytes]]:
    source, source_blobs = capture_correctness_source_authority(repository_root, protocol)
    component_hashes = _source_component_hashes(source_blobs)
    authority = SurfaceCorrectnessExecutionAuthority(
        protocol_id=protocol.protocol_id,
        protocol_file_sha256=_sha256_bytes(
            source_blobs["contracts/matmul-collective-surface-correctness-v1.json"]
        ),
        source=source,
        **component_hashes,
        project=_metadata("project/project-id"),
        zone=_metadata("instance/zone").rsplit("/", maxsplit=1)[-1],
        hostname=_metadata("instance/name"),
        numeric_project_id=_metadata("project/numeric-project-id"),
        instance_id=_metadata("instance/id"),
        instance_hostname=_metadata("instance/hostname"),
        machine_type=_metadata("instance/machine-type").rsplit("/", maxsplit=1)[-1],
        cpu_platform=_metadata("instance/cpu-platform"),
        compiler_environment=_compiler_environment(design),
        devices=tuple(
            SurfaceCorrectnessDevice(
                id=int(device.id),
                process_index=int(device.process_index),
                platform=str(device.platform),
                device_kind=str(device.device_kind),
            )
            for device in jax.devices()
        ),
    )
    validate_correctness_execution_authority(
        authority,
        protocol,
        design,
        source_blobs,
    )
    running = Path(__file__)
    expected = source_blobs[CORRECTNESS_WORKER_SOURCE_PATH.removeprefix("src/")]
    if (
        running.is_symlink()
        or running.resolve() != (repository_root / CORRECTNESS_WORKER_SOURCE_PATH).resolve()
        or running.read_bytes() != expected
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_RUNNING_WORKER_MISMATCH")
    _validate_loaded_tpu_cake_sources(repository_root, source_blobs)
    return authority, source_blobs


def _validate_empty_compilation_cache() -> Path:
    raw = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if raw is None:
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_CACHE_UNDECLARED")
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir() or any(path.iterdir()):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_CACHE_NOT_EMPTY")
    return path


def _validate_worker_authorization(
    root: Path,
    request_path: Path,
    request: SurfaceCorrectnessWorkerRequest,
) -> SurfaceCorrectnessRunIdentity:
    root = root.resolve(strict=True)
    if request_path.resolve(strict=True) != (root / "worker-request.json").resolve(strict=True):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_WORKER_REQUEST_PATH_MISMATCH")
    identity = SurfaceCorrectnessRunIdentity.model_validate_json(
        (root / "run_identity.json").read_text()
    )
    authority = SurfaceCorrectnessExecutionAuthority.model_validate_json(
        (root / "execution_authority.json").read_text()
    )
    started = {
        "attempt_id": request.attempt_id,
        "invocation_nonce": request.invocation_nonce,
        "split": request.split.value,
        "state": "started",
    }
    if (
        identity.attempt_id != request.attempt_id
        or identity.protocol_id != request.protocol.protocol_id
        or identity.split is not request.split
        or identity.execution_authority_sha256 != request.execution_authority_sha256
        or identity.output_root != str(root)
        or request.parent_snapshot_path != str(root / "parent_compile")
        or json.loads((root / "STARTED.json").read_text()) != started
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_WORKER_AUTHORIZATION_MISMATCH")
    expected_claim = _attempt_claim_path(request.protocol, request.split)
    claim = Path(identity.attempt_claim_path)
    archived_claim = root / "attempt_claim.json"
    if (
        claim != expected_claim
        or claim.is_symlink()
        or not claim.is_file()
        or claim.stat().st_nlink != 1
        or archived_claim.is_symlink()
        or not archived_claim.is_file()
        or archived_claim.stat().st_nlink != 1
        or _file_sha256(claim) != identity.attempt_claim_sha256
        or _file_sha256(archived_claim) != identity.attempt_claim_sha256
        or archived_claim.read_bytes() != claim.read_bytes()
        or json.loads(claim.read_text())
        != _attempt_claim_payload(
            root,
            request.attempt_id,
            request.protocol,
            request.split,
            authority.source.source_commit,
        )
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_WORKER_CLAIM_MISMATCH")
    return identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--probe-output", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--design", type=Path)
    args = parser.parse_args()
    if args.probe_output is not None:
        if (
            args.protocol is None
            or args.design is None
            or args.root is not None
            or args.request is not None
        ):
            parser.error("authority probe requires --probe-output, --protocol, and --design only")
        protocol = MatmulCollectiveSurfaceCorrectnessProtocol.model_validate_json(
            args.protocol.read_text()
        )
        design = MatmulCollectiveSurfaceDesignContract.model_validate_json(args.design.read_text())
        authority, _ = capture_execution_authority(
            Path(design.compilation_source_root),
            protocol,
            design,
        )
        _write_model_exclusive(args.probe_output, authority)
        return
    if (
        args.root is None
        or args.request is None
        or any(value is not None for value in (args.protocol, args.design))
    ):
        parser.error("worker execution requires --root and --request only")
    request = SurfaceCorrectnessWorkerRequest.model_validate_json(args.request.read_text())
    if request.split.value != "calibration":
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_HOLDOUT_NOT_AUTHORIZED")
    _validate_worker_authorization(args.root, args.request, request)
    _validate_empty_compilation_cache()
    recorded_authority = SurfaceCorrectnessExecutionAuthority.model_validate_json(
        (args.root / "execution_authority.json").read_text()
    )
    observed_authority, source_blobs = capture_execution_authority(
        Path(request.design.compilation_source_root),
        request.protocol,
        request.design,
    )
    if (
        observed_authority != recorded_authority
        or request.execution_authority_sha256 != recorded_authority.authority_sha256
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_WORKER_AUTHORITY_MISMATCH")
    parent_report = MatmulCollectiveSurfaceCompileReport.model_validate_json(
        (Path(request.parent_snapshot_path) / "compile_report.json").read_text()
    )
    result = execute_correctness_worker(args.root, request, parent_report)
    _validate_loaded_tpu_cake_sources(
        Path(request.design.compilation_source_root),
        source_blobs,
    )
    if (
        SurfaceCorrectnessExecutionAuthority.model_validate_json(
            (args.root / "execution_authority.json").read_text()
        )
        != recorded_authority
    ):
        raise ValueError("MATMUL_COLLECTIVE_SURFACE_CORRECTNESS_WORKER_AUTHORITY_CHANGED")
    payload = result.model_dump_json(indent=2, exclude_computed_fields=True).encode() + b"\n"
    _write_bytes_exclusive(args.root / "worker-result.json", payload)


if __name__ == "__main__":
    main()
