from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import ml_dtypes
import numpy as np
from jax.sharding import PartitionSpec

from tpu_cake.canonical import canonical_text
from tpu_cake.compiler_analysis import (
    analyze_compiler_collectives,
    capture_compiler_analysis,
)
from tpu_cake.cost_model import tpu7x_tensorcore_rates
from tpu_cake.dialects.tpu_schedule import CollectiveOp, VectorComputeOp
from tpu_cake.identity import array_sha256, json_sha256
from tpu_cake.ledger import EvidenceRun, RunState
from tpu_cake.physical_cost_model import analyze_physical_kernel
from tpu_cake.runner import _runtime_identity
from tpu_cake.seqax_numerical import (
    SeqaxBf16NumericalPolicy,
    SeqaxBf16ScenarioParameters,
    assess_seqax_bf16_candidate_checkpoints,
    assess_seqax_bf16_final_outputs,
    encode_seqax_bf16_checkpoint,
    rounded_mathematical_silu_bf16,
    seqax_bf16_checkpoint_contract,
)
from tpu_cake.seqax_pallas_lowering import (
    SeqaxPallasPlan,
    lower_seqax_physical_to_pallas,
    place_inputs,
)
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.seqax_silu_fusion import (
    SeqaxSiluFusionDesignContract,
    SeqaxSiluFusionPlanContract,
)
from tpu_cake.seqax_silu_fusion_compiler import (
    SeqaxSiluFusionCompilerCandidate,
    SeqaxSiluFusionCompilerPair,
    analyze_seqax_silu_fusion_compiler_hlo,
    exact_integral_ring_equivalent_bytes,
    live_seqax_silu_fusion_compiler_hlo,
    validate_seqax_silu_fusion_compiler_collectives,
)
from tpu_cake.seqax_silu_fusion_correctness import (
    SeqaxSiluFusionCandidateCorrectness,
    SeqaxSiluFusionCheckpointMetrics,
    SeqaxSiluFusionCorrectnessDevice,
    SeqaxSiluFusionCorrectnessHost,
    SeqaxSiluFusionCorrectnessObservation,
    SeqaxSiluFusionCorrectnessPlan,
    SeqaxSiluFusionCorrectnessResult,
    SeqaxSiluFusionCorrectnessWorkerRequest,
    SeqaxSiluFusionCorrectnessWorkerResult,
    SeqaxSiluFusionFinalOutputMetrics,
)
from tpu_cake.workloads.seqax_forward import (
    SeqaxFeedForwardVectorExecution,
    SeqaxNumericalSemantics,
    seqax_forward_schedule,
)
from tpu_cake.workloads.seqax_oracle import (
    seqax_forward_canonical_reference,
    seqax_forward_inputs,
)

_DESIGN_PATH = Path("contracts/seqax-silu-fusion-design-v1.json")
_PAIR_PATH = Path("contracts/seqax-silu-fusion-compiler-pair-v1.json")
_CHECKPOINT_SPECS = (
    PartitionSpec("d", None, None),
    PartitionSpec("d", None, None),
    PartitionSpec("d", None, None),
    PartitionSpec("d", None, None),
    PartitionSpec("d", None, None),
    PartitionSpec("d", None, "t"),
    PartitionSpec("d", None, "t"),
    PartitionSpec("d", None, "t"),
    PartitionSpec("d", None, "t"),
    PartitionSpec("d", None, "t"),
    PartitionSpec("d", None, "t"),
    PartitionSpec("d", None, None),
    PartitionSpec("d", None, None),
)
_COMPILER_ENVIRONMENT_PREFIXES = ("JAX_", "XLA_", "PJRT_", "LIBTPU_")


@dataclass(frozen=True)
class _PreparedCandidate:
    expected: SeqaxSiluFusionPlanContract
    distributed: Any
    physical: Any
    plan: SeqaxPallasPlan


@dataclass(frozen=True)
class _CompiledPath:
    prepared: _PreparedCandidate
    executable: Any
    mesh: Any
    stablehlo: str
    pre_optimization_hlo: str
    compiler_hlo: str


@dataclass(frozen=True)
class _CompiledCandidate:
    prepared: _PreparedCandidate
    uninstrumented: _CompiledPath
    instrumented: _CompiledPath
    record: SeqaxSiluFusionCorrectnessPlan


@dataclass(frozen=True)
class _CompilerAnalysisExecutable:
    executable: Any
    memory: Any

    def as_text(self) -> str:
        return self.executable.as_text()

    def cost_analysis(self) -> Any:
        return self.executable.cost_analysis()

    def memory_analysis(self) -> Any:
        return self.memory


class _RejectMetadataRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl) -> None:
        raise ValueError(
            f"SEQAX_SILU_FUSION_CORRECTNESS_METADATA_REDIRECT code={code} url={newurl}"
        )


def _metadata(path: str) -> str:
    request = urllib.request.Request(
        f"http://metadata.google.internal/computeMetadata/v1/{path}",
        headers={"Metadata-Flavor": "Google"},
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _RejectMetadataRedirects(),
    )
    with opener.open(request, timeout=5) as response:
        if response.headers.get("Metadata-Flavor") != "Google":
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_METADATA_HEADER_MISSING")
        payload = response.read(4097)
    if len(payload) > 4096:
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_METADATA_RESPONSE_TOO_LARGE")
    return payload.decode().strip()


def _host_identity() -> SeqaxSiluFusionCorrectnessHost:
    return SeqaxSiluFusionCorrectnessHost(
        project=_metadata("project/project-id"),
        numeric_project_id=_metadata("project/numeric-project-id"),
        zone=_metadata("instance/zone").rsplit("/", maxsplit=1)[-1],
        hostname=_metadata("instance/name"),
        instance_hostname=_metadata("instance/hostname"),
        machine_type=_metadata("instance/machine-type").rsplit("/", maxsplit=1)[-1],
        instance_id=_metadata("instance/id"),
        cpu_platform=_metadata("instance/cpu-platform"),
    )


def _devices() -> tuple[SeqaxSiluFusionCorrectnessDevice, ...]:
    return tuple(
        SeqaxSiluFusionCorrectnessDevice(
            id=int(device.id),
            process_index=int(device.process_index),
            platform=str(device.platform),
            device_kind=str(device.device_kind),
        )
        for device in jax.devices()
    )


def _environment(
    root: Path,
    request: SeqaxSiluFusionCorrectnessWorkerRequest,
) -> tuple[dict[str, str], dict[str, str]]:
    expected_worker = request.contract.worker_environment
    expected_compiler = request.contract.compiler_environment
    expected = {
        **expected_worker,
        **expected_compiler,
        "PYTHONPATH": str(root / "source" / "committed" / "src"),
    }
    observed = {key: os.environ.get(key) for key in expected}
    forbidden = {
        key: value
        for key, value in os.environ.items()
        if (key == "TPU_LIBRARY_PATH" or key.startswith(_COMPILER_ENVIRONMENT_PREFIXES))
        and key not in expected
    }
    if observed != expected or forbidden:
        raise ValueError(
            "SEQAX_SILU_FUSION_CORRECTNESS_ENVIRONMENT_MISMATCH "
            f"observed={observed} forbidden={forbidden}"
        )
    return dict(expected_worker), dict(expected_compiler)


def _write_bytes_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_exclusive(path: Path, value: object) -> None:
    _write_bytes_exclusive(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
    )


def _save_array(path: Path, value: np.ndarray) -> str:
    array = np.asarray(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return array_sha256(array)


def _canonical_hlo(value: str) -> str:
    return value.rstrip("\n") + "\n"


def _load_bound_compiler_evidence(
    root: Path,
    request: SeqaxSiluFusionCorrectnessWorkerRequest,
) -> tuple[SeqaxSiluFusionDesignContract, SeqaxSiluFusionCompilerPair]:
    bundle = root / "source" / "committed"
    design_path = bundle / _DESIGN_PATH
    pair_path = bundle / _PAIR_PATH
    design_blob = design_path.read_bytes()
    pair_blob = pair_path.read_bytes()
    contract = request.contract
    if (
        hashlib.sha256(design_blob).hexdigest() != contract.compiler_design_sha256
        or hashlib.sha256(pair_blob).hexdigest() != contract.compiler_pair_sha256
    ):
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_COMPILER_EVIDENCE_HASH_MISMATCH")
    design = SeqaxSiluFusionDesignContract.model_validate_json(design_blob)
    pair = SeqaxSiluFusionCompilerPair.model_validate_json(pair_blob)
    if (
        design.design_id != contract.compiler_design_id
        or pair.design_id != contract.compiler_design_id
        or pair.pair_id != contract.compiler_pair_id
        or tuple(value.capture_id for value in pair.captures) != contract.compiler_capture_ids
        or pair.captures[0].candidate_semantic_ids != contract.candidate_semantic_ids
        or pair.captures[1].candidate_semantic_ids != contract.candidate_semantic_ids
    ):
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_COMPILER_EVIDENCE_MISMATCH")
    return design, pair


def _prepare_candidates(
    request: SeqaxSiluFusionCorrectnessWorkerRequest,
    design: SeqaxSiluFusionDesignContract,
) -> tuple[_PreparedCandidate, ...]:
    parameters = dict(request.contract.parameters)
    parameters["numerical_semantics"] = SeqaxNumericalSemantics(parameters["numerical_semantics"])
    prepared = []
    for expected in design.candidates:
        distributed = seqax_forward_schedule(
            **parameters,
            feed_forward_fusion=expected.candidate,
            feed_forward_vector_execution=(SeqaxFeedForwardVectorExecution.PALLAS_FULL_LOCAL),
            residual_norm_strategy=request.contract.residual_norm_strategy,
        )
        physical = lower_seqax_forward_to_physical(distributed).module
        plan = lower_seqax_physical_to_pallas(distributed, physical)
        vectors = tuple(
            operation for operation in physical.walk() if isinstance(operation, VectorComputeOp)
        )
        owned = tuple(
            operation.function.data for operation in vectors if operation.implementation is not None
        )
        collectives = Counter(
            operation.kind.data.value
            for operation in physical.walk()
            if isinstance(operation, CollectiveOp)
        )
        resources = analyze_physical_kernel(physical, hardware=tpu7x_tensorcore_rates())
        observed = (
            plan.distributed_schedule_sha256,
            plan.physical_schedule_sha256,
            plan.source_sha256(),
            json_sha256(plan.manifest()),
            tuple(f"seqax_strict_bf16_{value}" for value in owned),
            len(owned),
            len(vectors),
            plan.pallas_region_count,
            collectives.get("all_gather", 0),
            collectives.get("all_reduce", 0),
            collectives.get("reduce_scatter", 0),
            resources.memory.allocated_vmem_bytes_per_device,
            resources.memory.peak_live_vmem_bytes_per_device,
            exact_integral_ring_equivalent_bytes(
                resources.devices[0].collective_ring_equivalent_bytes
            ),
        )
        required = (
            expected.distributed_schedule_sha256,
            expected.physical_schedule_sha256,
            expected.pallas_source_sha256,
            expected.pallas_manifest_sha256,
            expected.expected_strict_vector_kernels,
            expected.expected_strict_vector_regions,
            expected.expected_physical_vector_operations,
            expected.expected_pallas_regions,
            expected.expected_all_gathers,
            expected.expected_all_reduces,
            expected.expected_reduce_scatters,
            expected.allocated_vmem_bytes_per_device,
            expected.peak_live_vmem_bytes_per_device,
            expected.ring_equivalent_ici_bytes_per_device,
        )
        if observed != required:
            raise ValueError(
                "SEQAX_SILU_FUSION_CORRECTNESS_STATIC_PLAN_MISMATCH "
                f"candidate={expected.candidate.value} observed={observed} required={required}"
            )
        prepared.append(
            _PreparedCandidate(
                expected=expected,
                distributed=distributed,
                physical=physical,
                plan=plan,
            )
        )
    return tuple(prepared)


def _compile_path(
    prepared: _PreparedCandidate,
    host_inputs: tuple[np.ndarray, ...],
    devices: tuple[Any, ...],
    *,
    instrumented: bool,
) -> _CompiledPath:
    if instrumented:
        callable_value, mesh = prepared.plan.build_with_strict_mlp_checkpoints(
            expected_layers=1,
            checkpoint_specs=_CHECKPOINT_SPECS,
            interpret=False,
            devices=devices,
        )
    else:
        callable_value, mesh = prepared.plan.build(interpret=False, devices=devices)
    resident = place_inputs(host_inputs, prepared.plan.input_contracts, mesh)
    lowered = callable_value.lower(*resident)
    stablehlo = _canonical_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    pre_optimization_hlo = _canonical_hlo(lowered.compiler_ir(dialect="hlo").as_hlo_text())
    executable = lowered.compile()
    return _CompiledPath(
        prepared=prepared,
        executable=executable,
        mesh=mesh,
        stablehlo=stablehlo,
        pre_optimization_hlo=pre_optimization_hlo,
        compiler_hlo=_canonical_hlo(executable.as_text()),
    )


def _qualify_uninstrumented(
    compiled: _CompiledPath,
) -> SeqaxSiluFusionCompilerCandidate:
    prepared = compiled.prepared
    expected = prepared.expected
    runtime_memory = compiled.executable.memory_analysis()
    analysis = capture_compiler_analysis(
        _CompilerAnalysisExecutable(compiled.executable, runtime_memory),
        stablehlo=compiled.stablehlo.rstrip("\n"),
        compiler_hlo=compiled.compiler_hlo.rstrip("\n"),
    )
    reachable = analyze_compiler_collectives(
        stablehlo=compiled.stablehlo,
        compiler_hlo=live_seqax_silu_fusion_compiler_hlo(compiled.compiler_hlo),
    )
    validate_seqax_silu_fusion_compiler_collectives(
        expected,
        analysis.collectives,
        reachable,
    )
    fusion = analyze_seqax_silu_fusion_compiler_hlo(
        compiled.compiler_hlo,
        expected.candidate,
        expected_schedule_sha256=expected.physical_schedule_sha256,
    )
    resources = analyze_physical_kernel(
        prepared.physical,
        hardware=tpu7x_tensorcore_rates(),
    )
    memory = analysis.memory
    return SeqaxSiluFusionCompilerCandidate(
        candidate=expected.candidate,
        distributed_schedule_sha256=prepared.plan.distributed_schedule_sha256,
        physical_schedule_sha256=prepared.plan.physical_schedule_sha256,
        pallas_source_sha256=prepared.plan.source_sha256(),
        pallas_manifest_sha256=json_sha256(prepared.plan.manifest()),
        pre_optimization_hlo_sha256=hashlib.sha256(
            compiled.pre_optimization_hlo.encode()
        ).hexdigest(),
        compiler_analysis=analysis,
        reachable_collectives=reachable,
        fusion_analysis=fusion,
        buffer_assignment_size_bytes=memory.buffer_assignment_size_bytes,
        buffer_assignment_sha256=memory.buffer_assignment_sha256,
        allocated_vmem_bytes_per_device=resources.memory.allocated_vmem_bytes_per_device,
        peak_live_vmem_bytes_per_device=resources.memory.peak_live_vmem_bytes_per_device,
        ring_equivalent_ici_bytes_per_device=exact_integral_ring_equivalent_bytes(
            resources.devices[0].collective_ring_equivalent_bytes
        ),
    )


def _write_plan_artifacts(
    root: Path,
    prepared: _PreparedCandidate,
    uninstrumented: _CompiledPath,
    instrumented: _CompiledPath,
    compiler_candidate: SeqaxSiluFusionCompilerCandidate,
) -> None:
    plan_root = root / "plans" / prepared.expected.candidate.value
    _write_bytes_exclusive(
        plan_root / "distributed.xdsl",
        canonical_text(prepared.distributed).encode(),
    )
    _write_bytes_exclusive(
        plan_root / "physical.xdsl",
        canonical_text(prepared.physical).encode(),
    )
    _write_bytes_exclusive(
        plan_root / "lowered_pallas.py",
        prepared.plan.render_executable_source().encode(),
    )
    _write_json_exclusive(plan_root / "plan_manifest.json", prepared.plan.manifest())
    _write_bytes_exclusive(
        plan_root / "uninstrumented_stablehlo.txt",
        uninstrumented.stablehlo.encode(),
    )
    _write_bytes_exclusive(
        plan_root / "uninstrumented_pre_optimization_hlo.txt",
        uninstrumented.pre_optimization_hlo.encode(),
    )
    _write_bytes_exclusive(
        plan_root / "uninstrumented_compiler_hlo.txt",
        uninstrumented.compiler_hlo.encode(),
    )
    _write_bytes_exclusive(
        plan_root / "instrumented_stablehlo.txt",
        instrumented.stablehlo.encode(),
    )
    _write_bytes_exclusive(
        plan_root / "instrumented_pre_optimization_hlo.txt",
        instrumented.pre_optimization_hlo.encode(),
    )
    _write_bytes_exclusive(
        plan_root / "instrumented_compiler_hlo.txt",
        instrumented.compiler_hlo.encode(),
    )
    _write_json_exclusive(
        plan_root / "compiler_candidate.json",
        compiler_candidate.model_dump(mode="json", exclude_computed_fields=True),
    )


def _compile_candidates(
    root: Path,
    prepared: tuple[_PreparedCandidate, ...],
    semantic_ids: tuple[str, str],
    host_inputs: tuple[np.ndarray, ...],
    devices: tuple[Any, ...],
) -> tuple[_CompiledCandidate, ...]:
    results = []
    for index, value in enumerate(prepared):
        uninstrumented = _compile_path(
            value,
            host_inputs,
            devices,
            instrumented=False,
        )
        instrumented = _compile_path(
            value,
            host_inputs,
            devices,
            instrumented=True,
        )
        compiler_candidate = _qualify_uninstrumented(uninstrumented)
        if compiler_candidate.semantic_id != semantic_ids[index]:
            raise ValueError(
                "SEQAX_SILU_FUSION_CORRECTNESS_COMPILER_SEMANTIC_MISMATCH "
                f"candidate={value.expected.candidate.value} "
                f"observed={compiler_candidate.semantic_id} required={semantic_ids[index]}"
            )
        _write_plan_artifacts(
            root,
            value,
            uninstrumented,
            instrumented,
            compiler_candidate,
        )
        record = SeqaxSiluFusionCorrectnessPlan(
            candidate=value.expected.candidate,
            candidate_semantic_id=compiler_candidate.semantic_id,
            distributed_schedule_sha256=value.plan.distributed_schedule_sha256,
            physical_schedule_sha256=value.plan.physical_schedule_sha256,
            pallas_source_sha256=value.plan.source_sha256(),
            pallas_manifest_sha256=json_sha256(value.plan.manifest()),
            uninstrumented_stablehlo_sha256=hashlib.sha256(
                uninstrumented.stablehlo.encode()
            ).hexdigest(),
            uninstrumented_pre_optimization_hlo_sha256=hashlib.sha256(
                uninstrumented.pre_optimization_hlo.encode()
            ).hexdigest(),
            uninstrumented_compiler_hlo_sha256=hashlib.sha256(
                uninstrumented.compiler_hlo.encode()
            ).hexdigest(),
            instrumented_stablehlo_sha256=hashlib.sha256(
                instrumented.stablehlo.encode()
            ).hexdigest(),
            instrumented_pre_optimization_hlo_sha256=hashlib.sha256(
                instrumented.pre_optimization_hlo.encode()
            ).hexdigest(),
            instrumented_compiler_hlo_sha256=hashlib.sha256(
                instrumented.compiler_hlo.encode()
            ).hexdigest(),
        )
        _write_json_exclusive(
            root / "plans" / value.expected.candidate.value / "record.json",
            record.model_dump(mode="json"),
        )
        results.append(
            _CompiledCandidate(
                prepared=value,
                uninstrumented=uninstrumented,
                instrumented=instrumented,
                record=record,
            )
        )
    return tuple(results)


def _numerical_policy(
    request: SeqaxSiluFusionCorrectnessWorkerRequest,
) -> SeqaxBf16NumericalPolicy:
    policy = request.contract.policy
    return SeqaxBf16NumericalPolicy(
        cpu_relative_l2_units=policy.cpu_relative_l2_units,
        cpu_row_scaled_max_units=policy.cpu_row_scaled_max_units,
        cross_path_relative_l2_units=policy.cross_path_relative_l2_units,
        cross_path_row_scaled_max_units=policy.cross_path_row_scaled_max_units,
        row_scale_floor=policy.row_scale_floor,
        metric_quantization_decimals=policy.metric_quantization_decimals,
        mathematical_silu_max_ulp=policy.mathematical_silu_max_ulp,
        rms_inverse_relative_error_units=policy.rms_inverse_relative_error_units,
    )


def _scenario_parameters(
    request: SeqaxSiluFusionCorrectnessWorkerRequest,
) -> SeqaxBf16ScenarioParameters:
    values = dict(request.contract.parameters)
    values.pop("numerical_semantics")
    return SeqaxBf16ScenarioParameters.model_validate(values)


def _execute_outputs(
    executable: Any,
    host_inputs: tuple[np.ndarray, ...],
    plan: SeqaxPallasPlan,
    mesh: Any,
) -> tuple[np.ndarray, ...]:
    resident = place_inputs(host_inputs, plan.input_contracts, mesh)
    outputs = executable(*resident)
    jax.block_until_ready(outputs)
    return tuple(np.asarray(value) for value in outputs)


def _split_instrumented_outputs(
    outputs: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, tuple[tuple[np.ndarray, ...], ...]]:
    if len(outputs) != 14:
        raise ValueError(
            "SEQAX_SILU_FUSION_CORRECTNESS_INSTRUMENTED_OUTPUT_COUNT_MISMATCH "
            f"observed={len(outputs)} required=14"
        )
    return outputs[0], tuple((value,) for value in outputs[1:])


def _checkpoint_contracts(checkpoint_contract: Any) -> tuple[Any, ...]:
    return (
        checkpoint_contract.rms_input_checkpoints,
        checkpoint_contract.rms_mean_square_checkpoints,
        checkpoint_contract.rms_inverse_checkpoints,
        checkpoint_contract.normalized_float32_checkpoints,
        checkpoint_contract.normalized_input_checkpoints,
        checkpoint_contract.gate_float32_checkpoints,
        checkpoint_contract.gate_checkpoints,
        checkpoint_contract.silu_checkpoints,
        checkpoint_contract.up_float32_checkpoints,
        checkpoint_contract.up_checkpoints,
        checkpoint_contract.hidden_checkpoints,
        checkpoint_contract.down_float32_checkpoints,
        checkpoint_contract.down_bfloat16_checkpoints,
    )


def _stored_checkpoint(
    value: np.ndarray,
    contract: Any,
) -> np.ndarray:
    if contract.dtype == "bfloat16":
        return encode_seqax_bf16_checkpoint(value, contract)
    array = np.asarray(value)
    if contract.dtype != "float32" or array.dtype != np.float32 or array.shape != contract.shape:
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_FLOAT32_CHECKPOINT_ABI_MISMATCH")
    if not np.all(np.isfinite(array)):
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_CHECKPOINT_NONFINITE")
    return array


def _save_candidate_checkpoints(
    root: Path,
    checkpoints: tuple[tuple[np.ndarray, ...], ...],
    checkpoint_contract: Any,
    names: tuple[str, ...],
) -> tuple[str, ...]:
    hashes = []
    contracts = _checkpoint_contracts(checkpoint_contract)
    for name, values, tensor_contracts in zip(
        names,
        checkpoints,
        contracts,
        strict=True,
    ):
        if len(values) != 1 or len(tensor_contracts) != 1:
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_CHECKPOINT_LAYER_MISMATCH")
        stored = _stored_checkpoint(values[0], tensor_contracts[0])
        hashes.append(_save_array(root / f"{name}.npy", stored))
    return tuple(hashes)


def _final_output_metrics(assessment: Any) -> SeqaxSiluFusionFinalOutputMetrics:
    return SeqaxSiluFusionFinalOutputMetrics(
        cpu_relative_l2=assessment.cpu_pallas_relative_l2,
        cpu_row_scaled_max=assessment.cpu_pallas_row_scaled_max,
        cpu_top1_match=assessment.pallas_top1_matches_cpu,
        final_output_policy_passed=assessment.final_outputs_satisfy_policy,
    )


def _checkpoint_metrics(
    root: Path,
    assessment: Any,
    checkpoints: tuple[tuple[np.ndarray, ...], ...],
) -> SeqaxSiluFusionCheckpointMetrics:
    assessment_path = root / "checkpoint_assessment.json"
    _write_json_exclusive(assessment_path, assessment.model_dump(mode="json"))
    conversions_match = all(
        (
            assessment.pallas_normalized_bfloat16_matches_float32,
            assessment.pallas_gate_bfloat16_matches_float32,
            assessment.pallas_up_bfloat16_matches_float32,
            assessment.pallas_down_bfloat16_matches_float32,
        )
    )
    return SeqaxSiluFusionCheckpointMetrics(
        rms_mean_square_max_bound_ratio=assessment.pallas_rms_mean_square_max_bound_ratio,
        rms_inverse_relative_error_units=(assessment.pallas_rms_inverse_relative_error_units),
        normalized_float32_max_bound_ratio=(assessment.pallas_normalized_float32_max_bound_ratio),
        gate_float32_max_bound_ratio=assessment.pallas_gate_float32_max_bound_ratio,
        silu_max_ulp_of_mathematical=_bf16_max_ulp_distance(
            checkpoints[7][0],
            rounded_mathematical_silu_bf16(checkpoints[6][0]),
        ),
        up_float32_max_bound_ratio=assessment.pallas_up_float32_max_bound_ratio,
        hidden_matches_product=assessment.pallas_hidden_matches_product,
        down_float32_max_bound_ratio=assessment.pallas_down_float32_max_bound_ratio,
        bfloat16_conversions_match=conversions_match,
        checkpoint_values_consistent=assessment.checkpoint_values_consistent,
        full_assessment_sha256=hashlib.sha256(assessment_path.read_bytes()).hexdigest(),
    )


def _bf16_max_ulp_distance(actual: np.ndarray, expected: np.ndarray) -> int:
    def ordered(value: np.ndarray) -> np.ndarray:
        bits = np.asarray(value).view(np.uint16)
        return np.where(
            bits & np.uint16(0x8000),
            np.bitwise_not(bits),
            bits | np.uint16(0x8000),
        ).astype(np.int32)

    return int(np.max(np.abs(ordered(actual) - ordered(expected))))


def _silu_float32(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value, dtype=np.float32)
    sigmoid = np.empty_like(source)
    nonnegative = source >= 0
    sigmoid[nonnegative] = np.float32(1.0) / (np.float32(1.0) + np.exp(-source[nonnegative]))
    exponential = np.exp(source[~nonnegative])
    sigmoid[~nonnegative] = exponential / (np.float32(1.0) + exponential)
    return source * sigmoid


def _boundary_discriminator(
    root: Path,
    candidates: tuple[tuple[tuple[np.ndarray, ...], ...], ...],
) -> int:
    separate, fused = candidates
    gate = separate[6][0]
    up = separate[9][0]
    strict_silu = separate[7][0]
    strict = np.asarray(
        strict_silu.astype(np.float32) * up.astype(np.float32),
        dtype=ml_dtypes.bfloat16,
    )
    mutant = np.asarray(
        _silu_float32(gate) * up.astype(np.float32),
        dtype=ml_dtypes.bfloat16,
    )
    difference_count = int(np.count_nonzero(strict != mutant))
    if (
        not np.array_equal(fused[7][0], strict_silu)
        or not np.array_equal(separate[10][0], strict)
        or not np.array_equal(fused[10][0], strict)
        or difference_count <= 0
        or np.array_equal(separate[10][0], mutant)
        or np.array_equal(fused[10][0], mutant)
    ):
        raise ValueError(
            "SEQAX_SILU_FUSION_CORRECTNESS_BOUNDARY_DISCRIMINATOR_FAILED "
            f"difference_count={difference_count}"
        )
    _save_array(root / "strict_hidden.npy", strict.view(np.uint16))
    _save_array(root / "mutant_hidden.npy", mutant.view(np.uint16))
    return difference_count


def _run_seed(
    root: Path,
    request: SeqaxSiluFusionCorrectnessWorkerRequest,
    compiled: tuple[_CompiledCandidate, ...],
    seed: int,
    policy: SeqaxBf16NumericalPolicy,
    checkpoint_contract: Any,
) -> SeqaxSiluFusionCorrectnessObservation:
    parameters = checkpoint_contract.parameters.model_dump()
    host_inputs = tuple(
        np.asarray(value) for value in seqax_forward_inputs(seed=seed, **parameters)
    )
    seed_root = root / "seeds" / f"seed-{seed}"
    input_hashes = tuple(
        _save_array(seed_root / "inputs" / f"{index:02d}.npy", value)
        for index, value in enumerate(host_inputs)
    )
    cpu_reference = np.asarray(
        seqax_forward_canonical_reference(
            host_inputs,
            quantization_decimals=policy.cpu_reference_quantization_decimals,
            **parameters,
        )
    )
    cpu_reference_sha256 = _save_array(seed_root / "cpu_reference.npy", cpu_reference)

    records = []
    logical_checkpoints = []
    outputs = []
    instrumented_outputs = []
    for value in compiled:
        candidate = value.prepared.expected.candidate
        candidate_root = seed_root / candidate.value
        uninstrumented_values = _execute_outputs(
            value.uninstrumented.executable,
            host_inputs,
            value.prepared.plan,
            value.uninstrumented.mesh,
        )
        if len(uninstrumented_values) != 1:
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_OUTPUT_COUNT_MISMATCH")
        uninstrumented_output = uninstrumented_values[0]
        instrumented_values = _execute_outputs(
            value.instrumented.executable,
            host_inputs,
            value.prepared.plan,
            value.instrumented.mesh,
        )
        instrumented_output, checkpoints = _split_instrumented_outputs(instrumented_values)
        uninstrumented_hash = _save_array(
            candidate_root / "uninstrumented_output.npy",
            uninstrumented_output,
        )
        instrumented_hash = _save_array(
            candidate_root / "instrumented_output.npy",
            instrumented_output,
        )
        checkpoint_hashes = _save_candidate_checkpoints(
            candidate_root / "checkpoints",
            checkpoints,
            checkpoint_contract,
            request.contract.checkpoint_names,
        )
        uninstrumented_assessment = assess_seqax_bf16_final_outputs(
            uninstrumented_output,
            uninstrumented_output,
            cpu_reference,
            policy=policy,
            expected_shape=request.contract.output_shape,
            layers=1,
        )
        instrumented_assessment = assess_seqax_bf16_final_outputs(
            instrumented_output,
            instrumented_output,
            cpu_reference,
            policy=policy,
            expected_shape=request.contract.output_shape,
            layers=1,
        )
        checkpoint_assessment = assess_seqax_bf16_candidate_checkpoints(
            instrumented_output,
            seed=seed,
            inputs=host_inputs,
            checkpoints=checkpoints,
            policy=policy,
            contract=checkpoint_contract,
            declared_seeds=(*request.contract.correctness_seeds, request.contract.boundary_seed),
        )
        _write_json_exclusive(
            candidate_root / "uninstrumented_assessment.json",
            uninstrumented_assessment.model_dump(mode="json"),
        )
        _write_json_exclusive(
            candidate_root / "instrumented_assessment.json",
            instrumented_assessment.model_dump(mode="json"),
        )
        record = SeqaxSiluFusionCandidateCorrectness(
            candidate=candidate,
            uninstrumented_output_sha256=uninstrumented_hash,
            instrumented_output_sha256=instrumented_hash,
            checkpoint_sha256=checkpoint_hashes,
            checkpoint_capture_modes=request.contract.checkpoint_capture_modes[candidate.value],
            uninstrumented_metrics=_final_output_metrics(uninstrumented_assessment),
            instrumented_metrics=_final_output_metrics(instrumented_assessment),
            checkpoint_metrics=_checkpoint_metrics(
                candidate_root,
                checkpoint_assessment,
                checkpoints,
            ),
            instrumentation_output_exact=bool(
                np.array_equal(uninstrumented_output, instrumented_output)
            ),
        )
        records.append(record)
        logical_checkpoints.append(checkpoints)
        outputs.append(uninstrumented_output)
        instrumented_outputs.append(instrumented_output)

    candidate_outputs_exact = bool(np.array_equal(outputs[0], outputs[1]))
    candidate_instrumented_outputs_exact = bool(
        np.array_equal(instrumented_outputs[0], instrumented_outputs[1])
    )
    candidate_checkpoints_exact = all(
        np.array_equal(separate, fused)
        for separate_group, fused_group in zip(
            logical_checkpoints[0],
            logical_checkpoints[1],
            strict=True,
        )
        for separate, fused in zip(separate_group, fused_group, strict=True)
    )
    if not (
        candidate_outputs_exact
        and candidate_instrumented_outputs_exact
        and candidate_checkpoints_exact
    ):
        raise ValueError(
            "SEQAX_SILU_FUSION_CORRECTNESS_CANDIDATE_PARITY_MISMATCH "
            f"seed={seed} outputs={candidate_outputs_exact} "
            f"instrumented={candidate_instrumented_outputs_exact} "
            f"checkpoints={candidate_checkpoints_exact}"
        )
    boundary_case = seed == request.contract.boundary_seed
    boundary_difference_count = (
        _boundary_discriminator(
            seed_root / "boundary",
            tuple(logical_checkpoints),
        )
        if boundary_case
        else 0
    )
    observation = SeqaxSiluFusionCorrectnessObservation(
        seed=seed,
        input_sha256=input_hashes,
        cpu_reference_sha256=cpu_reference_sha256,
        candidates=tuple(records),
        candidate_uninstrumented_outputs_exact=candidate_outputs_exact,
        candidate_instrumented_outputs_exact=candidate_instrumented_outputs_exact,
        candidate_checkpoints_exact=candidate_checkpoints_exact,
        boundary_case=boundary_case,
        boundary_strict_mutant_difference_count=boundary_difference_count,
        boundary_mutant_rejected=boundary_case,
    )
    _write_json_exclusive(
        seed_root / "observation.json",
        observation.model_dump(mode="json"),
    )
    return observation


def run_worker(
    root: Path,
    request: SeqaxSiluFusionCorrectnessWorkerRequest,
) -> SeqaxSiluFusionCorrectnessWorkerResult:
    run = EvidenceRun(root / "ledger.sqlite", request.claim.claim_id)
    worker_environment, compiler_environment = _environment(root, request)
    runtime = _runtime_identity()
    host = _host_identity()
    devices = _devices()
    if runtime != request.contract.runtime or runtime != request.source.runtime:
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_RUNTIME_MISMATCH")
    if tuple(value.id for value in devices) != tuple(range(8)):
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_DEVICE_INVENTORY_MISMATCH")
    if jax.default_backend() != "tpu" or len(devices) != 8:
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_BACKEND_MISMATCH")
    run.transition(
        RunState.VERIFIED,
        {
            "runtime": runtime.model_dump(mode="json"),
            "host": host.model_dump(mode="json"),
            "devices": [value.model_dump(mode="json") for value in devices],
        },
    )

    design, _pair = _load_bound_compiler_evidence(root, request)
    prepared = _prepare_candidates(request, design)
    run.transition(
        RunState.LOWERED,
        {"physical_schedules": [value.expected.physical_schedule_sha256 for value in prepared]},
    )
    first_seed = request.contract.correctness_seeds[0]
    parameters = _scenario_parameters(request)
    first_inputs = tuple(
        np.asarray(value)
        for value in seqax_forward_inputs(seed=first_seed, **parameters.model_dump())
    )
    compiled = _compile_candidates(
        root,
        prepared,
        request.contract.candidate_semantic_ids,
        first_inputs,
        tuple(jax.devices()),
    )
    run.transition(
        RunState.COMPILED,
        {"candidate_semantic_ids": [value.record.candidate_semantic_id for value in compiled]},
    )
    policy = _numerical_policy(request)
    checkpoint_contract = seqax_bf16_checkpoint_contract(parameters)
    seeds = (*request.contract.correctness_seeds, request.contract.boundary_seed)
    observations = tuple(
        _run_seed(
            root,
            request,
            compiled,
            seed,
            policy,
            checkpoint_contract,
        )
        for seed in seeds
    )
    result = SeqaxSiluFusionCorrectnessResult(
        contract_id=request.contract.contract_id,
        claim_id=request.claim.claim_id,
        source=request.source,
        host=host,
        devices=devices,
        plans=tuple(value.record for value in compiled),
        observations=observations,
        model_outputs_executed=True,
        full_inputs_persisted=True,
        full_outputs_persisted=True,
        full_checkpoints_persisted=True,
        producer_passed=True,
        timing_collected=False,
        profile_collected=False,
        worker_pid=os.getpid(),
        worker_nonce=secrets.token_hex(16),
        source_import_root=str(root / "source" / "committed" / "src"),
        worker_environment=worker_environment,
        compiler_environment=compiler_environment,
    )
    worker_result = SeqaxSiluFusionCorrectnessWorkerResult(result=result)
    _write_json_exclusive(
        root / "worker-result.json",
        worker_result.model_dump(mode="json", exclude_computed_fields=True),
    )
    run.transition(
        RunState.CORRECT,
        {
            "result_id": result.result_id,
            "observation_count": len(observations),
        },
    )
    return worker_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    arguments = parser.parse_args()
    request = SeqaxSiluFusionCorrectnessWorkerRequest.model_validate_json(
        arguments.request.read_text()
    )
    run_worker(arguments.root, request)


if __name__ == "__main__":
    main()
