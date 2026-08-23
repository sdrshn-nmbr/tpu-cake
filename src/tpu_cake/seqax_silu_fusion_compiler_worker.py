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
import jax.numpy as jnp
from jax.sharding import NamedSharding

from tpu_cake.canonical import canonical_text
from tpu_cake.compiler_analysis import (
    analyze_compiler_collectives,
    capture_compiler_analysis,
)
from tpu_cake.cost_model import tpu7x_tensorcore_rates
from tpu_cake.dialects.tpu_schedule import CollectiveOp, VectorComputeOp
from tpu_cake.identity import json_sha256
from tpu_cake.ledger import EvidenceRun, RunState
from tpu_cake.physical_cost_model import analyze_physical_kernel
from tpu_cake.runner import _runtime_identity
from tpu_cake.seqax_pallas_lowering import SeqaxPallasPlan, lower_seqax_physical_to_pallas
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.seqax_silu_fusion import SeqaxSiluFusionPlanContract
from tpu_cake.seqax_silu_fusion_compiler import (
    SeqaxSiluFusionCompilerCandidate,
    SeqaxSiluFusionCompilerCapture,
    SeqaxSiluFusionCompilerDevice,
    SeqaxSiluFusionCompilerHostIdentity,
    SeqaxSiluFusionCompilerWorkerRequest,
    SeqaxSiluFusionCompilerWorkerResult,
    analyze_seqax_silu_fusion_compiler_hlo,
    live_seqax_silu_fusion_compiler_hlo,
)
from tpu_cake.stablehlo import StableHloInspector
from tpu_cake.workloads.seqax_forward import (
    SeqaxFeedForwardVectorExecution,
    SeqaxNumericalSemantics,
    seqax_forward_schedule,
)

_COMPILER_ENVIRONMENT_PREFIXES = ("JAX_", "XLA_", "PJRT_", "LIBTPU_")


@dataclass(frozen=True)
class _PreparedCandidate:
    expected: SeqaxSiluFusionPlanContract
    distributed: Any
    physical: Any
    plan: SeqaxPallasPlan


class _RejectMetadataRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl) -> None:
        raise ValueError(f"SEQAX_SILU_FUSION_METADATA_REDIRECT code={code} url={newurl}")


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
            raise ValueError("SEQAX_SILU_FUSION_METADATA_HEADER_MISSING")
        payload = response.read(4097)
    if len(payload) > 4096:
        raise ValueError("SEQAX_SILU_FUSION_METADATA_RESPONSE_TOO_LARGE")
    return payload.decode().strip()


def _host_identity() -> SeqaxSiluFusionCompilerHostIdentity:
    zone_resource = _metadata("instance/zone")
    machine_type_resource = _metadata("instance/machine-type")
    return SeqaxSiluFusionCompilerHostIdentity(
        project=_metadata("project/project-id"),
        numeric_project_id=_metadata("project/numeric-project-id"),
        zone=zone_resource.rsplit("/", maxsplit=1)[-1],
        hostname=_metadata("instance/name"),
        instance_hostname=_metadata("instance/hostname"),
        machine_type=machine_type_resource.rsplit("/", maxsplit=1)[-1],
        instance_id=_metadata("instance/id"),
        cpu_platform=_metadata("instance/cpu-platform"),
    )


def _expected_host(
    request: SeqaxSiluFusionCompilerWorkerRequest,
) -> SeqaxSiluFusionCompilerHostIdentity:
    design = request.design
    return SeqaxSiluFusionCompilerHostIdentity(
        project=design.project,
        numeric_project_id=design.numeric_project_id,
        zone=design.zone,
        hostname=design.hostname,
        instance_hostname=design.instance_hostname,
        machine_type=design.machine_type,
        instance_id=design.instance_id,
        cpu_platform=design.cpu_platform,
    )


def _compiler_environment(
    root: Path,
    request: SeqaxSiluFusionCompilerWorkerRequest,
) -> tuple[dict[str, str], dict[str, str]]:
    expected_compiler = request.design.compiler_environment
    expected_worker = request.design.worker_environment
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
            "SEQAX_SILU_FUSION_COMPILER_ENVIRONMENT_MISMATCH "
            f"observed={observed} forbidden={forbidden}"
        )
    return dict(expected_compiler), dict(expected_worker)


def _devices() -> tuple[SeqaxSiluFusionCompilerDevice, ...]:
    return tuple(
        SeqaxSiluFusionCompilerDevice(
            id=int(device.id),
            process_index=int(device.process_index),
            platform=str(device.platform),
            device_kind=str(device.device_kind),
        )
        for device in jax.devices()
    )


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


def _canonical_hlo(value: str) -> str:
    return value.rstrip("\n") + "\n"


def _prepare(request: SeqaxSiluFusionCompilerWorkerRequest) -> tuple[_PreparedCandidate, ...]:
    design = request.design
    parameters = dict(design.parameters)
    parameters["numerical_semantics"] = SeqaxNumericalSemantics(parameters["numerical_semantics"])
    prepared = []
    for expected in design.candidates:
        distributed = seqax_forward_schedule(
            **parameters,
            feed_forward_fusion=expected.candidate,
            feed_forward_vector_execution=SeqaxFeedForwardVectorExecution.PALLAS_FULL_LOCAL,
            residual_norm_strategy=design.residual_norm_strategy,
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
            resources.devices[0].collective_ring_equivalent_bytes,
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
                "SEQAX_SILU_FUSION_STATIC_PLAN_MISMATCH "
                f"candidate={expected.candidate} expected={required} observed={observed}"
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


def _abstract_inputs(prepared: _PreparedCandidate, mesh: Any) -> tuple[Any, ...]:
    dtypes = {
        "bfloat16": jnp.bfloat16,
        "bool": jnp.bool_,
        "float32": jnp.float32,
        "int32": jnp.int32,
        "uint32": jnp.uint32,
    }
    values = []
    for contract in prepared.plan.input_contracts:
        dtype = dtypes.get(contract.dtype)
        if dtype is None:
            raise ValueError(f"SEQAX_SILU_FUSION_ABSTRACT_DTYPE_UNSUPPORTED dtype={contract.dtype}")
        values.append(
            jax.ShapeDtypeStruct(
                tuple(size for _name, size in contract.shape),
                dtype,
                sharding=NamedSharding(mesh, contract.partition_spec()),
            )
        )
    return tuple(values)


def _validate_stablehlo(prepared: _PreparedCandidate, stablehlo: str) -> None:
    expected = prepared.expected
    counts = StableHloInspector.parse(stablehlo).live_collective_counts()
    observed_collectives = tuple(
        counts[name] for name in ("all_gather", "all_reduce", "reduce_scatter")
    )
    required_collectives = (
        expected.expected_all_gathers,
        expected.expected_all_reduces,
        expected.expected_reduce_scatters,
    )
    if observed_collectives != required_collectives:
        raise ValueError("SEQAX_SILU_FUSION_STABLEHLO_COLLECTIVE_MISMATCH")
    strict_kernels = (
        "seqax_strict_bf16_silu",
        "seqax_strict_bf16_multiply",
        "seqax_strict_bf16_silu_multiply",
    )
    kernel_counts = {value: stablehlo.count(f'kernel_name = "{value}"') for value in strict_kernels}
    required_kernel_counts = {
        value: int(value in expected.expected_strict_vector_kernels) for value in strict_kernels
    }
    if kernel_counts != required_kernel_counts:
        raise ValueError("SEQAX_SILU_FUSION_STABLEHLO_VECTOR_KERNEL_MISMATCH")
    if stablehlo.count('kernel_name = "seqax_named_einsum"') != expected.expected_pallas_regions:
        raise ValueError("SEQAX_SILU_FUSION_STABLEHLO_EINSUM_COUNT_MISMATCH")


def _compile(
    root: Path,
    prepared: _PreparedCandidate,
    devices: tuple[Any, ...],
) -> SeqaxSiluFusionCompilerCandidate:
    expected = prepared.expected
    candidate_root = root / "candidates" / expected.candidate.value
    callable_value, mesh = prepared.plan.build(interpret=False, devices=devices)
    lowered = callable_value.lower(*_abstract_inputs(prepared, mesh))
    stablehlo = _canonical_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    pre_optimization_hlo = _canonical_hlo(lowered.compiler_ir(dialect="hlo").as_hlo_text())
    _validate_stablehlo(prepared, stablehlo)
    executable = lowered.compile()
    compiler_hlo = _canonical_hlo(executable.as_text())
    compiler_analysis = capture_compiler_analysis(
        executable,
        stablehlo=stablehlo.rstrip("\n"),
        compiler_hlo=compiler_hlo.rstrip("\n"),
    )
    reachable_collectives = analyze_compiler_collectives(
        stablehlo=stablehlo,
        compiler_hlo=live_seqax_silu_fusion_compiler_hlo(compiler_hlo),
    )
    required_collectives = (
        expected.expected_all_gathers,
        expected.expected_all_reduces,
        expected.expected_reduce_scatters,
        expected.expected_all_gathers,
        expected.expected_reduce_scatters,
    )
    observed_collectives = (
        reachable_collectives.compiler_all_gather_count,
        reachable_collectives.compiler_all_reduce_count,
        reachable_collectives.compiler_reduce_scatter_count,
        reachable_collectives.sparse_core_all_gather_count,
        reachable_collectives.sparse_core_reduce_scatter_count,
    )
    if observed_collectives != required_collectives:
        raise ValueError("SEQAX_SILU_FUSION_COMPILER_COLLECTIVE_MISMATCH")
    fusion_analysis = analyze_seqax_silu_fusion_compiler_hlo(
        compiler_hlo,
        expected.candidate,
        expected_schedule_sha256=expected.physical_schedule_sha256,
    )
    memory = executable.memory_analysis()
    buffer_assignment = memory.serialized_buffer_assignment_proto
    if not isinstance(buffer_assignment, bytes) or not buffer_assignment:
        raise ValueError("SEQAX_SILU_FUSION_BUFFER_ASSIGNMENT_UNAVAILABLE")
    resources = analyze_physical_kernel(
        prepared.physical,
        hardware=tpu7x_tensorcore_rates(),
    )
    _write_bytes_exclusive(
        candidate_root / "distributed.xdsl",
        canonical_text(prepared.distributed).encode(),
    )
    _write_bytes_exclusive(
        candidate_root / "physical.xdsl",
        canonical_text(prepared.physical).encode(),
    )
    _write_bytes_exclusive(
        candidate_root / "lowered_pallas.py",
        prepared.plan.render_executable_source().encode(),
    )
    _write_json_exclusive(candidate_root / "plan_manifest.json", prepared.plan.manifest())
    _write_json_exclusive(
        candidate_root / "physical_resources.json",
        resources.model_dump(mode="json"),
    )
    _write_bytes_exclusive(candidate_root / "stablehlo.txt", stablehlo.encode())
    _write_bytes_exclusive(
        candidate_root / "pre_optimization_hlo.txt",
        pre_optimization_hlo.encode(),
    )
    _write_bytes_exclusive(candidate_root / "compiler_hlo.txt", compiler_hlo.encode())
    _write_json_exclusive(
        candidate_root / "compiler_analysis.json",
        compiler_analysis.model_dump(mode="json"),
    )
    _write_bytes_exclusive(candidate_root / "buffer_assignment.pb", buffer_assignment)
    _write_json_exclusive(
        candidate_root / "fusion_analysis.json",
        fusion_analysis.model_dump(mode="json", exclude_computed_fields=True),
    )
    return SeqaxSiluFusionCompilerCandidate(
        candidate=expected.candidate,
        distributed_schedule_sha256=prepared.plan.distributed_schedule_sha256,
        physical_schedule_sha256=prepared.plan.physical_schedule_sha256,
        pallas_source_sha256=prepared.plan.source_sha256(),
        pallas_manifest_sha256=json_sha256(prepared.plan.manifest()),
        pre_optimization_hlo_sha256=hashlib.sha256(pre_optimization_hlo.encode()).hexdigest(),
        compiler_analysis=compiler_analysis,
        reachable_collectives=reachable_collectives,
        fusion_analysis=fusion_analysis,
        buffer_assignment_size_bytes=len(buffer_assignment),
        buffer_assignment_sha256=hashlib.sha256(buffer_assignment).hexdigest(),
        allocated_vmem_bytes_per_device=resources.memory.allocated_vmem_bytes_per_device,
        peak_live_vmem_bytes_per_device=resources.memory.peak_live_vmem_bytes_per_device,
        ring_equivalent_ici_bytes_per_device=(
            resources.devices[0].collective_ring_equivalent_bytes
        ),
    )


def run_worker(root: Path, request: SeqaxSiluFusionCompilerWorkerRequest) -> None:
    run = EvidenceRun(root / "ledger.sqlite", request.claim.claim_id)
    compiler_environment, worker_environment = _compiler_environment(root, request)
    runtime = _runtime_identity()
    host = _host_identity()
    devices = _devices()
    if runtime != request.design.runtime or runtime != request.source.runtime:
        raise ValueError("SEQAX_SILU_FUSION_RUNTIME_MISMATCH")
    if host != _expected_host(request):
        raise ValueError("SEQAX_SILU_FUSION_HOST_MISMATCH")
    if tuple(device.id for device in devices) != tuple(range(8)):
        raise ValueError("SEQAX_SILU_FUSION_DEVICE_INVENTORY_MISMATCH")
    if jax.default_backend() != request.design.backend or len(devices) != 8:
        raise ValueError("SEQAX_SILU_FUSION_BACKEND_MISMATCH")
    run.transition(
        RunState.VERIFIED,
        {
            "runtime": runtime.model_dump(mode="json"),
            "host": host.model_dump(mode="json"),
            "devices": [device.model_dump(mode="json") for device in devices],
        },
    )
    prepared = _prepare(request)
    run.transition(
        RunState.LOWERED,
        {"plan_sha256": [value.expected.physical_schedule_sha256 for value in prepared]},
    )
    jax_devices = tuple(jax.devices())
    candidates = tuple(_compile(root, value, jax_devices) for value in prepared)
    capture = SeqaxSiluFusionCompilerCapture(
        design_id=request.design.design_id,
        capture_ordinal=request.claim.capture_ordinal,
        invocation_id=request.claim.invocation_id,
        claim_id=request.claim.claim_id,
        source=request.source,
        host=host,
        worker_environment=worker_environment,
        compiler_environment=compiler_environment,
        source_import_root=str(root / "source" / "committed" / "src"),
        compile_input_mode="abstract-only",
        devices=devices,
        worker_pid=os.getpid(),
        worker_nonce=secrets.token_hex(16),
        candidates=candidates,
        model_outputs_executed=False,
        correctness_outputs_collected=False,
        timing_collected=False,
        profile_collected=False,
    )
    result = SeqaxSiluFusionCompilerWorkerResult(capture=capture)
    _write_json_exclusive(root / "worker-result.json", result.wire_payload())
    run.transition(
        RunState.COMPILED,
        {
            "capture_id": capture.capture_id,
            "semantic_pair_id": capture.semantic_pair_id,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    arguments = parser.parse_args()
    request = SeqaxSiluFusionCompilerWorkerRequest.model_validate_json(
        arguments.request.read_text()
    )
    run_worker(arguments.root, request)


if __name__ == "__main__":
    main()
