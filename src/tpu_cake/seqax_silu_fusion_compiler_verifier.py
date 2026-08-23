from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import stat
from collections import Counter
from pathlib import Path

from tpu_cake.artifacts import file_sha256, validate_artifact_manifest
from tpu_cake.canonical import canonical_text
from tpu_cake.compiler_analysis import (
    analyze_compiler_collectives,
    validate_compiler_analysis,
)
from tpu_cake.contracts import ArtifactRole, SourceFileContract
from tpu_cake.cost_model import tpu7x_tensorcore_rates
from tpu_cake.dialects.tpu_schedule import CollectiveOp, VectorComputeOp
from tpu_cake.identity import json_sha256
from tpu_cake.ledger import RunState, payload_sha256
from tpu_cake.physical_cost_model import (
    PhysicalKernelResourceReport,
    analyze_physical_kernel,
)
from tpu_cake.seqax_pallas_lowering import lower_seqax_physical_to_pallas
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.seqax_silu_fusion import (
    SeqaxSiluFusionDesignContract,
    default_seqax_silu_fusion_design_contract,
)
from tpu_cake.seqax_silu_fusion_compiler import (
    SEQAX_SILU_FUSION_COMPILER_ARTIFACT_ROLES,
    SeqaxSiluFusionCompilerAnalysis,
    SeqaxSiluFusionCompilerAttemptClaim,
    SeqaxSiluFusionCompilerCandidate,
    SeqaxSiluFusionCompilerHostIdentity,
    SeqaxSiluFusionCompilerPair,
    SeqaxSiluFusionCompilerReceipt,
    SeqaxSiluFusionCompilerReplaySeal,
    SeqaxSiluFusionCompilerSourceAuthority,
    SeqaxSiluFusionCompilerWorkerRequest,
    SeqaxSiluFusionCompilerWorkerResult,
    analyze_seqax_silu_fusion_compiler_hlo,
    live_seqax_silu_fusion_compiler_hlo,
    seqax_silu_fusion_compiler_pair_member,
)
from tpu_cake.stablehlo import StableHloInspector
from tpu_cake.workloads.seqax_forward import (
    SeqaxFeedForwardVectorExecution,
    SeqaxNumericalSemantics,
    seqax_forward_schedule,
)

_DESIGN_RELATIVE_PATH = "contracts/seqax-silu-fusion-design-v1.json"
_SOURCE_PATHS = {
    "runner_source_sha256": "src/tpu_cake/seqax_silu_fusion_compiler_runner.py",
    "worker_source_sha256": "src/tpu_cake/seqax_silu_fusion_compiler_worker.py",
    "compiler_source_sha256": "src/tpu_cake/seqax_silu_fusion_compiler.py",
    "pair_source_sha256": "src/tpu_cake/seqax_silu_fusion_compiler_pair.py",
}


def _artifact_role(path: Path) -> ArtifactRole:
    value = path.as_posix()
    fixed = {
        "attempt_claim.json": ArtifactRole.INVOCATION,
        "contract.json": ArtifactRole.EXPERIMENT,
        "source.json": ArtifactRole.SOURCE_STATE,
        "source_manifest.json": ArtifactRole.SOURCE_STATE,
        "worker_request.json": ArtifactRole.INVOCATION,
        "worker-result.json": ArtifactRole.COMPILER_ANALYSIS,
        "ledger.sqlite": ArtifactRole.EXECUTION_LEDGER,
    }
    if value in fixed:
        return fixed[value]
    if value.startswith("source/committed/"):
        return ArtifactRole.SOURCE_STATE
    candidate_roles = {
        "distributed.xdsl": ArtifactRole.DISTRIBUTED_IR,
        "physical.xdsl": ArtifactRole.PHYSICAL_IR,
        "lowered_pallas.py": ArtifactRole.PALLAS_SOURCE,
        "plan_manifest.json": ArtifactRole.PLAN_MANIFEST,
        "physical_resources.json": ArtifactRole.COST_MODEL,
        "stablehlo.txt": ArtifactRole.STABLEHLO,
        "pre_optimization_hlo.txt": ArtifactRole.COMPILER_HLO,
        "compiler_hlo.txt": ArtifactRole.COMPILER_HLO,
        "compiler_analysis.json": ArtifactRole.COMPILER_ANALYSIS,
        "buffer_assignment.pb": ArtifactRole.COMPILER_ANALYSIS,
        "fusion_analysis.json": ArtifactRole.COMPILER_ANALYSIS,
    }
    match = re.fullmatch(r"candidates/(separate|silu_multiply)/([^/]+)", value)
    if match is not None and match.group(2) in candidate_roles:
        return candidate_roles[match.group(2)]
    raise ValueError(f"SEQAX_SILU_FUSION_ARTIFACT_ROLE_UNKNOWN path={value}")


def _preflight_root(root: Path) -> None:
    status = root.lstat()
    if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.getuid():
        raise ValueError("SEQAX_SILU_FUSION_ROOT_AUTHORITY_INVALID")
    if status.st_mode & 0o077:
        raise ValueError("SEQAX_SILU_FUSION_ROOT_PERMISSIONS_INVALID")
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("SEQAX_SILU_FUSION_ARTIFACT_SYMLINK")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("SEQAX_SILU_FUSION_ARTIFACT_NONREGULAR")
        if info.st_nlink != 1:
            raise ValueError("SEQAX_SILU_FUSION_ARTIFACT_HARDLINK")


def _registry_file(design: SeqaxSiluFusionDesignContract, name: str) -> Path:
    registry = Path(design.compiler_claim_registry_root)
    if registry.is_symlink() or not registry.is_dir():
        raise ValueError("SEQAX_SILU_FUSION_CLAIM_REGISTRY_INVALID")
    registry_info = registry.lstat()
    if (
        not stat.S_ISDIR(registry_info.st_mode)
        or registry_info.st_uid != os.getuid()
        or registry_info.st_mode & 0o077
    ):
        raise ValueError("SEQAX_SILU_FUSION_CLAIM_REGISTRY_INVALID")
    path = registry / name
    if path.is_symlink() or not path.is_file():
        raise ValueError("SEQAX_SILU_FUSION_REGISTRY_FILE_INVALID")
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_nlink != 1
        or info.st_mode & 0o077
    ):
        raise ValueError("SEQAX_SILU_FUSION_REGISTRY_FILE_INVALID")
    return path


def _validate_source_bundle(
    root: Path,
    source: SeqaxSiluFusionCompilerSourceAuthority,
) -> None:
    manifest = tuple(
        SourceFileContract.model_validate(value)
        for value in json.loads((root / "source_manifest.json").read_text())
    )
    if manifest != source.source_manifest:
        raise ValueError("SEQAX_SILU_FUSION_SOURCE_MANIFEST_MISMATCH")
    bundle = root / "source" / "committed"
    observed = tuple(
        sorted(path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file())
    )
    if observed != tuple(value.path for value in manifest):
        raise ValueError("SEQAX_SILU_FUSION_SOURCE_BUNDLE_NOT_CLOSED")
    for value in manifest:
        path = bundle / value.path
        if file_sha256(path) != value.sha256:
            raise ValueError(f"SEQAX_SILU_FUSION_SOURCE_FILE_MISMATCH path={value.path}")
    if file_sha256(bundle / "uv.lock") != source.uv_lock_sha256:
        raise ValueError("SEQAX_SILU_FUSION_UV_LOCK_MISMATCH")
    if file_sha256(bundle / "src/tpu_cake/cli.py") != source.cli_sha256:
        raise ValueError("SEQAX_SILU_FUSION_CLI_MISMATCH")
    if file_sha256(bundle / _DESIGN_RELATIVE_PATH) != source.design_file_sha256:
        raise ValueError("SEQAX_SILU_FUSION_DESIGN_FILE_MISMATCH")
    for field, path in _SOURCE_PATHS.items():
        if file_sha256(bundle / path) != getattr(source, field):
            raise ValueError(f"SEQAX_SILU_FUSION_AUTHORITY_SOURCE_MISMATCH path={path}")


def _ledger_state(
    root: Path,
    design: SeqaxSiluFusionDesignContract,
    claim: SeqaxSiluFusionCompilerAttemptClaim,
    capture,
) -> RunState:
    connection = sqlite3.connect(f"file:{root / 'ledger.sqlite'}?mode=ro&immutable=1", uri=True)
    try:
        columns = connection.execute("PRAGMA table_info(events)").fetchall()
        rows = connection.execute(
            "SELECT sequence, run_id, state, timestamp_ns, payload_sha256 "
            "FROM events ORDER BY sequence"
        ).fetchall()
    finally:
        connection.close()
    expected_columns = [
        (0, "sequence", "INTEGER", 0, None, 1),
        (1, "run_id", "TEXT", 1, None, 0),
        (2, "state", "TEXT", 1, None, 0),
        (3, "timestamp_ns", "INTEGER", 1, None, 0),
        (4, "payload_sha256", "TEXT", 1, None, 0),
    ]
    payloads = (
        {
            "claim_id": claim.claim_id,
            "claim_path": str(
                Path(design.compiler_claim_registry_root)
                / f"{design.compiler_claim_key}-{claim.capture_ordinal}.json"
            ),
            "design_id": design.design_id,
            "capture_ordinal": claim.capture_ordinal,
        },
        {
            "runtime": capture.source.runtime.model_dump(mode="json"),
            "host": capture.host.model_dump(mode="json"),
            "devices": [value.model_dump(mode="json") for value in capture.devices],
        },
        {"plan_sha256": [value.physical_schedule_sha256 for value in design.candidates]},
        {
            "capture_id": capture.capture_id,
            "semantic_pair_id": capture.semantic_pair_id,
        },
    )
    states = (
        RunState.CREATED,
        RunState.VERIFIED,
        RunState.LOWERED,
        RunState.COMPILED,
    )
    expected_rows = [
        (index, claim.claim_id, state.value, payload_sha256(payload))
        for index, (state, payload) in enumerate(zip(states, payloads, strict=True), start=1)
    ]
    observed_rows = [(row[0], row[1], row[2], row[4]) for row in rows]
    timestamps = [row[3] for row in rows]
    if (
        columns != expected_columns
        or observed_rows != expected_rows
        or any(not isinstance(value, int) or value < 0 for value in timestamps)
        or timestamps != sorted(timestamps)
    ):
        raise ValueError(f"SEQAX_SILU_FUSION_LEDGER_MISMATCH rows={rows}")
    return RunState.COMPILED


def _plans(design: SeqaxSiluFusionDesignContract):
    parameters = dict(design.parameters)
    parameters["numerical_semantics"] = SeqaxNumericalSemantics(parameters["numerical_semantics"])
    return tuple(
        (
            expected,
            distributed,
            physical,
            lower_seqax_physical_to_pallas(distributed, physical),
        )
        for expected in design.candidates
        for distributed in (
            seqax_forward_schedule(
                **parameters,
                feed_forward_fusion=expected.candidate,
                feed_forward_vector_execution=SeqaxFeedForwardVectorExecution.PALLAS_FULL_LOCAL,
                residual_norm_strategy=design.residual_norm_strategy,
            ),
        )
        for physical in (lower_seqax_forward_to_physical(distributed).module,)
    )


def _validate_stablehlo(stablehlo: str, expected) -> None:
    counts = StableHloInspector.parse(stablehlo).live_collective_counts()
    if tuple(counts[name] for name in ("all_gather", "all_reduce", "reduce_scatter")) != (
        expected.expected_all_gathers,
        expected.expected_all_reduces,
        expected.expected_reduce_scatters,
    ):
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


def _validate_candidate(
    root: Path,
    expected,
    distributed,
    physical,
    plan,
    recorded: SeqaxSiluFusionCompilerCandidate,
) -> None:
    candidate_root = root / "candidates" / expected.candidate.value
    if (candidate_root / "distributed.xdsl").read_text() != canonical_text(distributed):
        raise ValueError("SEQAX_SILU_FUSION_DISTRIBUTED_IR_MISMATCH")
    if (candidate_root / "physical.xdsl").read_text() != canonical_text(physical):
        raise ValueError("SEQAX_SILU_FUSION_PHYSICAL_IR_MISMATCH")
    if (candidate_root / "lowered_pallas.py").read_text() != plan.render_executable_source():
        raise ValueError("SEQAX_SILU_FUSION_PALLAS_SOURCE_MISMATCH")
    manifest = json.loads((candidate_root / "plan_manifest.json").read_text())
    if manifest != plan.manifest():
        raise ValueError("SEQAX_SILU_FUSION_PALLAS_MANIFEST_MISMATCH")
    resources = PhysicalKernelResourceReport.model_validate_json(
        (candidate_root / "physical_resources.json").read_text()
    )
    replayed_resources = analyze_physical_kernel(physical, hardware=tpu7x_tensorcore_rates())
    if resources != replayed_resources:
        raise ValueError("SEQAX_SILU_FUSION_RESOURCE_REPLAY_MISMATCH")
    stablehlo = (candidate_root / "stablehlo.txt").read_text()
    pre_optimization_hlo = (candidate_root / "pre_optimization_hlo.txt").read_text()
    compiler_hlo = (candidate_root / "compiler_hlo.txt").read_text()
    _validate_stablehlo(stablehlo, expected)
    compiler_analysis = validate_compiler_analysis(
        candidate_root / "compiler_analysis.json",
        stablehlo_path=candidate_root / "stablehlo.txt",
        compiler_hlo_path=candidate_root / "compiler_hlo.txt",
    )
    fusion_analysis = SeqaxSiluFusionCompilerAnalysis.model_validate_json(
        (candidate_root / "fusion_analysis.json").read_text()
    )
    replayed_fusion = analyze_seqax_silu_fusion_compiler_hlo(
        compiler_hlo,
        expected.candidate,
        expected_schedule_sha256=expected.physical_schedule_sha256,
    )
    if fusion_analysis != replayed_fusion:
        raise ValueError("SEQAX_SILU_FUSION_COMPILER_GRAPH_REPLAY_MISMATCH")
    reachable_collectives = analyze_compiler_collectives(
        stablehlo=stablehlo,
        compiler_hlo=live_seqax_silu_fusion_compiler_hlo(compiler_hlo),
    )
    buffer_assignment = (candidate_root / "buffer_assignment.pb").read_bytes()
    observed = SeqaxSiluFusionCompilerCandidate(
        candidate=expected.candidate,
        distributed_schedule_sha256=plan.distributed_schedule_sha256,
        physical_schedule_sha256=plan.physical_schedule_sha256,
        pallas_source_sha256=plan.source_sha256(),
        pallas_manifest_sha256=json_sha256(plan.manifest()),
        pre_optimization_hlo_sha256=hashlib.sha256(pre_optimization_hlo.encode()).hexdigest(),
        compiler_analysis=compiler_analysis,
        reachable_collectives=reachable_collectives,
        fusion_analysis=replayed_fusion,
        buffer_assignment_size_bytes=len(buffer_assignment),
        buffer_assignment_sha256=hashlib.sha256(buffer_assignment).hexdigest(),
        allocated_vmem_bytes_per_device=resources.memory.allocated_vmem_bytes_per_device,
        peak_live_vmem_bytes_per_device=resources.memory.peak_live_vmem_bytes_per_device,
        ring_equivalent_ici_bytes_per_device=(
            resources.devices[0].collective_ring_equivalent_bytes
        ),
    )
    if observed != recorded:
        raise ValueError("SEQAX_SILU_FUSION_CANDIDATE_REPLAY_MISMATCH")
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
    if (
        tuple(f"seqax_strict_bf16_{value}" for value in owned)
        != expected.expected_strict_vector_kernels
        or len(vectors) != expected.expected_physical_vector_operations
        or collectives
        != {
            "all_gather": expected.expected_all_gathers,
            "all_reduce": expected.expected_all_reduces,
            "reduce_scatter": expected.expected_reduce_scatters,
        }
    ):
        raise ValueError("SEQAX_SILU_FUSION_STATIC_BOUNDARY_REPLAY_MISMATCH")


def verify_capture(root: Path, design_path: Path) -> SeqaxSiluFusionCompilerReceipt:
    root = root.resolve(strict=True)
    _preflight_root(root)
    design = SeqaxSiluFusionDesignContract.model_validate_json(design_path.read_text())
    if design != default_seqax_silu_fusion_design_contract(design.runtime):
        raise ValueError("SEQAX_SILU_FUSION_DESIGN_NONCANONICAL")
    receipt = SeqaxSiluFusionCompilerReceipt.model_validate_json(
        (root / "receipt.json").read_text()
    )
    validate_artifact_manifest(
        root,
        receipt.artifacts,
        role_for_path=_artifact_role,
        duplicate_error="SEQAX_SILU_FUSION_ARTIFACT_DUPLICATE",
        closed_world_error="SEQAX_SILU_FUSION_ARTIFACT_SET_MISMATCH",
        mismatch_error=lambda path: f"SEQAX_SILU_FUSION_ARTIFACT_MISMATCH path={path}",
        symlink_error="SEQAX_SILU_FUSION_ARTIFACT_SYMLINK",
    )
    source = SeqaxSiluFusionCompilerSourceAuthority.model_validate_json(
        (root / "source.json").read_text()
    )
    _validate_source_bundle(root, source)
    if source != receipt.capture.source or source.runtime != design.runtime:
        raise ValueError("SEQAX_SILU_FUSION_SOURCE_AUTHORITY_MISMATCH")
    if file_sha256(root / "contract.json") != source.design_file_sha256:
        raise ValueError("SEQAX_SILU_FUSION_CAPTURE_CONTRACT_MISMATCH")
    claim = SeqaxSiluFusionCompilerAttemptClaim.model_validate_json(
        (root / "attempt_claim.json").read_text()
    )
    request = SeqaxSiluFusionCompilerWorkerRequest.model_validate_json(
        (root / "worker_request.json").read_text()
    )
    result = SeqaxSiluFusionCompilerWorkerResult.model_validate_json(
        (root / "worker-result.json").read_text()
    )
    capture = receipt.capture
    expected_host = SeqaxSiluFusionCompilerHostIdentity(
        project=design.project,
        numeric_project_id=design.numeric_project_id,
        zone=design.zone,
        hostname=design.hostname,
        instance_hostname=design.instance_hostname,
        machine_type=design.machine_type,
        instance_id=design.instance_id,
        cpu_platform=design.cpu_platform,
    )
    if (
        request.claim != claim
        or request.design != design
        or request.source != source
        or result.capture != capture
        or capture.design_id != design.design_id
        or capture.capture_ordinal != claim.capture_ordinal
        or capture.invocation_id != claim.invocation_id
        or claim.claim_id != capture.claim_id
        or claim.output_root != str(root)
        or claim.design_id != design.design_id
        or claim.source_commit != source.source_commit
        or claim.source_tree != source.source_tree
        or source.remote_url != design.source_remote_url
        or source.source_root != design.compilation_source_root
        or source.runtime != design.runtime
        or capture.host != expected_host
        or capture.worker_environment != design.worker_environment
        or capture.compiler_environment != design.compiler_environment
        or capture.source_import_root != str(root / "source" / "committed" / "src")
        or capture.compile_input_mode != design.compile_input_mode
    ):
        raise ValueError("SEQAX_SILU_FUSION_CAPTURE_LINKAGE_MISMATCH")
    external_claim_path = _registry_file(
        design,
        f"{design.compiler_claim_key}-{claim.capture_ordinal}.json",
    )
    if (
        SeqaxSiluFusionCompilerAttemptClaim.model_validate_json(external_claim_path.read_text())
        != claim
    ):
        raise ValueError("SEQAX_SILU_FUSION_EXTERNAL_CLAIM_MISMATCH")
    if _ledger_state(root, design, claim, capture) is not receipt.final_ledger_state:
        raise ValueError("SEQAX_SILU_FUSION_LEDGER_STATE_MISMATCH")
    for values in zip(_plans(design), receipt.capture.candidates, strict=True):
        (expected, distributed, physical, plan), recorded = values
        _validate_candidate(root, expected, distributed, physical, plan, recorded)
    if any(
        value.role not in SEQAX_SILU_FUSION_COMPILER_ARTIFACT_ROLES for value in receipt.artifacts
    ):
        raise ValueError("SEQAX_SILU_FUSION_NONCOMPILER_ARTIFACT")
    return receipt


def verify_pair(pair_path: Path, design_path: Path) -> SeqaxSiluFusionCompilerPair:
    if not pair_path.is_absolute():
        raise ValueError("SEQAX_SILU_FUSION_PAIR_PATH_NOT_ABSOLUTE")
    design = SeqaxSiluFusionDesignContract.model_validate_json(design_path.read_text())
    if design != default_seqax_silu_fusion_design_contract(design.runtime):
        raise ValueError("SEQAX_SILU_FUSION_DESIGN_NONCANONICAL")
    evidence_root = Path("/home/sudarshan/tpu-cake-evidence").resolve(strict=True)
    pattern = re.compile(
        rf"seqax-silu-fusion-compiler-pair-{design.design_id[:7]}-[0-9a-f]{{8}}\.json$"
    )
    if (
        pair_path.parent != evidence_root
        or pattern.fullmatch(pair_path.name) is None
        or pair_path.is_symlink()
        or not pair_path.is_file()
        or pair_path != pair_path.resolve(strict=True)
    ):
        raise ValueError("SEQAX_SILU_FUSION_PAIR_PATH_INVALID")
    info = pair_path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_nlink != 1
        or info.st_mode & 0o077
    ):
        raise ValueError("SEQAX_SILU_FUSION_PAIR_FILE_INVALID")
    pair = SeqaxSiluFusionCompilerPair.model_validate_json(pair_path.read_text())
    if pair.design_id != design.design_id:
        raise ValueError("SEQAX_SILU_FUSION_PAIR_DESIGN_MISMATCH")
    for member in pair.captures:
        root = Path(member.capture_root).resolve(strict=True)
        if str(root) != member.capture_root:
            raise ValueError("SEQAX_SILU_FUSION_PAIR_CAPTURE_ROOT_NONCANONICAL")
        receipt = verify_capture(root, design_path)
        seal_path = _registry_file(
            design,
            f"{design.compiler_claim_key}-{member.capture_ordinal}.replay.json",
        )
        seal = SeqaxSiluFusionCompilerReplaySeal.model_validate_json(seal_path.read_text())
        if seqax_silu_fusion_compiler_pair_member(root, receipt, seal) != member:
            raise ValueError("SEQAX_SILU_FUSION_PAIR_CAPTURE_MISMATCH")
    return pair


def main() -> None:
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--root", type=Path)
    target.add_argument("--pair", type=Path)
    parser.add_argument("--design", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.pair is not None:
        pair = verify_pair(arguments.pair, arguments.design)
        print(
            json.dumps(
                {
                    "capture_ids": [value.capture_id for value in pair.captures],
                    "pair_id": pair.pair_id,
                },
                sort_keys=True,
            )
        )
        return
    receipt = verify_capture(arguments.root, arguments.design)
    print(
        json.dumps(
            {
                "capture_id": receipt.capture.capture_id,
                "receipt_id": receipt.receipt_id,
                "semantic_pair_id": receipt.capture.semantic_pair_id,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
