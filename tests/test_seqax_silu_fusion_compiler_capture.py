from __future__ import annotations

import ast
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tpu_cake.compiler_analysis import (
    CompilerCollectiveAnalysis,
    CompilerCostMetric,
    CompilerExecutableAnalysis,
    CompilerMemoryAnalysis,
)
from tpu_cake.contracts import ArtifactReference, ArtifactRole, RuntimeIdentity, SourceFileContract
from tpu_cake.ledger import EvidenceRun, RunState
from tpu_cake.seqax_contract_types import SeqaxFeedForwardFusion
from tpu_cake.seqax_silu_fusion import default_seqax_silu_fusion_design_contract
from tpu_cake.seqax_silu_fusion_compiler import (
    SeqaxSiluFusionCompilerAnalysis,
    SeqaxSiluFusionCompilerAttemptClaim,
    SeqaxSiluFusionCompilerCall,
    SeqaxSiluFusionCompilerCandidate,
    SeqaxSiluFusionCompilerCapture,
    SeqaxSiluFusionCompilerDevice,
    SeqaxSiluFusionCompilerFailureReceipt,
    SeqaxSiluFusionCompilerFailureReplaySeal,
    SeqaxSiluFusionCompilerHostIdentity,
    SeqaxSiluFusionCompilerPair,
    SeqaxSiluFusionCompilerPairMember,
    SeqaxSiluFusionCompilerReceipt,
    SeqaxSiluFusionCompilerSourceAuthority,
    SeqaxSiluFusionCompilerWorkerRequest,
    SeqaxSiluFusionCompilerWorkerResult,
    exact_integral_ring_equivalent_bytes,
)
from tpu_cake.seqax_silu_fusion_compiler_pair import (
    _safe_pair_path,
    _verifier_environment,
)
from tpu_cake.seqax_silu_fusion_compiler_runner import (
    _artifact_role as _runner_artifact_role,
)
from tpu_cake.seqax_silu_fusion_compiler_runner import (
    _claim_capture,
    _record_worker_failure,
    _require_prior_replay_seal,
    _require_safe_new_root,
    _subprocess_environment,
)
from tpu_cake.seqax_silu_fusion_compiler_verifier import (
    _artifact_role as _verifier_artifact_role,
)
from tpu_cake.seqax_silu_fusion_compiler_verifier import (
    _failure_ledger_state,
    _ledger_state,
    _preflight_root,
    _registry_file,
    _replay_failed_collective_gate,
    _validate_buffer_assignment_artifact,
    _validate_stablehlo,
)
from tpu_cake.seqax_silu_fusion_compiler_worker import (
    _write_buffer_assignment_if_available,
)

_ROOT = Path(__file__).resolve().parents[1]


def _pair_member(ordinal: int, *, marker: str) -> SeqaxSiluFusionCompilerPairMember:
    return SeqaxSiluFusionCompilerPairMember(
        capture_root=(
            f"/home/sudarshan/tpu-cake-evidence/"
            f"seqax-silu-fusion-compiler-aaaaaaa-{ordinal}-{marker * 8}"
        ),
        capture_ordinal=ordinal,
        capture_id=marker * 64,
        receipt_id=("c" if ordinal == 0 else "d") * 64,
        replay_seal_id=("d" if ordinal == 0 else "e") * 64,
        claim_id=("e" if ordinal == 0 else "f") * 64,
        invocation_id=("1" if ordinal == 0 else "2") * 32,
        worker_pid=100 + ordinal,
        worker_nonce=("3" if ordinal == 0 else "4") * 32,
        source_commit="5" * 40,
        source_tree="6" * 40,
        source_authority_id="7" * 64,
        host_id="8" * 64,
        compiler_environment_id="9" * 64,
        worker_environment_id="0" * 64,
        device_inventory_id="a" * 64,
        semantic_pair_id="b" * 64,
        candidate_semantic_ids=("c" * 64, "d" * 64),
    )


def _pair() -> SeqaxSiluFusionCompilerPair:
    return SeqaxSiluFusionCompilerPair(
        design_id="a" * 64,
        captures=(_pair_member(0, marker="1"), _pair_member(1, marker="2")),
        independent_replay_performed=True,
        model_outputs_executed=False,
        correctness_outputs_collected=False,
        timing_collected=False,
        profile_collected=False,
    )


def _compiler_candidate(
    kind: SeqaxFeedForwardFusion,
    collectives: CompilerCollectiveAnalysis,
) -> SeqaxSiluFusionCompilerCandidate:
    kernels = (
        ("seqax_strict_bf16_silu", "seqax_strict_bf16_multiply")
        if kind is SeqaxFeedForwardFusion.SEPARATE
        else ("seqax_strict_bf16_silu_multiply",)
    )
    calls = tuple(
        SeqaxSiluFusionCompilerCall(
            ordinal=index,
            kernel=kernel,
            output_shape="bf16[128,1,1024]",
            operand_count=1 if kernel == "seqax_strict_bf16_silu" else 2,
            schedule_sha256="2" * 64,
            vector_region_index=index,
            implementation="pallas_full_local",
            instruction_name=f"fusion-{index}",
            operand_names=("gate",) if kernel == "seqax_strict_bf16_silu" else ("gate", "up"),
        )
        for index, kernel in enumerate(kernels)
    )
    memory = CompilerMemoryAnalysis(
        generated_code_size_in_bytes=1,
        argument_size_in_bytes=2,
        output_size_in_bytes=3,
        alias_size_in_bytes=0,
        temp_size_in_bytes=4,
        host_generated_code_size_in_bytes=0,
        host_argument_size_in_bytes=0,
        host_output_size_in_bytes=0,
        host_alias_size_in_bytes=0,
        host_temp_size_in_bytes=0,
        peak_memory_in_bytes=5,
        buffer_assignment_available=False,
        buffer_assignment_size_bytes=0,
        buffer_assignment_sha256=None,
    )
    fusion = SeqaxSiluFusionCompilerAnalysis(
        candidate=kind,
        strict_vector_call_count=len(calls),
        calls=calls,
        all_strict_vector_calls_are_live=True,
        gate_and_up_projection_lineages_are_distinct=True,
        silu_output_feeds_multiply=kind is SeqaxFeedForwardFusion.SEPARATE,
        vector_output_feeds_one_down_projection=True,
    )
    analysis = CompilerExecutableAnalysis(
        stablehlo_sha256="9" * 64,
        compiler_hlo_sha256="a" * 64,
        cost_metrics=(
            CompilerCostMetric(name="flops", raw_value=1.0, value=1.0, available=True),
        ),
        memory=memory,
        collectives=collectives,
    )
    return SeqaxSiluFusionCompilerCandidate(
        candidate=kind,
        distributed_schedule_sha256="1" * 64,
        physical_schedule_sha256="2" * 64,
        pallas_source_sha256="3" * 64,
        pallas_manifest_sha256="4" * 64,
        pre_optimization_hlo_sha256="5" * 64,
        compiler_analysis=analysis,
        reachable_collectives=collectives,
        fusion_analysis=fusion,
        buffer_assignment_size_bytes=0,
        buffer_assignment_sha256=None,
        allocated_vmem_bytes_per_device=6,
        peak_live_vmem_bytes_per_device=7,
        ring_equivalent_ici_bytes_per_device=8,
    )


def test_pair_requires_two_independent_same_semantic_captures() -> None:
    pair = _pair()

    assert tuple(value.capture_ordinal for value in pair.captures) == (0, 1)
    assert len(pair.pair_id) == 64

    duplicate_invocation = pair.captures[1].model_copy(
        update={"invocation_id": pair.captures[0].invocation_id}
    )
    with pytest.raises(ValidationError, match="PAIR_INDEPENDENCE_MISMATCH"):
        SeqaxSiluFusionCompilerPair(
            **pair.model_dump(exclude={"captures"}, exclude_computed_fields=True),
            captures=(pair.captures[0], duplicate_invocation),
        )

    different_semantics = pair.captures[1].model_copy(update={"semantic_pair_id": "0" * 64})
    with pytest.raises(ValidationError, match="PAIR_SEMANTIC_MISMATCH"):
        SeqaxSiluFusionCompilerPair(
            **pair.model_dump(exclude={"captures"}, exclude_computed_fields=True),
            captures=(pair.captures[0], different_semantics),
        )


def test_candidate_semantic_identity_excludes_raw_compiler_hashes() -> None:
    collectives = CompilerCollectiveAnalysis(
        stablehlo_reduce_scatter_count=1,
        stablehlo_all_gather_count=15,
        compiler_reduce_scatter_count=1,
        compiler_all_reduce_count=2,
        compiler_all_gather_count=15,
        sparse_core_reduce_scatter_count=1,
        sparse_core_all_gather_count=15,
    )
    fusion = SeqaxSiluFusionCompilerAnalysis(
        candidate=SeqaxFeedForwardFusion.SILU_MULTIPLY,
        strict_vector_call_count=1,
        calls=(
            SeqaxSiluFusionCompilerCall(
                ordinal=0,
                kernel="seqax_strict_bf16_silu_multiply",
                output_shape="bf16[128,1,1024]",
                operand_count=2,
                schedule_sha256="2" * 64,
                vector_region_index=0,
                implementation="pallas_full_local",
                instruction_name="fusion",
                operand_names=("gate", "up"),
            ),
        ),
        all_strict_vector_calls_are_live=True,
        gate_and_up_projection_lineages_are_distinct=True,
        silu_output_feeds_multiply=False,
        vector_output_feeds_one_down_projection=True,
    )
    fields = {
        "candidate": SeqaxFeedForwardFusion.SILU_MULTIPLY,
        "distributed_schedule_sha256": "1" * 64,
        "physical_schedule_sha256": "2" * 64,
        "pallas_source_sha256": "3" * 64,
        "pallas_manifest_sha256": "4" * 64,
        "pre_optimization_hlo_sha256": "5" * 64,
        "compiler_analysis": None,
        "reachable_collectives": collectives,
        "fusion_analysis": fusion,
        "buffer_assignment_size_bytes": 123,
        "buffer_assignment_sha256": "6" * 64,
        "allocated_vmem_bytes_per_device": 456,
        "peak_live_vmem_bytes_per_device": 321,
        "ring_equivalent_ici_bytes_per_device": 789,
    }
    first = SeqaxSiluFusionCompilerCandidate.model_construct(**fields)
    second = SeqaxSiluFusionCompilerCandidate.model_construct(
        **{
            **fields,
            "pre_optimization_hlo_sha256": "7" * 64,
            "buffer_assignment_size_bytes": 456,
            "buffer_assignment_sha256": "8" * 64,
        }
    )

    assert first.semantic_id == second.semantic_id


def test_candidate_accepts_backend_without_serialized_buffer_assignment() -> None:
    collectives = CompilerCollectiveAnalysis(
        stablehlo_reduce_scatter_count=1,
        stablehlo_all_gather_count=15,
        compiler_reduce_scatter_count=1,
        compiler_all_reduce_count=2,
        compiler_all_gather_count=15,
        sparse_core_reduce_scatter_count=1,
        sparse_core_all_gather_count=15,
    )
    candidate = _compiler_candidate(SeqaxFeedForwardFusion.SILU_MULTIPLY, collectives)

    assert candidate.buffer_assignment_size_bytes == 0
    assert candidate.buffer_assignment_sha256 is None

    payload = candidate.model_dump(exclude_computed_fields=True)
    payload["reachable_collectives"]["compiler_all_reduce_count"] = 1
    with pytest.raises(ValidationError, match="COLLECTIVE_REACHABILITY_MISMATCH"):
        SeqaxSiluFusionCompilerCandidate.model_validate(payload)


def test_ring_equivalent_bytes_require_an_exact_integer() -> None:
    assert exact_integral_ring_equivalent_bytes(Decimal(323744)) == 323_744
    for value in (Decimal("323744.5"), Decimal(0), Decimal("NaN"), Decimal("Infinity")):
        with pytest.raises(ValueError, match="RING_EQUIVALENT_BYTES_INVALID"):
            exact_integral_ring_equivalent_bytes(value)


def test_capture_rejects_candidate_collective_strategy_drift() -> None:
    collectives = CompilerCollectiveAnalysis(
        stablehlo_reduce_scatter_count=1,
        stablehlo_all_gather_count=15,
        compiler_reduce_scatter_count=0,
        compiler_all_reduce_count=5,
        compiler_all_gather_count=9,
        sparse_core_reduce_scatter_count=0,
        sparse_core_all_gather_count=9,
    )
    changed = collectives.model_copy(update={"compiler_all_reduce_count": 4})
    separate = _compiler_candidate(SeqaxFeedForwardFusion.SEPARATE, collectives)
    fused = _compiler_candidate(SeqaxFeedForwardFusion.SILU_MULTIPLY, changed)
    runtime = RuntimeIdentity(python="3.12.3")
    design = default_seqax_silu_fusion_design_contract(runtime)
    source = SeqaxSiluFusionCompilerSourceAuthority(
        source_commit="1" * 40,
        source_tree="2" * 40,
        branch="main",
        origin_main_commit="1" * 40,
        remote_main_commit="1" * 40,
        remote_url=design.source_remote_url,
        source_root=design.compilation_source_root,
        uv_lock_sha256="3" * 64,
        cli_sha256="355040b20f7e48683811b009fc77f460652617fafcdc44c68a3d7309fd71f740",
        design_file_sha256="4" * 64,
        runner_source_sha256="5" * 64,
        worker_source_sha256="6" * 64,
        compiler_source_sha256="7" * 64,
        pair_source_sha256="8" * 64,
        source_manifest=(SourceFileContract(path="source.py", sha256="9" * 64),),
        runtime=runtime,
    )

    with pytest.raises(ValidationError, match="COLLECTIVE_PARITY_MISMATCH"):
        SeqaxSiluFusionCompilerCapture(
            design_id="a" * 64,
            capture_ordinal=0,
            invocation_id="b" * 32,
            claim_id="c" * 64,
            source=source,
            host=SeqaxSiluFusionCompilerHostIdentity(
                project="project",
                numeric_project_id="1",
                zone="zone",
                hostname="host",
                instance_hostname="instance",
                machine_type="machine",
                instance_id="2",
                cpu_platform="platform",
            ),
            worker_environment={},
            compiler_environment={},
            source_import_root="/source",
            compile_input_mode="abstract-only",
            devices=tuple(
                SeqaxSiluFusionCompilerDevice(
                    id=index,
                    process_index=0,
                    platform="tpu",
                    device_kind="TPU7x",
                )
                for index in range(8)
            ),
            worker_pid=1,
            worker_nonce="d" * 32,
            candidates=(separate, fused),
            model_outputs_executed=False,
            correctness_outputs_collected=False,
            timing_collected=False,
            profile_collected=False,
        )


def test_failure_replay_requires_the_recorded_collective_gate() -> None:
    expected = SimpleNamespace(
        expected_all_gathers=15,
        expected_reduce_scatters=1,
        expected_compiler_all_gathers=9,
        expected_compiler_all_reduces=5,
        expected_compiler_reduce_scatters=0,
        expected_sparse_core_all_gathers=9,
        expected_sparse_core_reduce_scatters=0,
    )
    required = CompilerCollectiveAnalysis(
        stablehlo_reduce_scatter_count=1,
        stablehlo_all_gather_count=15,
        compiler_reduce_scatter_count=0,
        compiler_all_reduce_count=5,
        compiler_all_gather_count=9,
        sparse_core_reduce_scatter_count=0,
        sparse_core_all_gather_count=9,
    )
    changed = required.model_copy(update={"compiler_all_reduce_count": 4})
    diagnostic = "SEQAX_SILU_FUSION_COLLECTIVE_STRATEGY_MISMATCH"

    assert _replay_failed_collective_gate(expected, changed, changed, diagnostic) == diagnostic
    with pytest.raises(ValueError, match="COLLECTIVE_GATE_REPLAY_MISMATCH"):
        _replay_failed_collective_gate(expected, changed, changed, None)
    assert _replay_failed_collective_gate(expected, required, required, diagnostic) is None


def test_verifier_matches_optional_buffer_assignment_artifact(tmp_path: Path) -> None:
    fields = {
        "generated_code_size_in_bytes": 1,
        "argument_size_in_bytes": 2,
        "output_size_in_bytes": 3,
        "alias_size_in_bytes": 0,
        "temp_size_in_bytes": 4,
        "host_generated_code_size_in_bytes": 0,
        "host_argument_size_in_bytes": 0,
        "host_output_size_in_bytes": 0,
        "host_alias_size_in_bytes": 0,
        "host_temp_size_in_bytes": 0,
        "peak_memory_in_bytes": 5,
    }
    unavailable = CompilerMemoryAnalysis(
        **fields,
        buffer_assignment_available=False,
        buffer_assignment_size_bytes=0,
        buffer_assignment_sha256=None,
    )

    _validate_buffer_assignment_artifact(
        tmp_path,
        unavailable,
        required_if_available=True,
    )
    (tmp_path / "buffer_assignment.pb").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="BUFFER_ASSIGNMENT_UNEXPECTED"):
        _validate_buffer_assignment_artifact(
            tmp_path,
            unavailable,
            required_if_available=True,
        )

    payload = b"available-buffer-assignment"
    available = CompilerMemoryAnalysis(
        **fields,
        buffer_assignment_available=True,
        buffer_assignment_size_bytes=len(payload),
        buffer_assignment_sha256=hashlib.sha256(payload).hexdigest(),
    )
    (tmp_path / "buffer_assignment.pb").unlink()
    with pytest.raises(ValueError, match="BUFFER_ASSIGNMENT_MISSING"):
        _validate_buffer_assignment_artifact(
            tmp_path,
            available,
            required_if_available=True,
        )
    _validate_buffer_assignment_artifact(
        tmp_path,
        available,
        required_if_available=False,
    )
    (tmp_path / "buffer_assignment.pb").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="BUFFER_ASSIGNMENT_MISMATCH"):
        _validate_buffer_assignment_artifact(
            tmp_path,
            available,
            required_if_available=True,
        )
    (tmp_path / "buffer_assignment.pb").write_bytes(payload)
    _validate_buffer_assignment_artifact(
        tmp_path,
        available,
        required_if_available=True,
    )


def test_worker_writes_buffer_assignment_only_when_available(tmp_path: Path) -> None:
    fields = {
        "generated_code_size_in_bytes": 1,
        "argument_size_in_bytes": 2,
        "output_size_in_bytes": 3,
        "alias_size_in_bytes": 0,
        "temp_size_in_bytes": 4,
        "host_generated_code_size_in_bytes": 0,
        "host_argument_size_in_bytes": 0,
        "host_output_size_in_bytes": 0,
        "host_alias_size_in_bytes": 0,
        "host_temp_size_in_bytes": 0,
        "peak_memory_in_bytes": 5,
    }
    unavailable = CompilerMemoryAnalysis(
        **fields,
        buffer_assignment_available=False,
        buffer_assignment_size_bytes=0,
        buffer_assignment_sha256=None,
    )
    _write_buffer_assignment_if_available(tmp_path, unavailable, b"")
    assert not (tmp_path / "buffer_assignment.pb").exists()

    payload = b"available-buffer-assignment"
    available = CompilerMemoryAnalysis(
        **fields,
        buffer_assignment_available=True,
        buffer_assignment_size_bytes=len(payload),
        buffer_assignment_sha256=hashlib.sha256(payload).hexdigest(),
    )
    _write_buffer_assignment_if_available(tmp_path, available, payload)
    assert (tmp_path / "buffer_assignment.pb").read_bytes() == payload


def test_receipt_rejects_noncompiler_artifact_roles() -> None:
    artifact = ArtifactReference(
        path="timing.json",
        size_bytes=2,
        sha256="0" * 64,
        role=ArtifactRole.TIMING_SAMPLES,
    )

    receipt = SeqaxSiluFusionCompilerReceipt.model_construct(
        capture=SeqaxSiluFusionCompilerCapture.model_construct(),
        final_ledger_state=RunState.COMPILED,
        artifacts=(artifact,),
    )

    with pytest.raises(ValueError, match="NONCOMPILER_ARTIFACT"):
        receipt.artifacts_are_compile_only()


def test_capture_root_must_be_canonical_direct_evidence_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    monkeypatch.setattr(
        "tpu_cake.seqax_silu_fusion_compiler_runner._EVIDENCE_ROOT",
        evidence,
    )
    design = SimpleNamespace(design_id="a" * 64)
    name = "seqax-silu-fusion-compiler-aaaaaaa-0-12345678"
    canonical = evidence / name

    assert _require_safe_new_root(canonical, design, 0) == canonical
    with pytest.raises(ValueError, match="ROOT_NOT_DIRECT_EVIDENCE_CHILD"):
        _require_safe_new_root(evidence / "nested" / name, design, 0)
    with pytest.raises(ValueError, match="ROOT_NOT_CANONICAL"):
        _require_safe_new_root(evidence / "nested" / ".." / name, design, 0)


def test_pair_path_is_exclusive_canonical_evidence_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    monkeypatch.setattr(
        "tpu_cake.seqax_silu_fusion_compiler_pair._EVIDENCE_ROOT",
        evidence,
    )
    design = SimpleNamespace(design_id="a" * 64)
    name = "seqax-silu-fusion-compiler-pair-aaaaaaa-12345678.json"
    canonical = evidence / name

    assert _safe_pair_path(canonical, design) == canonical
    with pytest.raises(ValueError, match="PATH_OUTSIDE_EVIDENCE_ROOT"):
        _safe_pair_path(evidence / "nested" / name, design)
    with pytest.raises(ValueError, match="PATH_NOT_CANONICAL"):
        _safe_pair_path(evidence / "nested" / ".." / name, design)


def test_ordinal_one_requires_completed_ordinal_zero_replay_seal(tmp_path: Path) -> None:
    design = SimpleNamespace(
        compiler_claim_registry_root=str(tmp_path / "claims"),
        compiler_claim_key="seqax-silu-fusion-design-v1",
        design_id="a" * 64,
    )
    source = SimpleNamespace(source_commit="1" * 40, source_tree="2" * 40)

    with pytest.raises(ValueError, match="PRIOR_REPLAY_SEAL_MISSING"):
        _require_prior_replay_seal(design, source)


def test_claim_is_permanent_for_one_full_design_identity(tmp_path: Path) -> None:
    design = SimpleNamespace(
        compiler_claim_registry_root=str(tmp_path / "claims"),
        compiler_claim_key="seqax-silu-fusion-design-v1",
        design_id="a" * 64,
    )
    source = SimpleNamespace(source_commit="1" * 40, source_tree="2" * 40)

    claim_path, claim = _claim_capture(tmp_path / "capture", design, 0, source)

    assert claim_path.name == f"seqax-silu-fusion-design-v1-{'a' * 64}-0.json"
    assert claim.design_id == design.design_id
    with pytest.raises(ValueError, match="CAPTURE_PERMANENTLY_CLAIMED"):
        _claim_capture(tmp_path / "other-capture", design, 0, source)


def test_worker_request_wire_payload_excludes_nested_computed_fields() -> None:
    runtime = RuntimeIdentity(python="3.12.3")
    design = default_seqax_silu_fusion_design_contract(runtime)
    source = SeqaxSiluFusionCompilerSourceAuthority(
        source_commit="1" * 40,
        source_tree="2" * 40,
        branch="main",
        origin_main_commit="1" * 40,
        remote_main_commit="1" * 40,
        remote_url=design.source_remote_url,
        source_root=design.compilation_source_root,
        uv_lock_sha256="3" * 64,
        cli_sha256="355040b20f7e48683811b009fc77f460652617fafcdc44c68a3d7309fd71f740",
        design_file_sha256="4" * 64,
        runner_source_sha256="5" * 64,
        worker_source_sha256="6" * 64,
        compiler_source_sha256="7" * 64,
        pair_source_sha256="8" * 64,
        source_manifest=(SourceFileContract(path="source.py", sha256="9" * 64),),
        runtime=runtime,
    )
    claim = SeqaxSiluFusionCompilerAttemptClaim(
        design_id=design.design_id,
        capture_ordinal=0,
        invocation_id="a" * 32,
        source_commit=source.source_commit,
        source_tree=source.source_tree,
        output_root="/home/sudarshan/tpu-cake-evidence/capture",
    )
    request = SeqaxSiluFusionCompilerWorkerRequest(
        claim=claim,
        design=design,
        source=source,
    )

    payload = request.wire_payload()

    assert "claim_id" not in payload["claim"]
    assert "design_id" not in payload["design"]
    assert SeqaxSiluFusionCompilerWorkerRequest.model_validate_json(json.dumps(payload)) == request


def test_worker_result_wire_payload_excludes_nested_computed_fields() -> None:
    candidate = SeqaxSiluFusionCompilerCandidate.model_construct(
        candidate=SeqaxFeedForwardFusion.SEPARATE
    )
    capture = SeqaxSiluFusionCompilerCapture.model_construct(candidates=(candidate, candidate))
    result = SeqaxSiluFusionCompilerWorkerResult.model_construct(capture=capture)

    payload = result.wire_payload()

    assert "capture_id" not in payload["capture"]
    assert "semantic_pair_id" not in payload["capture"]
    assert all("semantic_id" not in value for value in payload["capture"]["candidates"])


def test_failed_worker_gets_an_immutable_replay_bound_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = RuntimeIdentity(python="3.12.3")
    design = default_seqax_silu_fusion_design_contract(runtime)
    source = SeqaxSiluFusionCompilerSourceAuthority(
        source_commit="1" * 40,
        source_tree="2" * 40,
        branch="main",
        origin_main_commit="1" * 40,
        remote_main_commit="1" * 40,
        remote_url=design.source_remote_url,
        source_root=design.compilation_source_root,
        uv_lock_sha256="3" * 64,
        cli_sha256="355040b20f7e48683811b009fc77f460652617fafcdc44c68a3d7309fd71f740",
        design_file_sha256="4" * 64,
        runner_source_sha256="5" * 64,
        worker_source_sha256="6" * 64,
        compiler_source_sha256="7" * 64,
        pair_source_sha256="8" * 64,
        source_manifest=(SourceFileContract(path="source.py", sha256="9" * 64),),
        runtime=runtime,
    )
    claim = SeqaxSiluFusionCompilerAttemptClaim(
        design_id=design.design_id,
        capture_ordinal=0,
        invocation_id="a" * 32,
        source_commit=source.source_commit,
        source_tree=source.source_tree,
        output_root=str(tmp_path),
    )
    EvidenceRun(tmp_path / "ledger.sqlite", claim.claim_id).create(
        {
            "claim_id": claim.claim_id,
            "claim_path": str(
                Path(design.compiler_claim_registry_root)
                / f"{design.compiler_claim_key}-{design.design_id}-0.json"
            ),
            "design_id": design.design_id,
            "capture_ordinal": 0,
        }
    )

    def replay(root: Path, *, allow_missing_seal: bool) -> dict[str, object]:
        receipt = SeqaxSiluFusionCompilerFailureReceipt.model_validate_json(
            (root / "failure-receipt.json").read_text()
        )
        payload = {"failure_receipt_id": receipt.failure_receipt_id}
        seal_path = root / "failure-replay.json"
        if seal_path.exists():
            seal = SeqaxSiluFusionCompilerFailureReplaySeal.model_validate_json(
                seal_path.read_text()
            )
            payload["failure_replay_seal_id"] = seal.failure_replay_seal_id
        elif not allow_missing_seal:
            raise AssertionError("failure replay seal is required")
        return payload

    monkeypatch.setattr(
        "tpu_cake.seqax_silu_fusion_compiler_runner._independent_verify_failure",
        replay,
    )
    monkeypatch.setattr(
        "tpu_cake.seqax_silu_fusion_compiler_runner._failure_replay_seal_path",
        lambda _design, _ordinal: tmp_path / "failure-replay.json",
    )
    completed = subprocess.CompletedProcess(
        args=("worker",),
        returncode=1,
        stdout="",
        stderr="compiler gate failed",
    )

    receipt, replay_seal = _record_worker_failure(tmp_path, design, claim, source, completed)

    assert receipt.final_ledger_state is RunState.CREATED
    assert receipt.failure.stderr == "compiler gate failed"
    assert receipt.independent_replay_required
    assert not receipt.independent_replay_performed_at_receipt_creation
    assert replay_seal.independent_replay_performed
    assert replay_seal.failure_receipt_id == receipt.failure_receipt_id
    assert not receipt.retry_authorized
    assert {value.path for value in receipt.artifacts} == {
        "ledger.sqlite",
        "worker-failure.json",
    }
    assert (tmp_path / "failure-receipt.json").is_file()
    assert (tmp_path / "failure-replay.json").is_file()
    assert _failure_ledger_state(tmp_path, design, claim, RunState.CREATED) is RunState.CREATED

    ordinal_one_claim = claim.model_copy(update={"capture_ordinal": 1})
    with pytest.raises(ValidationError, match="FAILURE_ORDINAL_MISMATCH"):
        SeqaxSiluFusionCompilerFailureReceipt(
            claim=ordinal_one_claim,
            source=source,
            final_ledger_state=RunState.CREATED,
            failure=receipt.failure,
            artifacts=receipt.artifacts,
            independent_replay_required=True,
            independent_replay_performed_at_receipt_creation=False,
            retry_authorized=False,
            ordinal_one_launched=False,
        )


def test_artifact_roles_require_exact_candidate_paths() -> None:
    expected = Path("candidates/separate/stablehlo.txt")
    reachable = Path("candidates/separate/reachable_collectives.json")
    decoy = Path("timing/private/stablehlo.txt")

    assert _runner_artifact_role(expected) is ArtifactRole.STABLEHLO
    assert _verifier_artifact_role(expected) is ArtifactRole.STABLEHLO
    assert _runner_artifact_role(reachable) is ArtifactRole.COMPILER_ANALYSIS
    assert _verifier_artifact_role(reachable) is ArtifactRole.COMPILER_ANALYSIS
    assert _runner_artifact_role(Path("worker-failure.json")) is ArtifactRole.COMPILER_ANALYSIS
    assert _verifier_artifact_role(Path("worker-failure.json")) is ArtifactRole.COMPILER_ANALYSIS
    with pytest.raises(ValueError, match="ARTIFACT_ROLE_UNKNOWN"):
        _runner_artifact_role(decoy)
    with pytest.raises(ValueError, match="ARTIFACT_ROLE_UNKNOWN"):
        _verifier_artifact_role(decoy)


def test_stablehlo_rejects_duplicate_expected_strict_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Parsed:
        def live_collective_counts(self) -> dict[str, int]:
            return {"all_gather": 15, "all_reduce": 2, "reduce_scatter": 1}

    monkeypatch.setattr(
        "tpu_cake.seqax_silu_fusion_compiler_verifier.StableHloInspector.parse",
        staticmethod(lambda _value: _Parsed()),
    )
    expected = SimpleNamespace(
        expected_all_gathers=15,
        expected_all_reduces=2,
        expected_reduce_scatters=1,
        expected_strict_vector_kernels=("seqax_strict_bf16_silu_multiply",),
        expected_pallas_regions=9,
    )
    stablehlo = "\n".join(
        [
            *(['kernel_name = "seqax_named_einsum"'] * 9),
            'kernel_name = "seqax_strict_bf16_silu_multiply"',
            'kernel_name = "seqax_strict_bf16_silu_multiply"',
        ]
    )

    with pytest.raises(ValueError, match="STABLEHLO_VECTOR_KERNEL_MISMATCH"):
        _validate_stablehlo(stablehlo, expected)


def test_registry_files_are_private_owned_regular_files(tmp_path: Path) -> None:
    registry = tmp_path / "claims"
    registry.mkdir(mode=0o700)
    claim = registry / "claim.json"
    claim.write_text("{}")
    claim.chmod(0o600)
    design = SimpleNamespace(compiler_claim_registry_root=str(registry))

    assert _registry_file(design, "claim.json") == claim
    claim.chmod(0o644)
    with pytest.raises(ValueError, match="REGISTRY_FILE_INVALID"):
        _registry_file(design, "claim.json")


def test_capture_preflight_rejects_undeclared_fifo(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    root.mkdir(mode=0o700)
    fifo = root / "undeclared.pipe"
    os.mkfifo(fifo, mode=0o600)

    with pytest.raises(ValueError, match="ARTIFACT_NONREGULAR"):
        _preflight_root(root)


class _RecordedValue:
    def __init__(self, value: object) -> None:
        self.value = value

    def model_dump(self, *, mode: str) -> object:
        assert mode == "json"
        return self.value


def test_ledger_replays_payload_hashes(tmp_path: Path) -> None:
    database = tmp_path / "ledger.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE events (sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
            "run_id TEXT NOT NULL, state TEXT NOT NULL, timestamp_ns INTEGER NOT NULL, "
            "payload_sha256 TEXT NOT NULL, UNIQUE(run_id, state))"
        )
        connection.executemany(
            "INSERT INTO events(run_id, state, timestamp_ns, payload_sha256) VALUES (?, ?, ?, ?)",
            [
                ("a" * 64, state.value, index, "0" * 64)
                for index, state in enumerate(
                    (
                        RunState.CREATED,
                        RunState.VERIFIED,
                        RunState.LOWERED,
                        RunState.COMPILED,
                    ),
                    start=1,
                )
            ],
        )
    design = SimpleNamespace(
        compiler_claim_registry_root="/claims",
        compiler_claim_key="key",
        design_id="b" * 64,
        candidates=(
            SimpleNamespace(physical_schedule_sha256="c" * 64),
            SimpleNamespace(physical_schedule_sha256="d" * 64),
        ),
    )
    claim = SimpleNamespace(claim_id="a" * 64, capture_ordinal=0)
    capture = SimpleNamespace(
        source=SimpleNamespace(runtime=_RecordedValue({"runtime": "fixed"})),
        host=_RecordedValue({"host": "fixed"}),
        devices=(_RecordedValue({"id": 0}),),
        capture_id="e" * 64,
        semantic_pair_id="f" * 64,
    )

    with pytest.raises(ValueError, match="LEDGER_MISMATCH"):
        _ledger_state(tmp_path, design, claim, capture)


def test_parent_runner_claims_before_launch_and_never_imports_jax() -> None:
    path = _ROOT / "src/tpu_cake/seqax_silu_fusion_compiler_runner.py"
    source = path.read_text()
    module = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or "" for node in ast.walk(module) if isinstance(node, ast.ImportFrom)
    )
    run_capture = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_capture"
    )
    calls = {
        node.func.id: node.lineno
        for node in ast.walk(run_capture)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert not any(value == "jax" or value.startswith("jax.") for value in imports)
    assert calls["_claim_capture"] < calls["_launch_worker"]

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import tpu_cake.seqax_silu_fusion_compiler_runner; "
                "print(sum(name == 'jax' or name.startswith('jax.') for name in sys.modules))"
            ),
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == "0"


def test_archived_worker_and_verifiers_disable_bytecode_writes(tmp_path: Path) -> None:
    runtime = RuntimeIdentity(python="3.12.3")
    design = default_seqax_silu_fusion_design_contract(runtime)
    bundle = tmp_path / "bundle"

    assert _subprocess_environment(design)["PYTHONDONTWRITEBYTECODE"] == "1"
    assert _verifier_environment(bundle)["PYTHONDONTWRITEBYTECODE"] == "1"
    runner = (_ROOT / "src/tpu_cake/seqax_silu_fusion_compiler_runner.py").read_text()
    pair = (_ROOT / "src/tpu_cake/seqax_silu_fusion_compiler_pair.py").read_text()
    assert runner.count('"-B",') == 3
    assert pair.count('"-B",') == 2


def test_worker_launch_uses_bundled_committed_source() -> None:
    source = (_ROOT / "src/tpu_cake/seqax_silu_fusion_compiler_runner.py").read_text()

    assert 'bundle = root / "source" / "committed"' in source
    assert 'environment["PYTHONPATH"] = str(bundle / "src")' in source
    assert "cwd=bundle" in source


def test_worker_is_compile_only_and_uses_abstract_inputs() -> None:
    source = (_ROOT / "src/tpu_cake/seqax_silu_fusion_compiler_worker.py").read_text()

    assert "jax.ShapeDtypeStruct" in source
    assert "lowered.compile()" in source
    assert ".block_until_ready(" not in source
    assert "time.perf_counter" not in source
    assert "correctness_outputs_collected=False" in source
    assert "timing_collected=False" in source
    assert "profile_collected=False" in source


def test_worker_persists_compiler_evidence_before_collective_gate() -> None:
    source = (_ROOT / "src/tpu_cake/seqax_silu_fusion_compiler_worker.py").read_text()
    module = ast.parse(source)
    functions = {
        node.name: ast.get_source_segment(source, node)
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }
    compile_source = functions["_compile_raw"]
    qualify_source = functions["_qualify_compiled"]
    buffer_source = functions["_write_buffer_assignment_if_available"]
    worker_source = functions["run_worker"]
    assert compile_source is not None
    assert qualify_source is not None
    assert buffer_source is not None
    assert worker_source is not None

    for artifact in (
        'candidate_root / "stablehlo.txt"',
        'candidate_root / "pre_optimization_hlo.txt"',
        'candidate_root / "compiler_hlo.txt"',
    ):
        assert artifact in compile_source
    collective_gate = qualify_source.index(
        "validate_seqax_silu_fusion_compiler_collectives("
    )
    for artifact in (
        'candidate_root / "compiler_analysis.json"',
        'candidate_root / "reachable_collectives.json"',
        'candidate_root / "fusion_analysis.json"',
    ):
        assert qualify_source.index(artifact) < collective_gate
    assert qualify_source.index("_write_buffer_assignment_if_available(") < collective_gate
    buffer_policy = buffer_source.index("if memory.buffer_assignment_available:")
    assert buffer_policy < buffer_source.index('candidate_root / "buffer_assignment.pb"')
    assert "SEQAX_SILU_FUSION_BUFFER_ASSIGNMENT_UNAVAILABLE" not in buffer_source
    assert qualify_source.count("executable.memory_analysis()") == 1
    assert "_CompilerAnalysisExecutable(executable, runtime_memory)" in qualify_source
    assert qualify_source.count("exact_integral_ring_equivalent_bytes(") == 1
    verifier_source = (
        _ROOT / "src/tpu_cake/seqax_silu_fusion_compiler_verifier.py"
    ).read_text()
    assert verifier_source.count("exact_integral_ring_equivalent_bytes(") == 1
    assert worker_source.index("tuple(_compile_raw") < worker_source.index(
        "tuple(_qualify_compiled"
    )
