from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import tpu_cake.seqax_large_residual_qualification_runner as qualification_runner
from tpu_cake.runner import _runtime_identity
from tpu_cake.seqax_large_residual_qualification import (
    SeqaxLargeResidualHost,
    SeqaxLargeResidualQualificationContract,
    SeqaxLargeResidualQualificationFailureRecord,
    analyze_large_residual_boundary,
    default_seqax_large_residual_qualification_contract,
    default_seqax_large_residual_qualification_failure_record,
)
from tpu_cake.seqax_large_residual_qualification_runner import (
    _record_failure,
    _source_manifest,
    _validate_forensic_capture,
    _write_exclusive_json,
)
from tpu_cake.seqax_large_residual_runner import SeqaxLargeResidualCompilerCaptureRecord


def _chain(index: int) -> str:
    return "\n".join(
        (
            (
                f"%reduce-scatter.{index}.call-start = ((f32[1,128,4096]{{2,1,0}}, "
                f"token[]), f32[1,128,1024]{{2,1,0}}) call-start(%operand), "
                'async_execution_thread="sparsecore", '
                'metadata={op_name="jit(physical_call)/shard_map/reduce_scatter"}, '
                'sparse_core_config={"offload":"OFFLOAD_COLLECTIVE"}'
            ),
            (
                f"%reduce-scatter.{index}.call-done = f32[1,128,1024]{{2,1,0}} "
                f"call-done(%reduce-scatter.{index}.call-start), "
                'metadata={op_name="jit(physical_call)/shard_map/reduce_scatter"}'
            ),
            (
                f"%convert_add_fusion.{index} = bf16[1,128,1024]{{2,1,0}} "
                f"fusion(%residual, %reduce-scatter.{index}.call-done), kind=kLoop, "
                'metadata={op_name="jit(physical_call)/shard_map/add"}'
            ),
            (
                f"%all-gather.{index}.call-start = ((bf16[1,128,1024]{{2,1,0}}, "
                f"token[]), bf16[1,128,4096]{{2,1,0}}) "
                f"call-start(%convert_add_fusion.{index}), "
                'async_execution_thread="sparsecore", '
                'metadata={op_name="jit(physical_call)/shard_map/all_gather"}, '
                'sparse_core_config={"offload":"OFFLOAD_COLLECTIVE"}'
            ),
        )
    )


def test_qualification_contract_round_trips_and_binds_pinned_sources() -> None:
    contract = default_seqax_large_residual_qualification_contract(_runtime_identity())
    saved = SeqaxLargeResidualQualificationContract.model_validate_json(
        contract.model_dump_json(exclude_computed_fields=True)
    )

    assert saved == contract
    assert saved.large_residual_contract_id == (
        "55a9da09cc188d8a582ddad566757d2f94413d3e9449ddc30f8603cb3661d9c1"
    )
    assert saved.compiler_capture_record_id == (
        "da420165c7ea57ac4d5b2830062803942236a2726411c5dc9b3bcc0bb1a63ff6"
    )
    assert saved.numerical_policy_source_contract_id == (
        "861cc164be5ed7a322cca351902cfc1400a49c4cebd3ab8bcf09c2a10ee6e905"
    )
    assert len(saved.correctness_seeds) == 5
    assert saved.repeat_executions == 2
    assert saved.allow_resume is False
    assert saved.allow_retry is False
    assert saved.collect_profile is False
    assert saved.collect_timings is False
    assert saved.correctness_scope == "final-output-plus-semantic-compiler-boundary-v2"
    assert saved.compiler_hlo_replay_rule == (
        "stablehlo-exact-compiler-collectives-memory-and-boundary-lineage-v2"
    )
    assert saved.superseded_qualification_id == (
        "5aaac3984ba05fcc995576b533ec82908ddb204f0a8c0ef8a0323a8504e6f341"
    )
    assert saved.hostname == "tpu-cake-v7x-rsag-wx7r"
    assert saved.instance_id == "5064039476077763048"


def test_external_qualification_contract_is_canonical() -> None:
    saved = SeqaxLargeResidualQualificationContract.model_validate_json(
        Path("contracts/seqax-large-residual-qualification-v2.json").read_text()
    )

    assert saved == default_seqax_large_residual_qualification_contract(saved.runtime)


def test_failed_model_4096_qualification_record_is_canonical() -> None:
    saved = SeqaxLargeResidualQualificationFailureRecord.model_validate_json(
        Path("contracts/seqax-large-residual-qualification-failure-v1.json").read_text()
    )

    assert saved == default_seqax_large_residual_qualification_failure_record()
    assert saved.assessment.pallas_top1_matches_control
    assert saved.assessment.cross_path_relative_l2 < 0.001
    assert not saved.assessment.final_outputs_satisfy_policy
    assert not saved.qualification_passed
    assert not saved.timing_authorized


def test_qualification_contract_rejects_relaxed_repeat_policy() -> None:
    contract = default_seqax_large_residual_qualification_contract(_runtime_identity())
    payload = contract.model_dump(exclude_computed_fields=True)
    payload["repeat_executions"] = 1

    with pytest.raises(ValidationError, match="greater than or equal to 2"):
        SeqaxLargeResidualQualificationContract.model_validate(payload)


def test_boundary_analysis_requires_two_full_rs_residual_ag_chains() -> None:
    analysis = analyze_large_residual_boundary(f"{_chain(6)}\n{_chain(8)}\n")

    assert analysis.chain_count == 2
    assert analysis.reduce_scatter_starts == (
        "reduce-scatter.6.call-start",
        "reduce-scatter.8.call-start",
    )


def test_boundary_analysis_rejects_count_only_false_positive() -> None:
    hlo = (
        _chain(6).replace(
            "%convert_add_fusion.6",
            "%unrelated_fusion.6",
        )
        + "\n"
        + _chain(8)
    )

    with pytest.raises(ValueError, match="BOUNDARY_CHAIN_MISMATCH"):
        analyze_large_residual_boundary(hlo)


def test_boundary_analysis_does_not_count_bf16_embedding_reduce_scatter() -> None:
    embedding = (
        _chain(4)
        .replace("f32[1,128,4096]", "bf16[1,128,4096]")
        .replace(
            "f32[1,128,1024]",
            "bf16[1,128,1024]",
        )
    )

    analysis = analyze_large_residual_boundary(f"{embedding}\n{_chain(6)}\n{_chain(8)}")

    assert analysis.chain_count == 2
    assert "reduce-scatter.4.call-start" not in analysis.reduce_scatter_starts


def test_source_manifest_explicitly_binds_shared_execution_helpers() -> None:
    paths = {value.path for value in _source_manifest()}

    assert "tpu_cake/artifacts.py" in paths
    assert "tpu_cake/seqax_pallas_search_runner.py" in paths
    assert "tpu_cake/seqax_large_residual_qualification.py" in paths
    assert "tpu_cake/seqax_large_residual_qualification_runner.py" in paths


def test_forensic_capture_proves_raw_hlo_is_not_the_semantic_gate() -> None:
    contract = default_seqax_large_residual_qualification_contract(_runtime_identity())
    pinned = SeqaxLargeResidualCompilerCaptureRecord.model_validate_json(
        Path("contracts/seqax-large-residual-compiler-captures-v1.json").read_text()
    )
    candidates = tuple(
        value.model_copy(
            update={
                "pallas_compiler_hlo_sha256": "0" * 64,
                "control_compiler_hlo_sha256": "1" * 64,
            }
        )
        for value in pinned.capture.candidates
    )
    forensic = pinned.capture.model_copy(
        update={
            "source_commit": contract.superseded_source_commit,
            "candidates": candidates,
        }
    )

    assert (
        _validate_forensic_capture(
            (forensic.model_dump_json() + "\n").encode(),
            contract,
            pinned,
        )
        == forensic
    )

    changed = forensic.model_copy(
        update={
            "candidates": (
                forensic.candidates[0].model_copy(update={"pallas_stablehlo_sha256": "2" * 64}),
                forensic.candidates[1],
            )
        }
    )
    with pytest.raises(ValueError, match="FORENSIC_DIAGNOSIS_MISMATCH"):
        _validate_forensic_capture(
            (changed.model_dump_json() + "\n").encode(),
            contract,
            pinned,
        )


def test_expected_host_is_exactly_the_assigned_tpu_vm() -> None:
    contract = default_seqax_large_residual_qualification_contract(_runtime_identity())

    assert qualification_runner._expected_host(contract) == SeqaxLargeResidualHost(
        project="astral-medley-465922-b2",
        numeric_project_id="541760035156",
        zone="us-central1-c",
        hostname="tpu-cake-v7x-rsag-wx7r",
        instance_hostname=(
            "tpu-cake-v7x-rsag-wx7r.us-central1-c.c.astral-medley-465922-b2.internal"
        ),
        machine_type="tpu7x-standard-4t",
        instance_id="5064039476077763048",
        cpu_platform="Intel Emerald Rapids",
        zone_resource="projects/541760035156/zones/us-central1-c",
        machine_type_resource=("projects/541760035156/machineTypes/tpu7x-standard-4t"),
    )


def test_permanent_claim_write_is_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "claim.json"
    payload = {"state": "claimed", "attempt_id": "1" * 64}

    _write_exclusive_json(path, payload)

    assert json.loads(path.read_text()) == payload
    with pytest.raises(FileExistsError):
        _write_exclusive_json(path, payload)


def test_permanent_claim_survives_post_create_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "claim.json"

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(qualification_runner.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="injected fsync failure"):
        _write_exclusive_json(path, {"state": "claimed"})

    assert path.exists()
    with pytest.raises(FileExistsError):
        _write_exclusive_json(path, {"state": "claimed"})


def test_failure_record_requires_claim_and_never_overwrites(tmp_path: Path) -> None:
    _record_failure(tmp_path, "1" * 64, ValueError("before claim"))
    assert not (tmp_path / "failure.json").exists()

    (tmp_path / "attempt_claim.json").write_text("{}\n")
    _record_failure(tmp_path, "1" * 64, ValueError("after claim"))
    first = (tmp_path / "failure.json").read_bytes()
    _record_failure(tmp_path, "1" * 64, RuntimeError("later"))

    assert (tmp_path / "failure.json").read_bytes() == first
    assert json.loads(first)["error"] == "after claim"
