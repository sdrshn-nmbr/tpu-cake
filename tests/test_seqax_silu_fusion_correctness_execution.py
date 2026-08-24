from __future__ import annotations

import ast
import stat
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from tpu_cake import seqax_silu_fusion_correctness_runner as correctness_runner
from tpu_cake.contracts import RuntimeIdentity, SourceFileContract
from tpu_cake.ledger import EvidenceRun, RunState, finalize_ledger
from tpu_cake.seqax_contract_types import SeqaxFeedForwardFusion
from tpu_cake.seqax_silu_fusion_correctness import (
    SEQAX_SILU_FUSION_FUSED_CHECKPOINT_CAPTURE_MODES,
    SEQAX_SILU_FUSION_SEPARATE_CHECKPOINT_CAPTURE_MODES,
    SeqaxSiluFusionCandidateCorrectness,
    SeqaxSiluFusionCheckpointMetrics,
    SeqaxSiluFusionCorrectnessAttemptClaim,
    SeqaxSiluFusionCorrectnessContract,
    SeqaxSiluFusionCorrectnessDevice,
    SeqaxSiluFusionCorrectnessFailure,
    SeqaxSiluFusionCorrectnessFailureReceipt,
    SeqaxSiluFusionCorrectnessHost,
    SeqaxSiluFusionCorrectnessObservation,
    SeqaxSiluFusionCorrectnessPlan,
    SeqaxSiluFusionCorrectnessReceipt,
    SeqaxSiluFusionCorrectnessResult,
    SeqaxSiluFusionCorrectnessSourceAuthority,
    SeqaxSiluFusionCorrectnessWorkerResult,
    SeqaxSiluFusionFinalOutputMetrics,
)
from tpu_cake.seqax_silu_fusion_correctness_runner import (
    _artifact_role as runner_artifact_role,
)
from tpu_cake.seqax_silu_fusion_correctness_verifier import (
    _artifact_role as verifier_artifact_role,
)
from tpu_cake.seqax_silu_fusion_correctness_verifier import _validate_ledger

_ROOT = Path(__file__).resolve().parents[1]


def _source_authority() -> SeqaxSiluFusionCorrectnessSourceAuthority:
    return SeqaxSiluFusionCorrectnessSourceAuthority(
        source_commit="a" * 40,
        source_tree="b" * 40,
        branch="main",
        origin_main_commit="a" * 40,
        remote_main_commit="a" * 40,
        remote_url="https://github.com/sdrshn-nmbr/tpu-cake.git",
        source_root="/home/sudarshan/tpu-cake-main",
        uv_lock_sha256="03c153a4daf4f1bf2c77d89620824e4f6c11fa946a9166f0f512e195d1025ed9",
        cli_sha256="355040b20f7e48683811b009fc77f460652617fafcdc44c68a3d7309fd71f740",
        correctness_contract_sha256=(
            "ffd334ec7ed3449266a5e4229eae1439ba6eb12afedf395819ca1e3ef542ffdd"
        ),
        compiler_design_sha256=("e6c7fd18da0edba925b2bc6e0725c25b51f233f06013a9147c5eca0c977d0bff"),
        compiler_pair_sha256=("06ed3cc20bc6642a91f8dbddf7ae9ed56c705f73a57efe15b60d135c25626807"),
        correctness_schema_source_sha256="c" * 64,
        runner_source_sha256="d" * 64,
        worker_source_sha256="e" * 64,
        verifier_source_sha256="f" * 64,
        source_manifest=(SourceFileContract(path="uv.lock", sha256="0" * 64),),
        runtime=RuntimeIdentity(
            python="3.12.3",
            jax="0.11.0",
            jaxlib="0.11.0",
            libtpu="0.0.44.1",
            xla=" --xla_tpu_use_enhanced_launch_barrier=true",
        ),
    )


def _synthetic_result(
    contract: SeqaxSiluFusionCorrectnessContract,
    claim: SeqaxSiluFusionCorrectnessAttemptClaim,
    source: SeqaxSiluFusionCorrectnessSourceAuthority,
) -> SeqaxSiluFusionCorrectnessResult:
    output_metrics = SeqaxSiluFusionFinalOutputMetrics(
        cpu_relative_l2=0.0,
        cpu_row_scaled_max=0.0,
        cpu_top1_match=True,
        final_output_policy_passed=True,
    )
    checkpoint_metrics = SeqaxSiluFusionCheckpointMetrics(
        rms_mean_square_max_bound_ratio=0.0,
        rms_inverse_relative_error_units=0.0,
        normalized_float32_max_bound_ratio=0.0,
        gate_float32_max_bound_ratio=0.0,
        silu_max_ulp_of_mathematical=0,
        up_float32_max_bound_ratio=0.0,
        hidden_matches_product=True,
        down_float32_max_bound_ratio=0.0,
        bfloat16_conversions_match=True,
        checkpoint_values_consistent=True,
        full_assessment_sha256="1" * 64,
    )

    def candidate(
        fusion: SeqaxFeedForwardFusion,
    ) -> SeqaxSiluFusionCandidateCorrectness:
        modes = (
            SEQAX_SILU_FUSION_SEPARATE_CHECKPOINT_CAPTURE_MODES
            if fusion is SeqaxFeedForwardFusion.SEPARATE
            else SEQAX_SILU_FUSION_FUSED_CHECKPOINT_CAPTURE_MODES
        )
        return SeqaxSiluFusionCandidateCorrectness(
            candidate=fusion,
            uninstrumented_output_sha256="2" * 64,
            instrumented_output_sha256="2" * 64,
            checkpoint_sha256=("3" * 64,) * 13,
            checkpoint_capture_modes=modes,
            uninstrumented_metrics=output_metrics,
            instrumented_metrics=output_metrics,
            checkpoint_metrics=checkpoint_metrics,
            instrumentation_output_exact=True,
        )

    candidates = (
        candidate(SeqaxFeedForwardFusion.SEPARATE),
        candidate(SeqaxFeedForwardFusion.SILU_MULTIPLY),
    )
    seeds = (*contract.correctness_seeds, contract.boundary_seed)
    observations = tuple(
        SeqaxSiluFusionCorrectnessObservation(
            seed=seed,
            input_sha256=("4" * 64,) * 13,
            cpu_reference_sha256="5" * 64,
            candidates=candidates,
            candidate_uninstrumented_outputs_exact=True,
            candidate_instrumented_outputs_exact=True,
            candidate_checkpoints_exact=True,
            boundary_case=index == len(seeds) - 1,
            boundary_strict_mutant_difference_count=(1 if index == len(seeds) - 1 else 0),
            boundary_mutant_rejected=index == len(seeds) - 1,
        )
        for index, seed in enumerate(seeds)
    )
    plans = tuple(
        SeqaxSiluFusionCorrectnessPlan(
            candidate=fusion,
            candidate_semantic_id=semantic_id,
            distributed_schedule_sha256="6" * 64,
            physical_schedule_sha256="7" * 64,
            pallas_source_sha256="8" * 64,
            pallas_manifest_sha256="9" * 64,
            uninstrumented_stablehlo_sha256="a" * 64,
            uninstrumented_pre_optimization_hlo_sha256="b" * 64,
            uninstrumented_compiler_hlo_sha256="c" * 64,
            instrumented_stablehlo_sha256="d" * 64,
            instrumented_pre_optimization_hlo_sha256="e" * 64,
            instrumented_compiler_hlo_sha256="f" * 64,
        )
        for fusion, semantic_id in zip(
            (
                SeqaxFeedForwardFusion.SEPARATE,
                SeqaxFeedForwardFusion.SILU_MULTIPLY,
            ),
            contract.candidate_semantic_ids,
            strict=True,
        )
    )
    return SeqaxSiluFusionCorrectnessResult(
        contract_id=contract.contract_id,
        claim_id=claim.claim_id,
        source=source,
        host=SeqaxSiluFusionCorrectnessHost(
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
        ),
        devices=tuple(
            SeqaxSiluFusionCorrectnessDevice(
                id=index,
                process_index=0,
                platform="tpu",
                device_kind="TPU7x",
            )
            for index in range(8)
        ),
        plans=plans,
        observations=observations,
        model_outputs_executed=True,
        full_inputs_persisted=True,
        full_outputs_persisted=True,
        full_checkpoints_persisted=True,
        producer_passed=True,
        timing_collected=False,
        profile_collected=False,
        worker_pid=1,
        worker_nonce="0" * 32,
        source_import_root="/isolated/source/committed/src",
        worker_environment=contract.worker_environment,
        compiler_environment=contract.compiler_environment,
    )


@pytest.mark.parametrize(
    "path",
    (
        "attempt_claim.json",
        "compiler_pair.json",
        "contract.json",
        "ledger.sqlite",
        "source.json",
        "source/committed/uv.lock",
        "worker_request.json",
        "worker-result.json",
        "worker-failure.json",
        "plans/separate/distributed.xdsl",
        "plans/separate/physical.xdsl",
        "plans/separate/lowered_pallas.py",
        "plans/separate/plan_manifest.json",
        "plans/separate/uninstrumented_stablehlo.txt",
        "plans/separate/uninstrumented_pre_optimization_hlo.txt",
        "plans/silu_multiply/instrumented_pre_optimization_hlo.txt",
        "plans/silu_multiply/instrumented_compiler_hlo.txt",
        "plans/silu_multiply/compiler_candidate.json",
        "seeds/seed-1/inputs/00.npy",
        "seeds/seed-1/cpu_reference.npy",
        "seeds/seed-1/separate/uninstrumented_output.npy",
        "seeds/seed-1/silu_multiply/checkpoints/hidden_bfloat16.npy",
        "seeds/seed-1/observation.json",
        "seeds/seed-1/boundary/mutant_hidden.npy",
    ),
)
def test_runner_and_verifier_use_same_closed_artifact_roles(path: str) -> None:
    relative = Path(path)

    assert runner_artifact_role(relative) is verifier_artifact_role(relative)


@pytest.mark.parametrize(
    "path",
    (
        "timing.json",
        "profile.xplane.pb",
        "seeds/seed-1/trace.json",
        "plans/separate/counters.json",
    ),
)
def test_correctness_artifact_roles_reject_performance_evidence(path: str) -> None:
    with pytest.raises(ValueError, match="ARTIFACT_ROLE_UNKNOWN"):
        runner_artifact_role(Path(path))
    with pytest.raises(ValueError, match="ARTIFACT_ROLE_UNKNOWN"):
        verifier_artifact_role(Path(path))


def test_correctness_runner_import_does_not_initialize_jax() -> None:
    result = subprocess.run(
        (
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; import tpu_cake.seqax_silu_fusion_correctness_runner; "
                "assert not any(name == 'jax' or name.startswith('jax.') "
                "for name in sys.modules)"
            ),
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr


def test_source_authority_persisted_form_round_trips_without_computed_id() -> None:
    source = _source_authority()

    payload = source.model_dump_json(exclude_computed_fields=True)

    assert "source_authority_id" not in payload
    assert SeqaxSiluFusionCorrectnessSourceAuthority.model_validate_json(payload) == source


def test_correctness_run_accepts_verified_compiler_rebind() -> None:
    contract = SeqaxSiluFusionCorrectnessContract.model_validate_json(
        (_ROOT / "contracts/seqax-silu-fusion-correctness-v1.json").read_text()
    )

    correctness_runner._require_compiler_evidence_ready(contract)


def test_archive_replay_restores_validated_private_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if correctness_runner.shutil.which("zstd") is None:
        pytest.skip("zstd is required for archive replay")
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    (root / "artifact.json").write_text("{}\n")
    archive, _sha256, _members = correctness_runner._create_archive(root)

    def verify(path: Path, *, final: bool, relocated: bool) -> dict[str, str]:
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
        assert final
        assert relocated
        return {"status": "accepted"}

    monkeypatch.setattr(correctness_runner, "_verify", verify)

    def verify_failure(path: Path, *, relocated: bool) -> dict[str, str]:
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
        assert relocated
        return {"status": "failed"}

    monkeypatch.setattr(correctness_runner, "_verify_failure", verify_failure)

    assert correctness_runner._verify_extracted_archive(archive, root.name) == {
        "status": "accepted"
    }
    assert correctness_runner._verify_extracted_failure_archive(archive, root.name) == {
        "status": "failed"
    }


def test_archive_replay_rejects_nonprivate_recorded_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if correctness_runner.shutil.which("zstd") is None:
        pytest.skip("zstd is required for archive replay")
    root = tmp_path / "evidence"
    root.mkdir(mode=0o755)
    archive, _sha256, _members = correctness_runner._create_archive(root)
    monkeypatch.setattr(
        correctness_runner,
        "_verify",
        lambda *_args, **_kwargs: pytest.fail("invalid archive reached verifier"),
    )

    with pytest.raises(ValueError, match="ARCHIVE_LAYOUT_INVALID"):
        correctness_runner._verify_extracted_archive(archive, root.name)


def test_mock_success_lifecycle_replays_after_safe_archive_relocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if correctness_runner.shutil.which("zstd") is None:
        pytest.skip("zstd is required for archive replay")
    contract_path = _ROOT / "contracts/seqax-silu-fusion-correctness-v1.json"
    contract = SeqaxSiluFusionCorrectnessContract.model_validate_json(contract_path.read_text())
    source = _source_authority()
    root = tmp_path / "run"
    claim = SeqaxSiluFusionCorrectnessAttemptClaim(
        contract_id=contract.contract_id,
        invocation_id="1" * 32,
        source_commit=source.source_commit,
        source_tree=source.source_tree,
        output_root=str(root),
    )
    result = _synthetic_result(contract, claim, source)
    worker_result = SeqaxSiluFusionCorrectnessWorkerResult(result=result)
    registry = tmp_path / "registry"
    registry.mkdir(mode=0o700)
    claim_path = registry / "claim.json"
    blobs = {
        "contracts/seqax-silu-fusion-correctness-v1.json": contract_path.read_bytes(),
        "contracts/seqax-silu-fusion-compiler-pair-v1.json": (
            _ROOT / "contracts/seqax-silu-fusion-compiler-pair-v1.json"
        ).read_bytes(),
        "uv.lock": b"synthetic lifecycle source\n",
    }

    def launch_worker(path: Path, *_args: object) -> subprocess.CompletedProcess[str]:
        run = EvidenceRun(path / "ledger.sqlite", claim.claim_id)
        for state in (
            RunState.VERIFIED,
            RunState.LOWERED,
            RunState.COMPILED,
            RunState.CORRECT,
        ):
            run.transition(state, {"state": state.value})
        correctness_runner._write_json_exclusive(
            path / "worker-result.json",
            worker_result.model_dump(mode="json", exclude_computed_fields=True),
        )
        return subprocess.CompletedProcess([], 0, "", "")

    def verify(path: Path, *, final: bool, relocated: bool = False) -> dict[str, str]:
        if not final:
            assert not relocated
            return {"result_id": result.result_id}
        receipt = SeqaxSiluFusionCorrectnessReceipt.model_validate_json(
            (path / "receipt.json").read_text()
        )
        if relocated:
            assert path != root
            assert stat.S_IMODE(path.stat().st_mode) == 0o700
        return {
            "result_id": receipt.result.result_id,
            "receipt_id": receipt.receipt_id,
        }

    monkeypatch.setattr(correctness_runner, "_require_safe_new_root", lambda *_: root)
    monkeypatch.setattr(correctness_runner, "_source_authority", lambda *_: (source, blobs))
    monkeypatch.setattr(correctness_runner, "_claim_lock", lambda *_: nullcontext())
    monkeypatch.setattr(
        correctness_runner,
        "_prepare_claim",
        lambda *_: (claim_path, claim),
    )
    monkeypatch.setattr(correctness_runner, "_launch_worker", launch_worker)
    monkeypatch.setattr(correctness_runner, "_verify", verify)
    monkeypatch.setattr(
        correctness_runner,
        "_registry_path",
        lambda _contract, suffix: registry / f"{suffix}.json",
    )

    receipt, replay, archive = correctness_runner.run_correctness(root, contract_path)

    assert receipt.result.result_id == result.result_id
    assert replay.receipt_id == receipt.receipt_id
    assert archive.replay_seal_id == replay.replay_seal_id
    assert Path(archive.archive_path).is_file()
    assert claim_path.is_file()
    assert (
        SeqaxSiluFusionCorrectnessReceipt.model_validate_json((root / "receipt.json").read_text())
        == receipt
    )


def test_post_worker_empty_controller_failure_is_frozen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path = _ROOT / "contracts/seqax-silu-fusion-correctness-v1.json"
    contract = SeqaxSiluFusionCorrectnessContract.model_validate_json(contract_path.read_text())
    source = _source_authority()
    root = tmp_path / "run"
    claim = SeqaxSiluFusionCorrectnessAttemptClaim(
        contract_id=contract.contract_id,
        invocation_id="1" * 32,
        source_commit=source.source_commit,
        source_tree=source.source_tree,
        output_root=str(root),
    )
    claim_path = tmp_path / "registry" / "claim.json"
    blobs = {
        "contracts/seqax-silu-fusion-correctness-v1.json": contract_path.read_bytes(),
        "contracts/seqax-silu-fusion-compiler-pair-v1.json": (
            _ROOT / "contracts/seqax-silu-fusion-compiler-pair-v1.json"
        ).read_bytes(),
    }
    fake_result = SimpleNamespace(
        contract_id=contract.contract_id,
        claim_id=claim.claim_id,
        source=source,
        result_id="2" * 64,
    )

    class WorkerResultParser:
        @classmethod
        def model_validate_json(cls, _value: str) -> SimpleNamespace:
            return SimpleNamespace(result=fake_result)

    def verify_failure(path: Path, *, relocated: bool = False) -> dict[str, str]:
        del relocated
        receipt = SeqaxSiluFusionCorrectnessFailureReceipt.model_validate_json(
            (path / "failure-receipt.json").read_text()
        )
        return {"failure_receipt_id": receipt.failure_receipt_id}

    def launch_worker(path: Path, *_args: object) -> subprocess.CompletedProcess[str]:
        run = EvidenceRun(path / "ledger.sqlite", claim.claim_id)
        for state in (
            RunState.VERIFIED,
            RunState.LOWERED,
            RunState.COMPILED,
            RunState.CORRECT,
        ):
            run.transition(state, {"state": state.value})
        (path / "worker-result.json").write_text("{}")
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(correctness_runner, "_require_safe_new_root", lambda *_: root)
    monkeypatch.setattr(correctness_runner, "_require_compiler_evidence_ready", lambda *_: None)
    monkeypatch.setattr(correctness_runner, "_source_authority", lambda *_: (source, blobs))
    monkeypatch.setattr(correctness_runner, "_claim_lock", lambda *_: nullcontext())
    monkeypatch.setattr(
        correctness_runner,
        "_prepare_claim",
        lambda *_: (claim_path, claim),
    )
    monkeypatch.setattr(correctness_runner, "_reserve_claim", lambda *_: None)
    monkeypatch.setattr(
        correctness_runner,
        "_launch_worker",
        launch_worker,
    )
    monkeypatch.setattr(
        correctness_runner,
        "SeqaxSiluFusionCorrectnessWorkerResult",
        WorkerResultParser,
    )
    monkeypatch.setattr(
        correctness_runner,
        "_verify",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError()),
    )
    monkeypatch.setattr(correctness_runner, "_verify_failure", verify_failure)
    monkeypatch.setattr(
        correctness_runner,
        "_create_archive",
        lambda *_args, **_kwargs: (tmp_path / "failure.tar.zst", "3" * 64, 1),
    )
    monkeypatch.setattr(
        correctness_runner,
        "_verify_extracted_failure_archive",
        lambda *_args, **_kwargs: verify_failure(root),
    )
    monkeypatch.setattr(
        correctness_runner,
        "_registry_path",
        lambda _contract, label: tmp_path / "registry" / f"{label}.json",
    )

    with pytest.raises(RuntimeError, match="phase=preliminary-replay"):
        correctness_runner.run_correctness(root, contract_path)

    failure = SeqaxSiluFusionCorrectnessFailure.model_validate_json(
        (root / "worker-failure.json").read_text()
    )
    assert failure.phase == "preliminary-replay"
    assert failure.returncode == 0
    assert failure.error_type == "RuntimeError"
    assert failure.error_message == "RuntimeError()"
    assert (root / "failure-receipt.json").is_file()
    assert (tmp_path / "registry" / "failure-replay.json").is_file()
    assert (tmp_path / "registry" / "failure-archive.json").is_file()


def test_verifier_does_not_reuse_worker_implementation() -> None:
    path = _ROOT / "src/tpu_cake/seqax_silu_fusion_correctness_verifier.py"
    tree = ast.parse(path.read_text())
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "tpu_cake.seqax_silu_fusion_correctness_worker" not in imported_modules


def test_correctness_ledger_accepts_only_the_declared_path(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.sqlite"
    run_id = "a" * 64
    run = EvidenceRun(ledger_path, run_id)
    run.create({"state": "created"})
    for state in (
        RunState.VERIFIED,
        RunState.LOWERED,
        RunState.COMPILED,
        RunState.CORRECT,
        RunState.VALIDATED,
        RunState.ACCEPTED,
    ):
        run.transition(state, {"state": state.value})
    finalize_ledger(ledger_path)

    _validate_ledger(tmp_path, run_id, final=True)


def test_correctness_ledger_rejects_timing_state(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.sqlite"
    run_id = "b" * 64
    run = EvidenceRun(ledger_path, run_id)
    run.create({"state": "created"})
    for state in (
        RunState.VERIFIED,
        RunState.LOWERED,
        RunState.COMPILED,
        RunState.CORRECT,
        RunState.TIMED,
        RunState.ACCEPTED,
    ):
        run.transition(state, {"state": state.value})
    finalize_ledger(ledger_path)

    with pytest.raises(ValueError, match="LEDGER_MISMATCH"):
        _validate_ledger(tmp_path, run_id, final=True)
