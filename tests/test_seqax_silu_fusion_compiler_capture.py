from __future__ import annotations

import ast
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tpu_cake.compiler_analysis import CompilerCollectiveAnalysis
from tpu_cake.contracts import ArtifactReference, ArtifactRole
from tpu_cake.ledger import RunState
from tpu_cake.seqax_contract_types import SeqaxFeedForwardFusion
from tpu_cake.seqax_silu_fusion_compiler import (
    SeqaxSiluFusionCompilerAnalysis,
    SeqaxSiluFusionCompilerCandidate,
    SeqaxSiluFusionCompilerCapture,
    SeqaxSiluFusionCompilerPair,
    SeqaxSiluFusionCompilerPairMember,
    SeqaxSiluFusionCompilerReceipt,
)
from tpu_cake.seqax_silu_fusion_compiler_pair import _safe_pair_path
from tpu_cake.seqax_silu_fusion_compiler_runner import (
    _artifact_role as _runner_artifact_role,
)
from tpu_cake.seqax_silu_fusion_compiler_runner import (
    _require_prior_replay_seal,
    _require_safe_new_root,
)
from tpu_cake.seqax_silu_fusion_compiler_verifier import (
    _artifact_role as _verifier_artifact_role,
)
from tpu_cake.seqax_silu_fusion_compiler_verifier import (
    _ledger_state,
    _preflight_root,
    _registry_file,
    _validate_stablehlo,
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
    fusion = SeqaxSiluFusionCompilerAnalysis.model_construct(
        candidate=SeqaxFeedForwardFusion.SILU_MULTIPLY,
        strict_vector_call_count=1,
        calls=(),
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
            "buffer_assignment_sha256": "8" * 64,
        }
    )

    assert first.semantic_id == second.semantic_id


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
    )
    source = SimpleNamespace(source_commit="1" * 40, source_tree="2" * 40)

    with pytest.raises(ValueError, match="PRIOR_REPLAY_SEAL_MISSING"):
        _require_prior_replay_seal(design, source)


def test_artifact_roles_require_exact_candidate_paths() -> None:
    expected = Path("candidates/separate/stablehlo.txt")
    decoy = Path("timing/private/stablehlo.txt")

    assert _runner_artifact_role(expected) is ArtifactRole.STABLEHLO
    assert _verifier_artifact_role(expected) is ArtifactRole.STABLEHLO
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
