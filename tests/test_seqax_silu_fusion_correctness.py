from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from tpu_cake.seqax_numerical import SeqaxBf16NumericalPolicy
from tpu_cake.seqax_silu_fusion import SeqaxSiluFusionDesignContract
from tpu_cake.seqax_silu_fusion_compiler import SeqaxSiluFusionCompilerPair
from tpu_cake.seqax_silu_fusion_correctness import (
    SEQAX_SILU_FUSION_CHECKPOINTS,
    SeqaxSiluFusionCorrectnessContract,
    default_seqax_silu_fusion_correctness_contract,
)

_ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_PATH = _ROOT / "contracts/seqax-silu-fusion-correctness-v1.json"


def _contract() -> SeqaxSiluFusionCorrectnessContract:
    return SeqaxSiluFusionCorrectnessContract.model_validate_json(_CONTRACT_PATH.read_text())


def test_seqax_silu_fusion_correctness_contract_is_canonical() -> None:
    contract = _contract()

    assert contract == default_seqax_silu_fusion_correctness_contract(contract.runtime)
    assert contract.checkpoint_names == SEQAX_SILU_FUSION_CHECKPOINTS
    assert contract.output_shape == (256, 1, 16)
    assert not contract.allow_retry
    assert not contract.allow_resume
    assert contract.compiler_evidence_status == "pending-rebind"
    assert contract.correctness_claim_identity_scope == "contract-id"
    assert contract.correctness_claim_reservation == "exclusive-create-only"
    assert not contract.timing_authorized
    assert not contract.profile_authorized
    assert not contract.policy.timing_collected
    assert not contract.policy.profile_collected


def test_seqax_silu_fusion_correctness_contract_is_canonical_json() -> None:
    contract = _contract()
    canonical = json.dumps(
        contract.model_dump(mode="json", exclude_computed_fields=True),
        indent=2,
    )

    assert _CONTRACT_PATH.read_text() == canonical + "\n"


def test_seqax_silu_fusion_correctness_marks_compiler_evidence_pending_rebind() -> None:
    contract = _contract()
    design_path = _ROOT / contract.compiler_design_path
    pair_path = _ROOT / contract.compiler_pair_record_path
    design = SeqaxSiluFusionDesignContract.model_validate_json(design_path.read_text())
    pair = SeqaxSiluFusionCompilerPair.model_validate_json(pair_path.read_text())

    assert hashlib.sha256(design_path.read_bytes()).hexdigest() != contract.compiler_design_sha256
    assert hashlib.sha256(pair_path.read_bytes()).hexdigest() == (contract.compiler_pair_sha256)
    assert design.design_id != contract.compiler_design_id == pair.design_id
    assert pair.pair_id == contract.compiler_pair_id
    assert tuple(capture.capture_id for capture in pair.captures) == (contract.compiler_capture_ids)
    assert pair.captures[0].candidate_semantic_ids == (contract.candidate_semantic_ids)
    assert pair.captures[1].candidate_semantic_ids == (contract.candidate_semantic_ids)
    assert contract.compiler_evidence_status == "pending-rebind"
    assert not pair.model_outputs_executed
    assert not pair.correctness_outputs_collected
    assert not pair.timing_collected
    assert not pair.profile_collected

    assert contract.parameters == design.parameters
    assert contract.residual_norm_strategy is design.residual_norm_strategy
    assert contract.vector_execution is design.vector_execution
    assert contract.candidates == (design.baseline, design.candidate)
    assert contract.correctness_seeds == design.correctness_seeds
    assert contract.boundary_seed == design.boundary_seed
    assert contract.compilation_source_root == design.compilation_source_root
    assert contract.source_remote_url == design.source_remote_url
    assert contract.source_branch == design.source_branch
    assert contract.worker_environment == design.worker_environment
    assert contract.compiler_environment == design.compiler_environment
    assert contract.project == design.project
    assert contract.numeric_project_id == design.numeric_project_id
    assert contract.zone == design.zone
    assert contract.hostname == design.hostname
    assert contract.instance_hostname == design.instance_hostname
    assert contract.machine_type == design.machine_type
    assert contract.instance_id == design.instance_id
    assert contract.cpu_platform == design.cpu_platform
    assert contract.runtime == design.runtime
    assert contract.backend == design.backend
    assert contract.device_kind == design.device_kind
    assert contract.device_count == design.device_count
    assert contract.mesh == design.mesh


def test_seqax_silu_fusion_correctness_uses_canonical_bf16_policy() -> None:
    policy = _contract().policy
    canonical = SeqaxBf16NumericalPolicy(
        cpu_relative_l2_units=policy.cpu_relative_l2_units,
        cpu_row_scaled_max_units=policy.cpu_row_scaled_max_units,
        cross_path_relative_l2_units=policy.cross_path_relative_l2_units,
        cross_path_row_scaled_max_units=policy.cross_path_row_scaled_max_units,
        row_scale_floor=policy.row_scale_floor,
        metric_quantization_decimals=policy.metric_quantization_decimals,
        mathematical_silu_max_ulp=policy.mathematical_silu_max_ulp,
        rms_inverse_relative_error_units=policy.rms_inverse_relative_error_units,
    )

    assert policy.numerical_policy_schema == canonical.schema_version
    assert policy.numerical_semantics == canonical.numerical_semantics
    assert policy.cpu_reference == canonical.cpu_reference
    assert policy.unit_roundoff == canonical.unit_roundoff
    assert policy.cpu_reference_quantization_decimals == (
        canonical.cpu_reference_quantization_decimals
    )
    assert policy.depth_scaling == canonical.depth_scaling
    assert policy.cpu_replay_rule == canonical.cpu_replay_rule
    assert policy.checkpoint_storage_dtype == canonical.checkpoint_storage_dtype
    assert policy.checkpoint_logical_dtype == canonical.checkpoint_logical_dtype
    assert policy.checkpoint_encoding == canonical.checkpoint_encoding


def test_seqax_silu_fusion_correctness_binds_current_lock() -> None:
    assert hashlib.sha256((_ROOT / "uv.lock").read_bytes()).hexdigest() == (
        _contract().uv_lock_sha256
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("allow_retry", True),
        ("allow_resume", True),
        ("timing_authorized", True),
        ("profile_authorized", True),
        ("independent_replay_required", False),
        ("archive_required", False),
        ("compiler_evidence_status", "verified"),
        ("compiler_capture_ids", ("0" * 64, "1" * 64)),
        ("correctness_seeds", (1, 2, 3, 4, 5)),
        ("candidates", ("silu_multiply", "separate")),
        ("output_shape", (256, 1, 32)),
    ),
)
def test_seqax_silu_fusion_correctness_rejects_protocol_mutation(
    field: str,
    value: object,
) -> None:
    payload = _contract().model_dump(exclude_computed_fields=True)
    payload[field] = value

    with pytest.raises(ValidationError):
        SeqaxSiluFusionCorrectnessContract.model_validate(payload)


def test_seqax_silu_fusion_correctness_rejects_checkpoint_oracle_mutation() -> None:
    payload = _contract().model_dump(exclude_computed_fields=True)
    payload["policy"]["require_each_candidate_checkpoint_values_consistent"] = False

    with pytest.raises(ValidationError):
        SeqaxSiluFusionCorrectnessContract.model_validate(payload)


def test_seqax_silu_fusion_correctness_contract_module_has_no_jax_import() -> None:
    result = subprocess.run(
        (
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; import tpu_cake.seqax_silu_fusion_correctness; "
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
