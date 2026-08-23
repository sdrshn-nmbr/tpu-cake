from __future__ import annotations

import argparse
import json
import os
import re
import stat
from collections import Counter
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

from tpu_cake.artifacts import file_sha256, validate_artifact_manifest
from tpu_cake.canonical import canonical_text
from tpu_cake.compiler_analysis import analyze_compiler_collectives
from tpu_cake.contracts import ArtifactRole
from tpu_cake.dialects.tpu_schedule import CollectiveOp, VectorComputeOp
from tpu_cake.identity import array_sha256, json_sha256
from tpu_cake.ledger import RunState, read_ledger_history
from tpu_cake.seqax_numerical import (
    SeqaxBf16NumericalPolicy,
    SeqaxBf16ScenarioParameters,
    assess_seqax_bf16_candidate_checkpoints,
    assess_seqax_bf16_final_outputs,
    decode_seqax_bf16_checkpoint,
    rounded_mathematical_silu_bf16,
    seqax_bf16_checkpoint_contract,
)
from tpu_cake.seqax_pallas_lowering import lower_seqax_physical_to_pallas
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.seqax_silu_fusion import SeqaxSiluFusionDesignContract
from tpu_cake.seqax_silu_fusion_compiler import (
    SeqaxSiluFusionCompilerCandidate,
    SeqaxSiluFusionCompilerPair,
    analyze_seqax_silu_fusion_compiler_hlo,
    live_seqax_silu_fusion_compiler_hlo,
)
from tpu_cake.seqax_silu_fusion_correctness import (
    SeqaxSiluFusionCandidateCorrectness,
    SeqaxSiluFusionCheckpointMetrics,
    SeqaxSiluFusionCorrectnessAttemptClaim,
    SeqaxSiluFusionCorrectnessContract,
    SeqaxSiluFusionCorrectnessFailure,
    SeqaxSiluFusionCorrectnessFailureReceipt,
    SeqaxSiluFusionCorrectnessObservation,
    SeqaxSiluFusionCorrectnessPlan,
    SeqaxSiluFusionCorrectnessReceipt,
    SeqaxSiluFusionCorrectnessSourceAuthority,
    SeqaxSiluFusionCorrectnessWorkerRequest,
    SeqaxSiluFusionCorrectnessWorkerResult,
    SeqaxSiluFusionFinalOutputMetrics,
    default_seqax_silu_fusion_correctness_contract,
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

_BUNDLED_CONTRACT_PATH = Path("contracts/seqax-silu-fusion-correctness-v1.json")
_BUNDLED_DESIGN_PATH = Path("contracts/seqax-silu-fusion-design-v1.json")
_BUNDLED_PAIR_PATH = Path("contracts/seqax-silu-fusion-compiler-pair-v1.json")


def _artifact_role(path: Path) -> ArtifactRole:
    value = path.as_posix()
    fixed = {
        "attempt_claim.json": ArtifactRole.INVOCATION,
        "compiler_pair.json": ArtifactRole.COMPILER_ANALYSIS,
        "contract.json": ArtifactRole.EXPERIMENT,
        "ledger.sqlite": ArtifactRole.EXECUTION_LEDGER,
        "source.json": ArtifactRole.SOURCE_STATE,
        "source_manifest.json": ArtifactRole.SOURCE_STATE,
        "worker_request.json": ArtifactRole.INVOCATION,
        "worker-result.json": ArtifactRole.CORRECTNESS_OUTPUT,
        "worker-failure.json": ArtifactRole.CORRECTNESS_OUTPUT,
    }
    if value in fixed:
        return fixed[value]
    if value.startswith("source/committed/"):
        return ArtifactRole.SOURCE_STATE
    plan_match = re.fullmatch(r"plans/(separate|silu_multiply)/([^/]+)", value)
    if plan_match is not None:
        roles = {
            "distributed.xdsl": ArtifactRole.DISTRIBUTED_IR,
            "physical.xdsl": ArtifactRole.PHYSICAL_IR,
            "lowered_pallas.py": ArtifactRole.PALLAS_SOURCE,
            "plan_manifest.json": ArtifactRole.PLAN_MANIFEST,
            "uninstrumented_stablehlo.txt": ArtifactRole.STABLEHLO,
            "uninstrumented_pre_optimization_hlo.txt": ArtifactRole.COMPILER_HLO,
            "uninstrumented_compiler_hlo.txt": ArtifactRole.COMPILER_HLO,
            "instrumented_stablehlo.txt": ArtifactRole.STABLEHLO,
            "instrumented_pre_optimization_hlo.txt": ArtifactRole.COMPILER_HLO,
            "instrumented_compiler_hlo.txt": ArtifactRole.COMPILER_HLO,
            "compiler_candidate.json": ArtifactRole.COMPILER_ANALYSIS,
            "record.json": ArtifactRole.COMPILER_ANALYSIS,
        }
        role = roles.get(plan_match.group(2))
        if role is not None:
            return role
    if re.fullmatch(r"seeds/seed-[0-9]+/inputs/[0-9]{2}\.npy", value):
        return ArtifactRole.CORRECTNESS_INPUT
    if re.fullmatch(r"seeds/seed-[0-9]+/cpu_reference\.npy", value):
        return ArtifactRole.ORACLE_OUTPUT
    correctness_patterns = (
        r"seeds/seed-[0-9]+/observation\.json",
        r"seeds/seed-[0-9]+/boundary/(strict|mutant)_hidden\.npy",
        r"seeds/seed-[0-9]+/(separate|silu_multiply)/(un)?instrumented_output\.npy",
        (
            r"seeds/seed-[0-9]+/(separate|silu_multiply)/"
            r"(un)?instrumented_assessment\.json"
        ),
        r"seeds/seed-[0-9]+/(separate|silu_multiply)/checkpoint_assessment\.json",
        (
            r"seeds/seed-[0-9]+/(separate|silu_multiply)/checkpoints/"
            r"(rms_input|rms_mean_square|rms_inverse|normalized_float32|"
            r"normalized_bfloat16|gate_float32|gate_bfloat16|silu_bfloat16|"
            r"up_float32|up_bfloat16|hidden_bfloat16|down_float32|down_bfloat16)\.npy"
        ),
    )
    if any(re.fullmatch(pattern, value) for pattern in correctness_patterns):
        return ArtifactRole.CORRECTNESS_OUTPUT
    raise ValueError(f"SEQAX_SILU_FUSION_CORRECTNESS_ARTIFACT_ROLE_UNKNOWN path={value}")


def _preflight_root(root: Path) -> Path:
    root = root.resolve(strict=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_ROOT_INVALID")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_SYMLINK_FORBIDDEN")
        if path.is_file() and path.stat().st_nlink != 1:
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_HARDLINK_FORBIDDEN")
    return root


def _load_array(path: Path) -> np.ndarray:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"SEQAX_SILU_FUSION_CORRECTNESS_ARRAY_INVALID path={path}")
    with path.open("rb") as stream:
        value = np.load(stream, allow_pickle=False)
        if stream.read(1):
            raise ValueError(f"SEQAX_SILU_FUSION_CORRECTNESS_ARRAY_TRAILING_DATA path={path}")
    if value.dtype.hasobject:
        raise ValueError(f"SEQAX_SILU_FUSION_CORRECTNESS_ARRAY_OBJECT_DTYPE path={path}")
    return value


def _validate_source_bundle(
    root: Path,
    source: SeqaxSiluFusionCorrectnessSourceAuthority,
) -> Path:
    bundle = root / "source" / "committed"
    observed_paths = tuple(
        sorted(path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file())
    )
    expected_paths = tuple(value.path for value in source.source_manifest)
    if observed_paths != expected_paths:
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_SOURCE_BUNDLE_SET_MISMATCH")
    for expected in source.source_manifest:
        path = bundle / expected.path
        if file_sha256(path) != expected.sha256:
            raise ValueError(
                f"SEQAX_SILU_FUSION_CORRECTNESS_SOURCE_BLOB_MISMATCH path={expected.path}"
            )
    explicit = {
        _BUNDLED_CONTRACT_PATH: source.correctness_contract_sha256,
        _BUNDLED_DESIGN_PATH: source.compiler_design_sha256,
        _BUNDLED_PAIR_PATH: source.compiler_pair_sha256,
        Path("src/tpu_cake/seqax_silu_fusion_correctness.py"): (
            source.correctness_schema_source_sha256
        ),
        Path("src/tpu_cake/seqax_silu_fusion_correctness_runner.py"): (source.runner_source_sha256),
        Path("src/tpu_cake/seqax_silu_fusion_correctness_worker.py"): (source.worker_source_sha256),
        Path("src/tpu_cake/seqax_silu_fusion_correctness_verifier.py"): (
            source.verifier_source_sha256
        ),
        Path("src/tpu_cake/cli.py"): source.cli_sha256,
        Path("uv.lock"): source.uv_lock_sha256,
    }
    if any(file_sha256(bundle / path) != expected for path, expected in explicit.items()):
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_SOURCE_AUTHORITY_MISMATCH")
    return bundle


def _validate_protocol(
    root: Path,
    *,
    relocated: bool,
) -> tuple[
    SeqaxSiluFusionCorrectnessContract,
    SeqaxSiluFusionDesignContract,
    SeqaxSiluFusionCompilerPair,
    SeqaxSiluFusionCorrectnessSourceAuthority,
    SeqaxSiluFusionCorrectnessAttemptClaim,
    SeqaxSiluFusionCorrectnessWorkerRequest,
]:
    contract = SeqaxSiluFusionCorrectnessContract.model_validate_json(
        (root / "contract.json").read_text()
    )
    if contract != default_seqax_silu_fusion_correctness_contract(contract.runtime):
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_CONTRACT_NONCANONICAL")
    source = SeqaxSiluFusionCorrectnessSourceAuthority.model_validate_json(
        (root / "source.json").read_text()
    )
    bundle = _validate_source_bundle(root, source)
    if (root / "contract.json").read_bytes() != (bundle / _BUNDLED_CONTRACT_PATH).read_bytes():
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_CONTRACT_BUNDLE_MISMATCH")
    design = SeqaxSiluFusionDesignContract.model_validate_json(
        (bundle / _BUNDLED_DESIGN_PATH).read_text()
    )
    pair = SeqaxSiluFusionCompilerPair.model_validate_json(
        (bundle / _BUNDLED_PAIR_PATH).read_text()
    )
    if (root / "compiler_pair.json").read_bytes() != (bundle / _BUNDLED_PAIR_PATH).read_bytes():
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_PAIR_BUNDLE_MISMATCH")
    claim = SeqaxSiluFusionCorrectnessAttemptClaim.model_validate_json(
        (root / "attempt_claim.json").read_text()
    )
    request = SeqaxSiluFusionCorrectnessWorkerRequest.model_validate_json(
        (root / "worker_request.json").read_text()
    )
    registry_claim_path = Path(contract.correctness_claim_registry_root) / (
        f"{contract.correctness_claim_key}-{contract.contract_id}.json"
    )
    registry_info = registry_claim_path.lstat()
    if (
        registry_claim_path.is_symlink()
        or not stat.S_ISREG(registry_info.st_mode)
        or registry_info.st_uid != os.getuid()
        or registry_info.st_nlink != 1
        or registry_info.st_mode & 0o077
        or SeqaxSiluFusionCorrectnessAttemptClaim.model_validate_json(
            registry_claim_path.read_text()
        )
        != claim
    ):
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_REGISTRY_CLAIM_MISMATCH")
    if (
        request.contract != contract
        or request.source != source
        or request.claim != claim
        or claim.contract_id != contract.contract_id
        or (not relocated and claim.output_root != str(root))
        or design.design_id != contract.compiler_design_id
        or pair.pair_id != contract.compiler_pair_id
    ):
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_PROTOCOL_MISMATCH")
    return contract, design, pair, source, claim, request


def _reconstruct_plans(
    root: Path,
    contract: SeqaxSiluFusionCorrectnessContract,
    design: SeqaxSiluFusionDesignContract,
) -> tuple[SeqaxSiluFusionCorrectnessPlan, ...]:
    parameters = dict(contract.parameters)
    parameters["numerical_semantics"] = SeqaxNumericalSemantics(parameters["numerical_semantics"])
    records = []
    for index, expected in enumerate(design.candidates):
        distributed = seqax_forward_schedule(
            **parameters,
            feed_forward_fusion=expected.candidate,
            feed_forward_vector_execution=(SeqaxFeedForwardVectorExecution.PALLAS_FULL_LOCAL),
            residual_norm_strategy=contract.residual_norm_strategy,
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
        observed_static = (
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
        )
        required_static = (
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
        )
        if observed_static != required_static:
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_STATIC_REPLAY_MISMATCH")
        plan_root = root / "plans" / expected.candidate.value
        if (
            (plan_root / "distributed.xdsl").read_text() != canonical_text(distributed)
            or (plan_root / "physical.xdsl").read_text() != canonical_text(physical)
            or (plan_root / "lowered_pallas.py").read_text() != plan.render_executable_source()
            or json.loads((plan_root / "plan_manifest.json").read_text()) != plan.manifest()
        ):
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_PLAN_ARTIFACT_MISMATCH")
        uninstrumented_stablehlo = (plan_root / "uninstrumented_stablehlo.txt").read_text()
        uninstrumented_compiler_hlo = (plan_root / "uninstrumented_compiler_hlo.txt").read_text()
        compiler_candidate = SeqaxSiluFusionCompilerCandidate.model_validate_json(
            (plan_root / "compiler_candidate.json").read_text()
        )
        reachable = analyze_compiler_collectives(
            stablehlo=uninstrumented_stablehlo,
            compiler_hlo=live_seqax_silu_fusion_compiler_hlo(uninstrumented_compiler_hlo),
        )
        fusion = analyze_seqax_silu_fusion_compiler_hlo(
            uninstrumented_compiler_hlo,
            expected.candidate,
            expected_schedule_sha256=expected.physical_schedule_sha256,
        )
        if (
            compiler_candidate.reachable_collectives != reachable
            or compiler_candidate.fusion_analysis != fusion
            or compiler_candidate.semantic_id != contract.candidate_semantic_ids[index]
            or compiler_candidate.pre_optimization_hlo_sha256
            != file_sha256(plan_root / "uninstrumented_pre_optimization_hlo.txt")
        ):
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_COMPILER_REPLAY_MISMATCH")
        record = SeqaxSiluFusionCorrectnessPlan(
            candidate=expected.candidate,
            candidate_semantic_id=compiler_candidate.semantic_id,
            distributed_schedule_sha256=plan.distributed_schedule_sha256,
            physical_schedule_sha256=plan.physical_schedule_sha256,
            pallas_source_sha256=plan.source_sha256(),
            pallas_manifest_sha256=json_sha256(plan.manifest()),
            uninstrumented_stablehlo_sha256=file_sha256(plan_root / "uninstrumented_stablehlo.txt"),
            uninstrumented_pre_optimization_hlo_sha256=file_sha256(
                plan_root / "uninstrumented_pre_optimization_hlo.txt"
            ),
            uninstrumented_compiler_hlo_sha256=file_sha256(
                plan_root / "uninstrumented_compiler_hlo.txt"
            ),
            instrumented_stablehlo_sha256=file_sha256(plan_root / "instrumented_stablehlo.txt"),
            instrumented_pre_optimization_hlo_sha256=file_sha256(
                plan_root / "instrumented_pre_optimization_hlo.txt"
            ),
            instrumented_compiler_hlo_sha256=file_sha256(
                plan_root / "instrumented_compiler_hlo.txt"
            ),
        )
        saved_record = SeqaxSiluFusionCorrectnessPlan.model_validate_json(
            (plan_root / "record.json").read_text()
        )
        if saved_record != record:
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_PLAN_RECORD_MISMATCH")
        records.append(record)
    return tuple(records)


def _numerical_policy(
    contract: SeqaxSiluFusionCorrectnessContract,
) -> SeqaxBf16NumericalPolicy:
    policy = contract.policy
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


def _load_checkpoints(
    root: Path,
    names: tuple[str, ...],
    checkpoint_contract: Any,
) -> tuple[tuple[tuple[np.ndarray, ...], ...], tuple[str, ...]]:
    values = []
    hashes = []
    for name, tensor_contracts in zip(
        names,
        _checkpoint_contracts(checkpoint_contract),
        strict=True,
    ):
        if len(tensor_contracts) != 1:
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_CHECKPOINT_LAYER_MISMATCH")
        stored = _load_array(root / f"{name}.npy")
        hashes.append(array_sha256(stored))
        tensor_contract = tensor_contracts[0]
        if tensor_contract.dtype == "bfloat16":
            logical = decode_seqax_bf16_checkpoint(stored, tensor_contract)
        else:
            logical = stored
            if (
                tensor_contract.dtype != "float32"
                or logical.dtype != np.float32
                or logical.shape != tensor_contract.shape
                or not np.all(np.isfinite(logical))
            ):
                raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_FLOAT32_CHECKPOINT_ABI_MISMATCH")
        values.append((logical,))
    return tuple(values), tuple(hashes)


def _final_output_metrics(assessment: Any) -> SeqaxSiluFusionFinalOutputMetrics:
    return SeqaxSiluFusionFinalOutputMetrics(
        cpu_relative_l2=assessment.cpu_pallas_relative_l2,
        cpu_row_scaled_max=assessment.cpu_pallas_row_scaled_max,
        cpu_top1_match=assessment.pallas_top1_matches_cpu,
        final_output_policy_passed=assessment.final_outputs_satisfy_policy,
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


def _checkpoint_metrics(
    root: Path,
    assessment: Any,
    checkpoints: tuple[tuple[np.ndarray, ...], ...],
) -> SeqaxSiluFusionCheckpointMetrics:
    saved_path = root / "checkpoint_assessment.json"
    if json.loads(saved_path.read_text()) != assessment.model_dump(mode="json"):
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_CHECKPOINT_ASSESSMENT_MISMATCH")
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
        full_assessment_sha256=file_sha256(saved_path),
    )


def _silu_float32(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value, dtype=np.float32)
    sigmoid = np.empty_like(source)
    nonnegative = source >= 0
    sigmoid[nonnegative] = np.float32(1.0) / (np.float32(1.0) + np.exp(-source[nonnegative]))
    exponential = np.exp(source[~nonnegative])
    sigmoid[~nonnegative] = exponential / (np.float32(1.0) + exponential)
    return source * sigmoid


def _verify_boundary(
    root: Path,
    checkpoints: tuple[tuple[tuple[np.ndarray, ...], ...], ...],
) -> int:
    separate, fused = checkpoints
    gate = separate[6][0]
    up = separate[9][0]
    silu = separate[7][0]
    strict = np.asarray(
        silu.astype(np.float32) * up.astype(np.float32),
        dtype=ml_dtypes.bfloat16,
    )
    mutant = np.asarray(
        _silu_float32(gate) * up.astype(np.float32),
        dtype=ml_dtypes.bfloat16,
    )
    difference_count = int(np.count_nonzero(strict != mutant))
    saved_strict = _load_array(root / "strict_hidden.npy")
    saved_mutant = _load_array(root / "mutant_hidden.npy")
    if (
        not np.array_equal(fused[7][0], silu)
        or not np.array_equal(separate[10][0], strict)
        or not np.array_equal(fused[10][0], strict)
        or difference_count <= 0
        or np.array_equal(separate[10][0], mutant)
        or np.array_equal(fused[10][0], mutant)
        or not np.array_equal(saved_strict, strict.view(np.uint16))
        or not np.array_equal(saved_mutant, mutant.view(np.uint16))
    ):
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_BOUNDARY_REPLAY_MISMATCH")
    return difference_count


def _verify_observation(
    root: Path,
    contract: SeqaxSiluFusionCorrectnessContract,
    seed: int,
    policy: SeqaxBf16NumericalPolicy,
    checkpoint_contract: Any,
) -> SeqaxSiluFusionCorrectnessObservation:
    seed_root = root / "seeds" / f"seed-{seed}"
    parameters = checkpoint_contract.parameters.model_dump()
    expected_inputs = tuple(
        np.asarray(value) for value in seqax_forward_inputs(seed=seed, **parameters)
    )
    saved_inputs = tuple(
        _load_array(seed_root / "inputs" / f"{index:02d}.npy") for index in range(13)
    )
    if any(
        not np.array_equal(saved, expected)
        or saved.dtype != expected.dtype
        or saved.shape != expected.shape
        for saved, expected in zip(saved_inputs, expected_inputs, strict=True)
    ):
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_INPUT_REPLAY_MISMATCH")
    fresh_cpu = np.asarray(
        seqax_forward_canonical_reference(
            expected_inputs,
            quantization_decimals=policy.cpu_reference_quantization_decimals,
            **parameters,
        )
    )
    saved_cpu = _load_array(seed_root / "cpu_reference.npy")
    if not np.array_equal(saved_cpu, fresh_cpu):
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_CPU_REPLAY_MISMATCH")

    candidate_records = []
    candidate_outputs = []
    instrumented_outputs = []
    candidate_checkpoints = []
    for candidate in contract.candidates:
        candidate_root = seed_root / candidate.value
        uninstrumented = _load_array(candidate_root / "uninstrumented_output.npy")
        instrumented = _load_array(candidate_root / "instrumented_output.npy")
        checkpoints, checkpoint_hashes = _load_checkpoints(
            candidate_root / "checkpoints",
            contract.checkpoint_names,
            checkpoint_contract,
        )
        uninstrumented_assessment = assess_seqax_bf16_final_outputs(
            uninstrumented,
            uninstrumented,
            fresh_cpu,
            policy=policy,
            expected_shape=contract.output_shape,
            layers=1,
        )
        instrumented_assessment = assess_seqax_bf16_final_outputs(
            instrumented,
            instrumented,
            fresh_cpu,
            policy=policy,
            expected_shape=contract.output_shape,
            layers=1,
        )
        checkpoint_assessment = assess_seqax_bf16_candidate_checkpoints(
            instrumented,
            seed=seed,
            inputs=expected_inputs,
            checkpoints=checkpoints,
            policy=policy,
            contract=checkpoint_contract,
            declared_seeds=(*contract.correctness_seeds, contract.boundary_seed),
        )
        if json.loads(
            (candidate_root / "uninstrumented_assessment.json").read_text()
        ) != uninstrumented_assessment.model_dump(mode="json"):
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_UNINSTRUMENTED_ASSESSMENT_MISMATCH")
        if json.loads(
            (candidate_root / "instrumented_assessment.json").read_text()
        ) != instrumented_assessment.model_dump(mode="json"):
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_INSTRUMENTED_ASSESSMENT_MISMATCH")
        candidate_records.append(
            SeqaxSiluFusionCandidateCorrectness(
                candidate=candidate,
                uninstrumented_output_sha256=array_sha256(uninstrumented),
                instrumented_output_sha256=array_sha256(instrumented),
                checkpoint_sha256=checkpoint_hashes,
                checkpoint_capture_modes=contract.checkpoint_capture_modes[candidate.value],
                uninstrumented_metrics=_final_output_metrics(uninstrumented_assessment),
                instrumented_metrics=_final_output_metrics(instrumented_assessment),
                checkpoint_metrics=_checkpoint_metrics(
                    candidate_root,
                    checkpoint_assessment,
                    checkpoints,
                ),
                instrumentation_output_exact=bool(np.array_equal(uninstrumented, instrumented)),
            )
        )
        candidate_outputs.append(uninstrumented)
        instrumented_outputs.append(instrumented)
        candidate_checkpoints.append(checkpoints)

    outputs_exact = bool(np.array_equal(candidate_outputs[0], candidate_outputs[1]))
    instrumented_exact = bool(np.array_equal(instrumented_outputs[0], instrumented_outputs[1]))
    checkpoints_exact = all(
        np.array_equal(separate, fused)
        for separate_group, fused_group in zip(
            candidate_checkpoints[0],
            candidate_checkpoints[1],
            strict=True,
        )
        for separate, fused in zip(separate_group, fused_group, strict=True)
    )
    if not (outputs_exact and instrumented_exact and checkpoints_exact):
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_CANDIDATE_REPLAY_MISMATCH")
    boundary_case = seed == contract.boundary_seed
    difference_count = (
        _verify_boundary(seed_root / "boundary", tuple(candidate_checkpoints))
        if boundary_case
        else 0
    )
    observed = SeqaxSiluFusionCorrectnessObservation(
        seed=seed,
        input_sha256=tuple(array_sha256(value) for value in saved_inputs),
        cpu_reference_sha256=array_sha256(saved_cpu),
        candidates=tuple(candidate_records),
        candidate_uninstrumented_outputs_exact=outputs_exact,
        candidate_instrumented_outputs_exact=instrumented_exact,
        candidate_checkpoints_exact=checkpoints_exact,
        boundary_case=boundary_case,
        boundary_strict_mutant_difference_count=difference_count,
        boundary_mutant_rejected=boundary_case,
    )
    saved = SeqaxSiluFusionCorrectnessObservation.model_validate_json(
        (seed_root / "observation.json").read_text()
    )
    if saved != observed:
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_OBSERVATION_MISMATCH")
    return observed


def _validate_ledger(
    root: Path,
    claim_id: str,
    *,
    final: bool,
) -> None:
    history = read_ledger_history(root / "ledger.sqlite", claim_id)
    required = (
        RunState.CREATED,
        RunState.VERIFIED,
        RunState.LOWERED,
        RunState.COMPILED,
        RunState.CORRECT,
        *((RunState.VALIDATED, RunState.ACCEPTED) if final else ()),
    )
    observed = tuple(value.state for value in history)
    if observed != required or any(
        state in observed for state in (RunState.TIMED, RunState.TRACED, RunState.COUNTERED)
    ):
        raise ValueError(
            "SEQAX_SILU_FUSION_CORRECTNESS_LEDGER_MISMATCH "
            f"observed={[value.value for value in observed]}"
        )


def verify_correctness(
    root: Path,
    *,
    final: bool,
    relocated: bool = False,
) -> dict[str, str]:
    root = _preflight_root(root)
    contract, design, _pair, source, claim, _request = _validate_protocol(
        root,
        relocated=relocated,
    )
    worker_result = SeqaxSiluFusionCorrectnessWorkerResult.model_validate_json(
        (root / "worker-result.json").read_text()
    )
    plans = _reconstruct_plans(root, contract, design)
    values = dict(contract.parameters)
    values.pop("numerical_semantics")
    checkpoint_contract = seqax_bf16_checkpoint_contract(
        SeqaxBf16ScenarioParameters.model_validate(values)
    )
    policy = _numerical_policy(contract)
    seeds = (*contract.correctness_seeds, contract.boundary_seed)
    observations = tuple(
        _verify_observation(
            root,
            contract,
            seed,
            policy,
            checkpoint_contract,
        )
        for seed in seeds
    )
    result = worker_result.result
    if (
        result.contract_id != contract.contract_id
        or result.claim_id != claim.claim_id
        or result.source != source
        or result.plans != plans
        or result.observations != observations
    ):
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_RESULT_MISMATCH")
    _validate_ledger(root, claim.claim_id, final=final)
    response = {"result_id": result.result_id}
    if final:
        receipt = SeqaxSiluFusionCorrectnessReceipt.model_validate_json(
            (root / "receipt.json").read_text()
        )
        validate_artifact_manifest(
            root,
            receipt.artifacts,
            role_for_path=_artifact_role,
            duplicate_error="SEQAX_SILU_FUSION_CORRECTNESS_ARTIFACT_DUPLICATE",
            closed_world_error="SEQAX_SILU_FUSION_CORRECTNESS_ARTIFACT_SET_MISMATCH",
            mismatch_error=lambda path: (
                f"SEQAX_SILU_FUSION_CORRECTNESS_ARTIFACT_MISMATCH path={path}"
            ),
            symlink_error="SEQAX_SILU_FUSION_CORRECTNESS_ARTIFACT_SYMLINK",
            excluded_paths=("receipt.json",),
        )
        if receipt.result != result:
            raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_RECEIPT_RESULT_MISMATCH")
        response["receipt_id"] = receipt.receipt_id
    elif (root / "receipt.json").exists():
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_PRELIMINARY_RECEIPT_PRESENT")
    return response


def verify_failure(
    root: Path,
    *,
    relocated: bool = False,
) -> dict[str, str]:
    root = _preflight_root(root)
    contract, _design, _pair, source, claim, _request = _validate_protocol(
        root,
        relocated=relocated,
    )
    receipt = SeqaxSiluFusionCorrectnessFailureReceipt.model_validate_json(
        (root / "failure-receipt.json").read_text()
    )
    incomplete_receipt_path = root / "receipt.json"
    incomplete_receipt = (
        SeqaxSiluFusionCorrectnessReceipt.model_validate_json(incomplete_receipt_path.read_text())
        if incomplete_receipt_path.exists()
        else None
    )
    failure = SeqaxSiluFusionCorrectnessFailure.model_validate_json(
        (root / "worker-failure.json").read_text()
    )
    validate_artifact_manifest(
        root,
        receipt.artifacts,
        role_for_path=_artifact_role,
        duplicate_error="SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_ARTIFACT_DUPLICATE",
        closed_world_error="SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_ARTIFACT_SET_MISMATCH",
        mismatch_error=lambda path: (
            f"SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_ARTIFACT_MISMATCH path={path}"
        ),
        symlink_error="SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_ARTIFACT_SYMLINK",
        excluded_paths=("failure-receipt.json", "receipt.json"),
    )
    history = read_ledger_history(root / "ledger.sqlite", claim.claim_id)
    observed = tuple(value.state for value in history)
    success_prefix = (
        RunState.CREATED,
        RunState.VERIFIED,
        RunState.LOWERED,
        RunState.COMPILED,
        RunState.CORRECT,
        RunState.VALIDATED,
        RunState.ACCEPTED,
    )
    if not observed or observed != success_prefix[: len(observed)]:
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_LEDGER_MISMATCH")
    state = observed[-1]
    required_execution_status = (
        "executed"
        if state in {RunState.CORRECT, RunState.VALIDATED, RunState.ACCEPTED}
        else "may-have-executed"
        if state is RunState.COMPILED
        else "not-executed"
    )
    if (
        receipt.claim != claim
        or receipt.source != source
        or receipt.failure != failure
        or receipt.incomplete_success_receipt != incomplete_receipt
        or failure.final_ledger_state is not state
        or failure.model_outputs_execution_status != required_execution_status
        or receipt.claim.contract_id != contract.contract_id
    ):
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_FAILURE_RECEIPT_MISMATCH")
    return {"failure_receipt_id": receipt.failure_receipt_id}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--failed", action="store_true")
    parser.add_argument("--relocated", action="store_true")
    arguments = parser.parse_args()
    if arguments.failed and arguments.final:
        raise ValueError("SEQAX_SILU_FUSION_CORRECTNESS_VERIFY_MODE_CONFLICT")
    if arguments.failed:
        print(
            json.dumps(
                verify_failure(arguments.root, relocated=arguments.relocated),
                sort_keys=True,
            )
        )
        return
    print(
        json.dumps(
            verify_correctness(
                arguments.root,
                final=arguments.final,
                relocated=arguments.relocated,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
