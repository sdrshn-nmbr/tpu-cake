from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
from jax.sharding import PartitionSpec as P

from tpu_cake.artifacts import build_artifact_manifest, file_sha256
from tpu_cake.canonical import canonical_text
from tpu_cake.contracts import ArtifactReference, ArtifactRole, SourceFileContract
from tpu_cake.identity import array_sha256, arrays_sha256, semantic_sha256
from tpu_cake.ledger import EvidenceRun, RunState, payload_sha256, read_ledger_history
from tpu_cake.rpa_donation_confirmation import (
    INKLING_RPA_DONATION_CONFIRMATION_RECEIPT_SCHEMA,
    INKLING_RPA_DONATION_CONFIRMATION_SCHEMA,
    InklingRpaDonationArm,
    InklingRpaDonationConfirmationContract,
    InklingRpaDonationConfirmationReceipt,
    InklingRpaDonationConfirmationResult,
    InklingRpaDonationCorrectnessObservation,
    InklingRpaDonationHloCapture,
    InklingRpaDonationHloCaptureResult,
    InklingRpaDonationRunIdentity,
    InklingRpaDonationState,
    InklingRpaDonationTimingRound,
    default_inkling_rpa_donation_confirmation_contract,
    donation_confirmation_orders,
    donation_confirmation_statistics,
)
from tpu_cake.rpa_lowering import ShardedFusedRpaPlan, lower_inkling_sharded_rpa_to_pallas
from tpu_cake.rpa_surface_runner import (
    _QUERY_CACHE_ALIAS,
    _chain_query_and_cache,
    _device_inventory,
    _errors,
    _exclusive_lock,
    _git_output,
    _load_bf16,
    _rename_directory_noreplace,
    _repository_root,
    _require_backend_runtime,
    _require_backend_source,
    _require_clean_repository,
    _require_compilation_root,
    _require_safe_output_root,
    _save_bf16,
    _sha256,
    _source_state,
    _validate_output_abi,
    _write_json,
    _write_json_atomic,
    _write_text,
)
from tpu_cake.rpa_surface_runner import (
    _source_manifest as _surface_source_manifest,
)
from tpu_cake.runner import _runtime_identity
from tpu_cake.workloads.inkling_rpa import (
    inkling_sharded_fused_rpa_inputs,
    inkling_sharded_fused_rpa_reference,
    inkling_sharded_fused_rpa_schedule,
)


@dataclass(frozen=True)
class _CompiledArm:
    arm: InklingRpaDonationArm
    plan: ShardedFusedRpaPlan
    mesh: Any
    executable: Callable[..., tuple[Any, Any]]
    stablehlo: str
    compiler_hlo: str
    evidence: InklingRpaDonationHloCapture


def _build_executable(
    plan: ShardedFusedRpaPlan,
    kernel: Callable[..., tuple[Any, Any]],
    devices: tuple[Any, ...],
    donate_argnums: tuple[int, ...],
) -> tuple[Any, Callable[..., tuple[Any, Any]]]:
    mesh = plan.mesh(devices)
    plan.local_plan.validate_backend_callable(kernel)

    def local(*inputs: Any) -> tuple[Any, Any]:
        return plan.local_plan.invoke(
            kernel,
            *inputs,
            backend_manifest=plan.backend_manifest,
            device_kind="TPU7x",
        )

    sharded = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=tuple(P(*spec) for spec in plan.input_partition_specs),
        out_specs=tuple(P(*spec) for spec in plan.output_partition_specs),
    )
    return mesh, jax.jit(sharded, donate_argnums=donate_argnums)


def _compile_arm(
    contract: InklingRpaDonationConfirmationContract,
    arm: InklingRpaDonationArm,
    kernel: Callable[..., tuple[Any, Any]],
    plan: ShardedFusedRpaPlan,
    devices: tuple[Any, ...],
) -> _CompiledArm:
    arm_contract = next(value for value in contract.arms if value.arm is arm)
    mesh, executable = _build_executable(
        plan,
        kernel,
        devices,
        arm_contract.external_donate_argnums,
    )
    host_inputs = inkling_sharded_fused_rpa_inputs(contract.correctness_seeds[0])
    placed = plan.place_inputs(host_inputs, mesh=mesh)
    lowered = executable.lower(*placed)
    stablehlo = str(lowered.compiler_ir("stablehlo")).rstrip("\n") + "\n"
    compiled = lowered.compile()
    compiler_hlo = compiled.as_text().rstrip("\n") + "\n"
    if "tpu_custom_call" not in compiler_hlo or "RPAd" not in compiler_hlo:
        raise ValueError(f"INKLING_RPA_DONATION_COMPILER_HLO_MARKERS_MISSING arm={arm}")
    _validate_stablehlo_aliases(stablehlo, arm)
    _validate_compiler_hlo_aliases(compiler_hlo, arm)
    evidence = InklingRpaDonationHloCapture(
        arm=arm,
        stablehlo_sha256=hashlib.sha256(stablehlo.encode()).hexdigest(),
        compiler_hlo_sha256=hashlib.sha256(compiler_hlo.encode()).hexdigest(),
        compiler_hlo_alias_contract=arm_contract.compiler_hlo_alias_contract,
    )
    if evidence.stablehlo_sha256 != arm_contract.stablehlo_sha256:
        raise ValueError(f"INKLING_RPA_DONATION_STABLEHLO_IDENTITY_MISMATCH arm={arm}")
    return _CompiledArm(
        arm=arm,
        plan=plan,
        mesh=mesh,
        executable=compiled,
        stablehlo=stablehlo,
        compiler_hlo=compiler_hlo,
        evidence=evidence,
    )


def _save_compiled_arm(root: Path, compiled: _CompiledArm) -> None:
    arm_root = root / "arms" / compiled.arm.value
    _write_text(arm_root / "stablehlo.txt", compiled.stablehlo)
    _write_text(arm_root / "compiler_hlo.txt", compiled.compiler_hlo)


def _validate_stablehlo_aliases(stablehlo: str, arm: InklingRpaDonationArm) -> None:
    main_lines = tuple(line for line in stablehlo.splitlines() if "func.func public @main(" in line)
    if len(main_lines) != 1:
        raise ValueError(f"INKLING_RPA_DONATION_STABLEHLO_MAIN_MISMATCH arm={arm}")
    main = main_lines[0]
    aliases = tuple(
        (int(argument), int(output))
        for argument, output in re.findall(
            r"%arg(\d+):[^%]*?tf\.aliasing_output = (\d+) : i32",
            main,
        )
    )
    if main.count("tf.aliasing_output") != len(aliases):
        raise ValueError(f"INKLING_RPA_DONATION_STABLEHLO_ALIAS_PARSE_FAILED arm={arm}")
    expected = () if arm is InklingRpaDonationArm.NON_DONATING else ((0, 0), (3, 1))
    if aliases != expected:
        raise ValueError(
            f"INKLING_RPA_DONATION_STABLEHLO_ALIAS_MISMATCH arm={arm} observed={aliases}"
        )


def _validate_compiler_hlo_aliases(compiler_hlo: str, arm: InklingRpaDonationArm) -> None:
    header = compiler_hlo.splitlines()[0]
    marker = "input_output_alias={"
    expected = "{0}: (0, {}, may-alias), {1}: (3, {}, may-alias)"
    if arm is InklingRpaDonationArm.NON_DONATING:
        if marker in header:
            raise ValueError("INKLING_RPA_DONATION_BASELINE_COMPILER_ALIAS_PRESENT")
        return
    if marker not in header or "}, entry_computation_layout=" not in header:
        raise ValueError("INKLING_RPA_DONATION_CANDIDATE_COMPILER_ALIAS_MISSING")
    aliases = header.split(marker, maxsplit=1)[1].split("}, entry_computation_layout=", maxsplit=1)[
        0
    ]
    normalized = re.sub(r"\s+", " ", aliases).strip()
    if normalized != expected or _QUERY_CACHE_ALIAS not in header:
        raise ValueError(
            f"INKLING_RPA_DONATION_CANDIDATE_COMPILER_ALIAS_MISMATCH observed={normalized}"
        )


def _source_manifest() -> tuple[SourceFileContract, ...]:
    repository = _repository_root()
    additions = (
        "src/tpu_cake/cli.py",
        "src/tpu_cake/rpa_device_main.py",
        "src/tpu_cake/rpa_donation_confirmation.py",
        "src/tpu_cake/rpa_donation_confirmation_runner.py",
    )
    manifest = {value.path: value for value in _surface_source_manifest()}
    for path in additions:
        manifest[path] = SourceFileContract(
            path=path,
            sha256=file_sha256(repository / path),
        )
    return tuple(manifest[path] for path in sorted(manifest))


def capture_inkling_rpa_donation_hlo_identities(
    output_root: Path,
    contract: InklingRpaDonationConfirmationContract,
    kernel: Callable[..., tuple[Any, Any]],
) -> InklingRpaDonationHloCaptureResult:
    canonical = default_inkling_rpa_donation_confirmation_contract()
    if contract != canonical:
        raise ValueError("INKLING_RPA_DONATION_EXTERNAL_CONTRACT_MISMATCH")
    if contract.hlo_identity_status != "pending":
        raise ValueError("INKLING_RPA_DONATION_HLO_IDENTITIES_ALREADY_PINNED")
    output_root = output_root.absolute()
    repository = _repository_root()
    _require_compilation_root(repository, contract)
    _require_safe_output_root(output_root)
    _require_clean_repository(repository)
    runtime = _runtime_identity()
    if runtime != contract.runtime:
        raise ValueError("INKLING_RPA_DONATION_RUNTIME_MISMATCH")
    if (platform.system(), platform.machine()) != (
        contract.producer_system,
        contract.producer_machine,
    ):
        raise ValueError("INKLING_RPA_DONATION_PRODUCER_HOST_MISMATCH")
    _require_backend_source(contract)
    _require_backend_runtime(contract)
    if output_root.exists():
        raise ValueError(f"INKLING_RPA_DONATION_CAPTURE_EXISTS path={output_root}")
    devices = tuple(jax.devices())
    device_inventory = _device_inventory(devices)
    plan = lower_inkling_sharded_rpa_to_pallas(inkling_sharded_fused_rpa_schedule())
    if plan.source_sha256() != contract.plan.execution_sha256:
        raise ValueError("INKLING_RPA_DONATION_PLAN_IDENTITY_MISMATCH")
    output_root.mkdir(parents=True)
    compiled_arms = tuple(
        _compile_arm(contract, arm, kernel, plan, devices)
        for arm in (InklingRpaDonationArm.NON_DONATING, InklingRpaDonationArm.DONATING)
    )
    for compiled in compiled_arms:
        _save_compiled_arm(output_root, compiled)
    captures = tuple(value.evidence for value in compiled_arms)
    source_state = _source_state(repository)
    result = InklingRpaDonationHloCaptureResult(
        confirmation_id=contract.confirmation_id,
        source_commit=source_state["git_commit"],
        uv_lock_sha256=source_state["uv_lock_sha256"],
        source_manifest=_source_manifest(),
        runtime=runtime,
        devices=device_inventory,
        captures=captures,
    )
    (output_root / "capture.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    return result


def _execute(
    compiled: _CompiledArm,
    host_inputs: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    placed = compiled.plan.place_inputs(host_inputs, mesh=compiled.mesh)
    outputs = compiled.executable(*placed)
    jax.block_until_ready(outputs)
    return tuple(np.asarray(value) for value in outputs)


def _correctness_observations(
    root: Path,
    contract: InklingRpaDonationConfirmationContract,
    compiled_arms: tuple[_CompiledArm, _CompiledArm],
) -> tuple[InklingRpaDonationCorrectnessObservation, ...]:
    observations = []
    for seed in contract.correctness_seeds:
        host_inputs = inkling_sharded_fused_rpa_inputs(seed)
        oracle_output, oracle_cache = inkling_sharded_fused_rpa_reference(host_inputs)
        seed_root = root / "correctness" / f"seed-{seed}"
        _save_bf16(seed_root / "oracle_output.npy", oracle_output)
        executions = {}
        for compiled in compiled_arms:
            first = _execute(compiled, host_inputs)
            second = _execute(compiled, host_inputs)
            _validate_output_abi(contract, *first)
            _validate_output_abi(contract, *second)
            executions[compiled.arm] = (first, second)
            arm_root = seed_root / compiled.arm.value
            _save_bf16(arm_root / "output.npy", first[0])
            _save_bf16(arm_root / "repeat_output.npy", second[0])
            _save_bf16(arm_root / "cache.npy", first[1])
            _save_bf16(arm_root / "repeat_cache.npy", second[1])
        baseline = executions[InklingRpaDonationArm.NON_DONATING][0]
        candidate = executions[InklingRpaDonationArm.DONATING][0]
        cross_output_exact = np.array_equal(baseline[0], candidate[0])
        cross_cache_exact = np.array_equal(baseline[1], candidate[1])
        for compiled in compiled_arms:
            first, second = executions[compiled.arm]
            maximum, relative_l2 = _errors(first[0], oracle_output)
            repeated_output_exact = np.array_equal(first[0], second[0])
            repeated_cache_exact = np.array_equal(first[1], second[1])
            passed = (
                repeated_output_exact
                and repeated_cache_exact
                and cross_output_exact
                and cross_cache_exact
                and np.array_equal(first[1], oracle_cache)
                and maximum <= contract.output_maximum_absolute_error
                and relative_l2 <= contract.output_relative_l2_error
            )
            observation = InklingRpaDonationCorrectnessObservation(
                arm=compiled.arm,
                seed=seed,
                input_sha256=arrays_sha256(host_inputs),
                output_sha256=array_sha256(first[0]),
                repeat_output_sha256=array_sha256(second[0]),
                oracle_output_sha256=array_sha256(oracle_output),
                cache_sha256=array_sha256(first[1]),
                repeat_cache_sha256=array_sha256(second[1]),
                oracle_cache_sha256=array_sha256(oracle_cache),
                repeated_output_exact=repeated_output_exact,
                repeated_cache_exact=repeated_cache_exact,
                cross_arm_output_exact=cross_output_exact,
                cross_arm_cache_exact=cross_cache_exact,
                maximum_absolute_error=maximum,
                relative_l2_error=relative_l2,
                passed=passed,
            )
            _write_json(
                seed_root / compiled.arm.value / "observation.json",
                observation,
            )
            if not passed:
                raise ValueError(
                    f"INKLING_RPA_DONATION_CORRECTNESS_FAILED arm={compiled.arm} seed={seed}"
                )
            observations.append(observation)
    return tuple(
        next(value for value in observations if value.arm is arm and value.seed == seed)
        for arm in (InklingRpaDonationArm.NON_DONATING, InklingRpaDonationArm.DONATING)
        for seed in contract.correctness_seeds
    )


def _state(
    root: Path,
    label: str,
    compiled: _CompiledArm,
    host_inputs: tuple[np.ndarray, ...],
    *,
    executions: int = 1,
) -> InklingRpaDonationState:
    current = compiled.plan.place_inputs(host_inputs, mesh=compiled.mesh)
    jax.block_until_ready(current)
    outputs = None
    for _ in range(executions):
        outputs = compiled.executable(*current)
        jax.block_until_ready(outputs)
        current = _chain_query_and_cache(current, outputs)
    if outputs is None:
        raise ValueError("INKLING_RPA_DONATION_STATE_EXECUTIONS_EMPTY")
    output, cache = tuple(np.asarray(value) for value in outputs)
    _save_bf16(root / "timing" / compiled.arm.value / f"{label}_output.npy", output)
    _save_bf16(root / "timing" / compiled.arm.value / f"{label}_cache.npy", cache)
    return InklingRpaDonationState(
        arm=compiled.arm,
        output_sha256=array_sha256(output),
        cache_sha256=array_sha256(cache),
    )


def _timing_rounds(
    contract: InklingRpaDonationConfirmationContract,
    compiled_arms: tuple[_CompiledArm, _CompiledArm],
    host_inputs: tuple[np.ndarray, ...],
) -> tuple[InklingRpaDonationTimingRound, ...]:
    by_arm = {value.arm: value for value in compiled_arms}
    for compiled in compiled_arms:
        for _ in range(contract.warmup_blocks):
            current = compiled.plan.place_inputs(host_inputs, mesh=compiled.mesh)
            jax.block_until_ready(current)
            for _ in range(contract.calls_per_block):
                outputs = compiled.executable(*current)
                jax.block_until_ready(outputs)
                current = _chain_query_and_cache(current, outputs)
    rounds = []
    for round_index, order in enumerate(donation_confirmation_orders(contract)):
        for position, arm in enumerate(order):
            compiled = by_arm[arm]
            current = compiled.plan.place_inputs(host_inputs, mesh=compiled.mesh)
            jax.block_until_ready(current)
            samples = []
            outputs = None
            for _ in range(contract.calls_per_block):
                started = time.perf_counter_ns()
                outputs = compiled.executable(*current)
                jax.block_until_ready(outputs)
                samples.append(time.perf_counter_ns() - started)
                current = _chain_query_and_cache(current, outputs)
            if outputs is None:
                raise AssertionError("RPA donation timing block executed no calls")
            terminal_output, terminal_cache = tuple(np.asarray(value) for value in outputs)
            rounds.append(
                InklingRpaDonationTimingRound(
                    round_index=round_index,
                    position=position,
                    arm=arm,
                    samples_ns=tuple(samples),
                    median_ns=float(statistics.median(samples)),
                    terminal_output_sha256=array_sha256(terminal_output),
                    terminal_cache_sha256=array_sha256(terminal_cache),
                )
            )
    return tuple(rounds)


def _artifact_role(path: Path) -> ArtifactRole:
    relative = path.as_posix()
    if relative == "contract.json":
        return ArtifactRole.SEARCH_CONTRACT
    if relative == "run_identity.json":
        return ArtifactRole.INVOCATION
    if relative == "source_state.json":
        return ArtifactRole.SOURCE_STATE
    if relative == "source_diff.patch":
        return ArtifactRole.SOURCE_DIFF
    if relative == "source_manifest.json":
        return ArtifactRole.BACKEND_MANIFEST
    if relative == "physical.xdsl":
        return ArtifactRole.PHYSICAL_IR
    if relative == "plan.json":
        return ArtifactRole.PLAN_MANIFEST
    if relative.endswith("/stablehlo.txt"):
        return ArtifactRole.STABLEHLO
    if relative.endswith("/compiler_hlo.txt"):
        return ArtifactRole.COMPILER_HLO
    if relative == "ledger.sqlite":
        return ArtifactRole.EXECUTION_LEDGER
    if relative == "correctness.json" or relative.endswith("/observation.json"):
        return ArtifactRole.PROFILE_ASSESSMENT
    if relative.endswith("/oracle_output.npy"):
        return ArtifactRole.ORACLE_OUTPUT
    if relative.endswith(".npy"):
        return ArtifactRole.CORRECTNESS_OUTPUT
    if relative in {"rounds.json", "statistics.json"}:
        return ArtifactRole.TIMING_SAMPLES
    if relative == "result.json":
        return ArtifactRole.SEARCH_RESULT
    raise ValueError(f"INKLING_RPA_DONATION_UNKNOWN_ARTIFACT path={relative}")


def _artifacts(root: Path) -> tuple[ArtifactReference, ...]:
    return build_artifact_manifest(
        root,
        role_for_path=_artifact_role,
        excluded_paths=(),
        exclude_path=lambda path: path.name == "receipt.json",
    )


def _ledger_payload(
    contract: InklingRpaDonationConfirmationContract,
    run_id: str,
    source_state: dict[str, Any],
    root: Path,
    state: RunState,
) -> dict[str, Any]:
    if state is RunState.CREATED:
        return {"confirmation_id": contract.confirmation_id, "run_id": run_id}
    if state is RunState.VERIFIED:
        return {
            "source_commit": source_state["git_commit"],
            "arm_execution_sha256": [value.execution_sha256 for value in contract.arms],
        }
    if state is RunState.LOWERED:
        return {
            "physical_sha256": _sha256(root / "physical.xdsl"),
            "plan_sha256": _sha256(root / "plan.json"),
        }
    if state is RunState.COMPILED:
        return {
            value.arm.value: {
                "stablehlo_sha256": _sha256(root / "arms" / value.arm.value / "stablehlo.txt"),
                "compiler_hlo_sha256": _sha256(
                    root / "arms" / value.arm.value / "compiler_hlo.txt"
                ),
            }
            for value in contract.arms
        }
    if state is RunState.CORRECT:
        return {"correctness_sha256": _sha256(root / "correctness.json")}
    if state is RunState.TIMED:
        return {
            "rounds_sha256": _sha256(root / "rounds.json"),
            "statistics_sha256": _sha256(root / "statistics.json"),
        }
    if state is RunState.ACCEPTED:
        result = InklingRpaDonationConfirmationResult.model_validate_json(
            (root / "result.json").read_text()
        )
        return {
            "result_sha256": _sha256(root / "result.json"),
            "winner": result.winner,
            "accepted": result.accepted,
        }
    raise ValueError(f"INKLING_RPA_DONATION_LEDGER_STATE_UNSUPPORTED state={state}")


def _validate_source(root: Path, result: InklingRpaDonationConfirmationResult) -> None:
    repository = _repository_root()
    state = json.loads((root / "source_state.json").read_text())
    manifest = tuple(
        SourceFileContract.model_validate_json(json.dumps(value))
        for value in json.loads((root / "source_manifest.json").read_text())
    )
    if (
        state["git_dirty"] is not False
        or state["git_status"] != []
        or state["source_diff_sha256"] != hashlib.sha256(b"").hexdigest()
        or (root / "source_diff.patch").read_bytes() != b""
        or state["git_commit"] != result.source_commit
        or state["uv_lock_sha256"] != result.uv_lock_sha256
        or _sha256(root / "source_state.json") != result.source_state_sha256
        or _sha256(root / "source_manifest.json") != result.source_manifest_sha256
        or manifest != result.source_manifest
        or manifest != _source_manifest()
    ):
        raise ValueError("INKLING_RPA_DONATION_SOURCE_EVIDENCE_MISMATCH")
    if _git_output(repository, "rev-parse", "HEAD") != result.source_commit:
        raise ValueError("INKLING_RPA_DONATION_VERIFIER_COMMIT_MISMATCH")
    if _sha256(repository / "uv.lock") != result.uv_lock_sha256:
        raise ValueError("INKLING_RPA_DONATION_VERIFIER_LOCK_MISMATCH")
    for source in manifest:
        blob = subprocess.run(
            ["git", "show", f"{result.source_commit}:{source.path}"],
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
        if (
            _sha256(repository / source.path) != source.sha256
            or hashlib.sha256(blob).hexdigest() != source.sha256
        ):
            raise ValueError(f"INKLING_RPA_DONATION_SOURCE_BLOB_MISMATCH path={source.path}")


def _validate_correctness(
    root: Path,
    contract: InklingRpaDonationConfirmationContract,
    saved: tuple[InklingRpaDonationCorrectnessObservation, ...],
) -> None:
    saved_file = tuple(
        InklingRpaDonationCorrectnessObservation.model_validate_json(json.dumps(value))
        for value in json.loads((root / "correctness.json").read_text())
    )
    if saved_file != saved:
        raise ValueError("INKLING_RPA_DONATION_CORRECTNESS_FILE_MISMATCH")
    replayed = []
    for seed in contract.correctness_seeds:
        host_inputs = inkling_sharded_fused_rpa_inputs(seed)
        fresh_oracle, oracle_cache = inkling_sharded_fused_rpa_reference(host_inputs)
        seed_root = root / "correctness" / f"seed-{seed}"
        saved_oracle = _load_bf16(seed_root / "oracle_output.npy")
        replay_maximum, replay_relative_l2 = _errors(saved_oracle, fresh_oracle)
        states = {}
        for arm in (InklingRpaDonationArm.NON_DONATING, InklingRpaDonationArm.DONATING):
            arm_root = seed_root / arm.value
            output = _load_bf16(arm_root / "output.npy")
            repeat_output = _load_bf16(arm_root / "repeat_output.npy")
            cache = _load_bf16(arm_root / "cache.npy")
            repeat_cache = _load_bf16(arm_root / "repeat_cache.npy")
            _validate_output_abi(contract, output, cache)
            _validate_output_abi(contract, repeat_output, repeat_cache)
            maximum, relative_l2 = _errors(output, saved_oracle)
            fresh_maximum, fresh_relative_l2 = _errors(output, fresh_oracle)
            observation = InklingRpaDonationCorrectnessObservation.model_validate_json(
                (arm_root / "observation.json").read_text()
            )
            states[arm] = (output, cache)
            replayed.append(
                observation.model_copy(
                    update={
                        "input_sha256": arrays_sha256(host_inputs),
                        "output_sha256": array_sha256(output),
                        "repeat_output_sha256": array_sha256(repeat_output),
                        "oracle_output_sha256": array_sha256(saved_oracle),
                        "cache_sha256": array_sha256(cache),
                        "repeat_cache_sha256": array_sha256(repeat_cache),
                        "oracle_cache_sha256": array_sha256(oracle_cache),
                        "repeated_output_exact": np.array_equal(output, repeat_output),
                        "repeated_cache_exact": np.array_equal(cache, repeat_cache),
                        "maximum_absolute_error": maximum,
                        "relative_l2_error": relative_l2,
                    }
                )
            )
        cross_output = np.array_equal(states[contract.baseline][0], states[contract.candidate][0])
        cross_cache = np.array_equal(states[contract.baseline][1], states[contract.candidate][1])
        for index in (-2, -1):
            observation = replayed[index]
            replayed[index] = observation.model_copy(
                update={
                    "cross_arm_output_exact": cross_output,
                    "cross_arm_cache_exact": cross_cache,
                    "passed": (
                        observation.repeated_output_exact
                        and observation.repeated_cache_exact
                        and cross_output
                        and cross_cache
                        and observation.cache_sha256 == observation.oracle_cache_sha256
                        and observation.maximum_absolute_error
                        <= contract.output_maximum_absolute_error
                        and observation.relative_l2_error <= contract.output_relative_l2_error
                        and replay_maximum <= contract.output_maximum_absolute_error
                        and replay_relative_l2 <= contract.output_relative_l2_error
                        and fresh_maximum <= contract.output_maximum_absolute_error
                        and fresh_relative_l2 <= contract.output_relative_l2_error
                    ),
                }
            )
    ordered = tuple(
        next(value for value in replayed if value.arm is arm and value.seed == seed)
        for arm in (InklingRpaDonationArm.NON_DONATING, InklingRpaDonationArm.DONATING)
        for seed in contract.correctness_seeds
    )
    if ordered != saved or any(not value.passed for value in ordered):
        raise ValueError("INKLING_RPA_DONATION_CORRECTNESS_REPLAY_MISMATCH")


def _validate_timing_states(
    root: Path,
    contract: InklingRpaDonationConfirmationContract,
    result: InklingRpaDonationConfirmationResult,
) -> None:
    inputs = inkling_sharded_fused_rpa_inputs(contract.timing_seed)
    one_call_oracle = inkling_sharded_fused_rpa_reference(inputs)
    current = inputs
    terminal_oracle = None
    for _ in range(contract.calls_per_block):
        terminal_oracle = inkling_sharded_fused_rpa_reference(current)
        current = _chain_query_and_cache(current, terminal_oracle)
    if terminal_oracle is None:
        raise ValueError("INKLING_RPA_DONATION_REFERENCE_CHAIN_EMPTY")
    if result.timing_input_sha256 != arrays_sha256(inputs):
        raise ValueError("INKLING_RPA_DONATION_TIMING_INPUT_MISMATCH")
    loaded = {"pre": {}, "expected_terminal": {}, "post": {}}
    for label, expected_states in (
        ("pre", result.pre_timing_states),
        ("expected_terminal", result.expected_terminal_states),
        ("post", result.post_timing_states),
    ):
        oracle_output, oracle_cache = (
            terminal_oracle if label == "expected_terminal" else one_call_oracle
        )
        for expected in expected_states:
            arm_root = root / "timing" / expected.arm.value
            output = _load_bf16(arm_root / f"{label}_output.npy")
            cache = _load_bf16(arm_root / f"{label}_cache.npy")
            _validate_output_abi(contract, output, cache)
            maximum, relative_l2 = _errors(output, oracle_output)
            if (
                expected.output_sha256 != array_sha256(output)
                or expected.cache_sha256 != array_sha256(cache)
                or not np.array_equal(cache, oracle_cache)
                or maximum > contract.output_maximum_absolute_error
                or relative_l2 > contract.output_relative_l2_error
            ):
                raise ValueError(
                    f"INKLING_RPA_DONATION_TIMING_STATE_MISMATCH arm={expected.arm} label={label}"
                )
            loaded[label][expected.arm] = (output, cache)
    for label in ("pre", "expected_terminal", "post"):
        if not all(
            np.array_equal(left, right)
            for left, right in zip(
                loaded[label][contract.baseline],
                loaded[label][contract.candidate],
                strict=True,
            )
        ):
            raise ValueError(f"INKLING_RPA_DONATION_CROSS_ARM_TIMING_MISMATCH label={label}")
    for arm in (contract.baseline, contract.candidate):
        if not all(
            np.array_equal(left, right)
            for left, right in zip(loaded["pre"][arm], loaded["post"][arm], strict=True)
        ):
            raise ValueError(f"INKLING_RPA_DONATION_PRE_POST_MISMATCH arm={arm}")


def validate_inkling_rpa_donation_confirmation(
    root: Path,
    expected_contract: InklingRpaDonationConfirmationContract,
    *,
    require_accepted: bool = True,
    require_receipt: bool = True,
) -> InklingRpaDonationConfirmationResult:
    canonical = default_inkling_rpa_donation_confirmation_contract()
    if expected_contract != canonical or expected_contract.hlo_identity_status != "pinned":
        raise ValueError("INKLING_RPA_DONATION_EXTERNAL_CONTRACT_MISMATCH")
    if root.is_symlink():
        raise ValueError(f"INKLING_RPA_DONATION_ROOT_INVALID path={root}")
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"INKLING_RPA_DONATION_ROOT_INVALID path={root}")
    for path in root.rglob("*"):
        if path.is_symlink() or (path.is_file() and path.stat().st_nlink != 1):
            raise ValueError(f"INKLING_RPA_DONATION_LINK_INVALID path={path}")
    contract = InklingRpaDonationConfirmationContract.model_validate_json(
        (root / "contract.json").read_text()
    )
    if contract != expected_contract:
        raise ValueError("INKLING_RPA_DONATION_SAVED_CONTRACT_MISMATCH")
    result = InklingRpaDonationConfirmationResult.model_validate_json(
        (root / "result.json").read_text()
    )
    identity = InklingRpaDonationRunIdentity.model_validate_json(
        (root / "run_identity.json").read_text()
    )
    expected_run_id = semantic_sha256(
        INKLING_RPA_DONATION_CONFIRMATION_SCHEMA,
        contract.confirmation_id,
        result.source_commit,
    )
    if identity != InklingRpaDonationRunIdentity(
        schema_version=INKLING_RPA_DONATION_CONFIRMATION_SCHEMA,
        confirmation_id=contract.confirmation_id,
        run_id=expected_run_id,
        source_commit=result.source_commit,
    ) or (result.confirmation_id, result.run_id) != (contract.confirmation_id, expected_run_id):
        raise ValueError("INKLING_RPA_DONATION_RUN_IDENTITY_MISMATCH")
    if (
        result.source_surface_id != contract.source_surface_id
        or result.source_surface_receipt_sha256 != contract.source_surface_receipt_sha256
        or result.runtime != contract.runtime
        or (result.producer_system, result.producer_machine)
        != (contract.producer_system, contract.producer_machine)
        or result.plan != contract.plan
        or result.devices
        != tuple(
            type(result.devices[0])(
                id=index,
                process_index=0,
                platform="tpu",
                device_kind="TPU7x",
            )
            for index in range(8)
        )
    ):
        raise ValueError("INKLING_RPA_DONATION_RESULT_CONTRACT_MISMATCH")
    _validate_source(root, result)
    if canonical_text(inkling_sharded_fused_rpa_schedule()) != (
        root / "physical.xdsl"
    ).read_text() or json.loads((root / "plan.json").read_text()) != contract.plan.model_dump(
        mode="json"
    ):
        raise ValueError("INKLING_RPA_DONATION_PLAN_REPLAY_MISMATCH")
    for arm_contract, compiled in zip(contract.arms, result.compiled_arms, strict=True):
        arm_root = root / "arms" / arm_contract.arm.value
        stablehlo = (arm_root / "stablehlo.txt").read_text()
        compiler_hlo = (arm_root / "compiler_hlo.txt").read_text()
        if (
            compiled.arm is not arm_contract.arm
            or compiled.stablehlo_sha256 != arm_contract.stablehlo_sha256
            or compiled.stablehlo_sha256 != _sha256(arm_root / "stablehlo.txt")
            or compiled.compiler_hlo_sha256 != _sha256(arm_root / "compiler_hlo.txt")
            or compiled.compiler_hlo_alias_contract != arm_contract.compiler_hlo_alias_contract
        ):
            raise ValueError(f"INKLING_RPA_DONATION_HLO_REPLAY_MISMATCH arm={arm_contract.arm}")
        _validate_stablehlo_aliases(stablehlo, arm_contract.arm)
        _validate_compiler_hlo_aliases(compiler_hlo, arm_contract.arm)
    _validate_correctness(root, contract, result.correctness)
    _validate_timing_states(root, contract, result)
    rounds = tuple(
        InklingRpaDonationTimingRound.model_validate_json(json.dumps(value))
        for value in json.loads((root / "rounds.json").read_text())
    )
    statistics_record = donation_confirmation_statistics(
        contract,
        rounds,
        result.expected_terminal_states,
    )
    if (
        rounds != result.rounds
        or statistics_record != result.statistics
        or json.loads((root / "statistics.json").read_text())
        != statistics_record.model_dump(mode="json")
        or result.execution_orders != donation_confirmation_orders(contract)
        or result.winner
        != (InklingRpaDonationArm.DONATING if statistics_record.confirmed else None)
        or result.accepted != statistics_record.confirmed
    ):
        raise ValueError("INKLING_RPA_DONATION_STATISTICS_REPLAY_MISMATCH")
    source_state = json.loads((root / "source_state.json").read_text())
    completed_states = (
        RunState.CREATED,
        RunState.VERIFIED,
        RunState.LOWERED,
        RunState.COMPILED,
        RunState.CORRECT,
        RunState.TIMED,
    )
    states = (*completed_states, RunState.ACCEPTED) if require_accepted else completed_states
    history = read_ledger_history(root / "ledger.sqlite", expected_run_id)
    if tuple(value.state for value in history) != states or tuple(
        value.payload_sha256 for value in history
    ) != tuple(
        payload_sha256(_ledger_payload(contract, expected_run_id, source_state, root, state))
        for state in states
    ):
        raise ValueError("INKLING_RPA_DONATION_LEDGER_REPLAY_MISMATCH")
    if require_receipt and not require_accepted:
        raise ValueError("INKLING_RPA_DONATION_RECEIPT_REQUIRES_ACCEPTED_LEDGER")
    if require_receipt:
        receipt = InklingRpaDonationConfirmationReceipt.model_validate_json(
            (root / "receipt.json").read_text()
        )
        paths = tuple(
            path.relative_to(root).as_posix()
            for path in sorted(
                (value for value in root.rglob("*") if value.is_file()),
                key=lambda value: value.relative_to(root).as_posix(),
            )
            if path.name != "receipt.json"
        )
        if (
            receipt.confirmation_id != contract.confirmation_id
            or receipt.run_id != result.run_id
            or receipt.result_sha256 != _sha256(root / "result.json")
            or receipt.accepted != result.accepted
            or tuple(value.path for value in receipt.artifacts) != paths
        ):
            raise ValueError("INKLING_RPA_DONATION_RECEIPT_IDENTITY_MISMATCH")
        for artifact in receipt.artifacts:
            path = root / artifact.path
            if (
                path.stat().st_size != artifact.size_bytes
                or _sha256(path) != artifact.sha256
                or _artifact_role(Path(artifact.path)) is not artifact.role
            ):
                raise ValueError(f"INKLING_RPA_DONATION_RECEIPT_ARTIFACT_MISMATCH path={path}")
    elif (root / "receipt.json").exists():
        raise ValueError("INKLING_RPA_DONATION_UNEXPECTED_RECEIPT")
    return result


def _run_staged(
    root: Path,
    contract: InklingRpaDonationConfirmationContract,
    kernel: Callable[..., tuple[Any, Any]],
    devices: tuple[Any, ...],
    source_state: dict[str, Any],
) -> InklingRpaDonationConfirmationResult:
    source_manifest = _source_manifest()
    run_id = semantic_sha256(
        INKLING_RPA_DONATION_CONFIRMATION_SCHEMA,
        contract.confirmation_id,
        source_state["git_commit"],
    )
    identity = InklingRpaDonationRunIdentity(
        schema_version=INKLING_RPA_DONATION_CONFIRMATION_SCHEMA,
        confirmation_id=contract.confirmation_id,
        run_id=run_id,
        source_commit=source_state["git_commit"],
    )
    _write_json(root / "run_identity.json", identity)
    _write_json(root / "contract.json", contract)
    _write_json(root / "source_state.json", source_state)
    _write_text(root / "source_diff.patch", "")
    _write_json(root / "source_manifest.json", source_manifest)
    _write_text(root / "physical.xdsl", canonical_text(inkling_sharded_fused_rpa_schedule()))
    _write_json(root / "plan.json", contract.plan)
    with EvidenceRun(root / "ledger.sqlite", run_id) as run:
        run.create(
            _ledger_payload(contract, run_id, source_state, root, RunState.CREATED),
        )
        run.transition(
            RunState.VERIFIED,
            _ledger_payload(contract, run_id, source_state, root, RunState.VERIFIED),
        )
        run.transition(
            RunState.LOWERED,
            _ledger_payload(contract, run_id, source_state, root, RunState.LOWERED),
        )
        plan = lower_inkling_sharded_rpa_to_pallas(inkling_sharded_fused_rpa_schedule())
        compiled_arms = tuple(
            _compile_arm(contract, arm, kernel, plan, devices)
            for arm in (InklingRpaDonationArm.NON_DONATING, InklingRpaDonationArm.DONATING)
        )
        for compiled in compiled_arms:
            _save_compiled_arm(root, compiled)
        run.transition(
            RunState.COMPILED,
            _ledger_payload(contract, run_id, source_state, root, RunState.COMPILED),
        )
        correctness = _correctness_observations(root, contract, compiled_arms)
        _write_json(root / "correctness.json", correctness)
        run.transition(
            RunState.CORRECT,
            _ledger_payload(contract, run_id, source_state, root, RunState.CORRECT),
        )
        timing_inputs = inkling_sharded_fused_rpa_inputs(contract.timing_seed)
        pre_states = tuple(_state(root, "pre", arm, timing_inputs) for arm in compiled_arms)
        expected_terminal_states = tuple(
            _state(
                root,
                "expected_terminal",
                arm,
                timing_inputs,
                executions=contract.calls_per_block,
            )
            for arm in compiled_arms
        )
        rounds = _timing_rounds(contract, compiled_arms, timing_inputs)
        post_states = tuple(_state(root, "post", arm, timing_inputs) for arm in compiled_arms)
        statistics_record = donation_confirmation_statistics(
            contract,
            rounds,
            expected_terminal_states,
        )
        _write_json(root / "rounds.json", rounds)
        _write_json(root / "statistics.json", statistics_record)
        run.transition(
            RunState.TIMED,
            _ledger_payload(contract, run_id, source_state, root, RunState.TIMED),
        )
        winner = InklingRpaDonationArm.DONATING if statistics_record.confirmed else None
        result = InklingRpaDonationConfirmationResult(
            confirmation_id=contract.confirmation_id,
            run_id=run_id,
            source_surface_id=contract.source_surface_id,
            source_surface_receipt_sha256=contract.source_surface_receipt_sha256,
            source_commit=source_state["git_commit"],
            uv_lock_sha256=source_state["uv_lock_sha256"],
            source_state_sha256=_sha256(root / "source_state.json"),
            source_manifest_sha256=_sha256(root / "source_manifest.json"),
            source_manifest=source_manifest,
            runtime=contract.runtime,
            producer_system=platform.system(),
            producer_machine=platform.machine(),
            devices=_device_inventory(devices),
            plan=contract.plan,
            compiled_arms=tuple(value.evidence for value in compiled_arms),
            correctness=correctness,
            timing_input_sha256=arrays_sha256(timing_inputs),
            pre_timing_states=pre_states,
            expected_terminal_states=expected_terminal_states,
            execution_orders=donation_confirmation_orders(contract),
            rounds=rounds,
            post_timing_states=post_states,
            statistics=statistics_record,
            winner=winner,
            accepted=statistics_record.confirmed,
            claim_scope=contract.claim_scope,
        )
        _write_json(root / "result.json", result)
    validate_inkling_rpa_donation_confirmation(
        root,
        contract,
        require_accepted=False,
        require_receipt=False,
    )
    with EvidenceRun(root / "ledger.sqlite", run_id) as run:
        run.transition(
            RunState.ACCEPTED,
            _ledger_payload(contract, run_id, source_state, root, RunState.ACCEPTED),
        )
    validate_inkling_rpa_donation_confirmation(root, contract, require_receipt=False)
    artifacts = _artifacts(root)
    receipt = InklingRpaDonationConfirmationReceipt(
        receipt_schema=INKLING_RPA_DONATION_CONFIRMATION_RECEIPT_SCHEMA,
        confirmation_id=contract.confirmation_id,
        run_id=run_id,
        result_sha256=_sha256(root / "result.json"),
        artifact_count=len(artifacts),
        artifacts=artifacts,
        accepted=result.accepted,
        claim_scope=contract.claim_scope,
    )
    _write_json_atomic(root / "receipt.json", receipt)
    validate_inkling_rpa_donation_confirmation(root, contract)
    return result


def run_inkling_rpa_donation_confirmation(
    output_root: Path,
    contract: InklingRpaDonationConfirmationContract,
    kernel: Callable[..., tuple[Any, Any]],
) -> InklingRpaDonationConfirmationResult:
    canonical = default_inkling_rpa_donation_confirmation_contract()
    if contract != canonical or contract.hlo_identity_status != "pinned":
        raise ValueError("INKLING_RPA_DONATION_EXTERNAL_CONTRACT_MISMATCH")
    output_root = output_root.absolute()
    repository = _repository_root()
    _require_compilation_root(repository, contract)
    _require_safe_output_root(output_root)
    _require_clean_repository(repository)
    runtime = _runtime_identity()
    if runtime != contract.runtime:
        raise ValueError("INKLING_RPA_DONATION_RUNTIME_MISMATCH")
    if (platform.system(), platform.machine()) != (
        contract.producer_system,
        contract.producer_machine,
    ):
        raise ValueError("INKLING_RPA_DONATION_PRODUCER_HOST_MISMATCH")
    _require_backend_source(contract)
    _require_backend_runtime(contract)
    with _exclusive_lock(output_root):
        if output_root.exists():
            return validate_inkling_rpa_donation_confirmation(output_root, contract)
        source_state = _source_state(repository)
        devices = tuple(jax.devices())
        _device_inventory(devices)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
        )
        try:
            _run_staged(staging, contract, kernel, devices, source_state)
            _rename_directory_noreplace(staging, output_root)
            directory = os.open(output_root.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return validate_inkling_rpa_donation_confirmation(output_root, contract)
        except Exception as error:
            if staging.exists():
                _write_json(
                    staging / "failure.json",
                    {"error_type": type(error).__name__, "message": str(error)},
                )
                try:
                    _rename_directory_noreplace(staging, output_root)
                except OSError:
                    failure = output_root.with_name(f"{output_root.name}.failed-{time.time_ns()}")
                    _rename_directory_noreplace(staging, failure)
            raise
