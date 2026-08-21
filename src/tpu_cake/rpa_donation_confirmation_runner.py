from __future__ import annotations

import hashlib
import json
import platform
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
from jax.sharding import PartitionSpec as P

from tpu_cake.contracts import SourceFileContract
from tpu_cake.rpa_donation_confirmation import (
    InklingRpaDonationArm,
    InklingRpaDonationConfirmationContract,
    InklingRpaDonationHloCapture,
    InklingRpaDonationHloCaptureResult,
    default_inkling_rpa_donation_confirmation_contract,
)
from tpu_cake.rpa_lowering import ShardedFusedRpaPlan, lower_inkling_sharded_rpa_to_pallas
from tpu_cake.rpa_surface_runner import (
    _QUERY_CACHE_ALIAS,
    _device_inventory,
    _repository_root,
    _require_backend_runtime,
    _require_backend_source,
    _require_clean_repository,
    _require_compilation_root,
    _require_safe_output_root,
    _source_state,
)
from tpu_cake.rpa_surface_runner import (
    _source_manifest as _surface_source_manifest,
)
from tpu_cake.runner import _runtime_identity
from tpu_cake.workloads.inkling_rpa import (
    inkling_sharded_fused_rpa_inputs,
    inkling_sharded_fused_rpa_schedule,
)


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
    output_root: Path,
    contract: InklingRpaDonationConfirmationContract,
    arm: InklingRpaDonationArm,
    kernel: Callable[..., tuple[Any, Any]],
    plan: ShardedFusedRpaPlan,
    devices: tuple[Any, ...],
) -> InklingRpaDonationHloCapture:
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
    compiler_hlo = lowered.compile().as_text().rstrip("\n") + "\n"
    if "tpu_custom_call" not in compiler_hlo or "RPAd" not in compiler_hlo:
        raise ValueError(f"INKLING_RPA_DONATION_COMPILER_HLO_MARKERS_MISSING arm={arm}")
    _validate_stablehlo_aliases(stablehlo, arm)
    _validate_compiler_hlo_aliases(compiler_hlo, arm)
    arm_root = output_root / arm.value
    arm_root.mkdir(parents=True)
    (arm_root / "stablehlo.txt").write_text(stablehlo)
    (arm_root / "compiler_hlo.txt").write_text(compiler_hlo)
    return InklingRpaDonationHloCapture(
        arm=arm,
        stablehlo_sha256=hashlib.sha256(stablehlo.encode()).hexdigest(),
        compiler_hlo_sha256=hashlib.sha256(compiler_hlo.encode()).hexdigest(),
        compiler_hlo_alias_contract=arm_contract.compiler_hlo_alias_contract,
    )


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
        "src/tpu_cake/rpa_device_main.py",
        "src/tpu_cake/rpa_donation_confirmation.py",
        "src/tpu_cake/rpa_donation_confirmation_runner.py",
    )
    manifest = {value.path: value for value in _surface_source_manifest()}
    for path in additions:
        manifest[path] = SourceFileContract(
            path=path,
            sha256=hashlib.sha256((repository / path).read_bytes()).hexdigest(),
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
    captures = tuple(
        _compile_arm(output_root, contract, arm, kernel, plan, devices)
        for arm in (InklingRpaDonationArm.NON_DONATING, InklingRpaDonationArm.DONATING)
    )
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
