from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tpu_cake.inkling_gmm_route_corpus import InklingGmmRouteCorpusReport
from tpu_cake.inkling_gmm_tile_search import (
    GmmArmName,
    GmmOperation,
    GmmPolicyPair,
    GmmScreenObservation,
    GmmSearchFamily,
    InklingGmmTileSearchContract,
)

_RUNNER_SCHEMA = "inkling-gmm-tile-search-runner-observations-v1"
_VERIFIED_SCHEMA = "inkling-gmm-tile-search-screening-verification-v1"
_SCOPE_PATTERN = re.compile(
    r"gmm_v2-g_[0-9]+-m_[0-9]+-k_[0-9]+-n_[0-9]+"
    r"-tm_[0-9]+-tk_[0-9]+-tn_[0-9]+"
)
_HLO_INSTRUCTION_PATTERN = re.compile(r"^\s*(?:ROOT\s+)?%?[^=\s]+\s*=")
_LIMITATIONS = [
    "These are raw execution observations, not correctness evidence.",
    "This runner does not create an immutable receipt.",
    "This runner does not make or authorize a promotion decision.",
]
_RAW_KEYS = {
    "schema_version",
    "search_id",
    "contract_sha256",
    "route_report_id",
    "route_report_sha256",
    "source_environment",
    "execution_target",
    "runtime",
    "residency",
    "compiled_policies",
    "screening_observations",
    "limitations",
}
_SOURCE_KEYS = {
    "tpu_cake_git_commit",
    "tpu_cake_uv_lock_sha256",
    "runner_source_sha256",
    "verifier_source_sha256",
    "inkling_git_commit",
    "inkling_uv_lock_sha256",
}
_EXECUTION_TARGET_KEYS = {"project_id", "zone", "instance_name", "accelerator_type"}
_RUNTIME_KEYS = {
    "jax",
    "jaxlib",
    "libtpu",
    "process_count",
    "process_index",
    "devices",
}
_RESIDENCY_KEYS = {
    "estimated_operand_bytes_per_device",
    "free_memory_before_allocation",
    "free_memory_before_timing",
}
_DEVICE_KEYS = {
    "id",
    "process_index",
    "platform",
    "device_kind",
    "coords",
    "core_on_chip",
}
_COMPILED_POLICY_KEYS = {
    "policy",
    "stablehlo_path",
    "stablehlo_sha256",
    "compiler_hlo_path",
    "compiler_hlo_sha256",
    "gmm_scope_labels",
    "stablehlo_bytes",
    "compiler_hlo_bytes",
}
_POLICY_KEYS = {"gate_up", "down", "name"}
_OBSERVATION_KEYS = {"family", "round_index", "position", "arm", "duration_ns"}


def _fail(code: str, **context: object) -> None:
    suffix = " ".join(f"{key}={value}" for key, value in sorted(context.items()))
    raise ValueError(f"INKLING_GMM_SCREEN_VERIFY_{code} {suffix}".rstrip())


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _sha256(raw)


def _object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"{label}_JSON", error=error)
    if not isinstance(value, dict):
        _fail(f"{label}_OBJECT")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        _fail(
            f"{label}_SCHEMA",
            missing=tuple(sorted(expected - set(value))),
            extra=tuple(sorted(set(value) - expected)),
        )


def _hex(value: object, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def _read_contract(path: Path) -> tuple[InklingGmmTileSearchContract, bytes]:
    try:
        raw = path.read_bytes()
        return InklingGmmTileSearchContract.model_validate_json(raw), raw
    except (OSError, ValueError) as error:
        _fail("CONTRACT_READ", path=path, error=error)


def _read_report(path: Path) -> tuple[InklingGmmRouteCorpusReport, bytes]:
    try:
        raw = path.read_bytes()
        return InklingGmmRouteCorpusReport.model_validate_json(raw), raw
    except (OSError, ValueError) as error:
        _fail("ROUTE_REPORT_READ", path=path, error=error)


def _verify_route_report(
    contract: InklingGmmTileSearchContract,
    report: InklingGmmRouteCorpusReport,
    report_raw: bytes,
) -> None:
    report_payload = report.model_dump(mode="json", exclude={"report_id"})
    if report.report_id == "0" * 64 or _json_sha256(report_payload) != report.report_id:
        _fail("ROUTE_REPORT_ID")
    binding = contract.route_corpus
    if report.report_id != binding.report_id:
        _fail("ROUTE_REPORT_BINDING_ID")
    if _sha256(report_raw) != binding.report_sha256:
        _fail("ROUTE_REPORT_BINDING_SHA256")
    if report.corpus_sha256 != binding.corpus_sha256:
        _fail("ROUTE_REPORT_BINDING_CORPUS")
    if report.cohort_scope != binding.cohort_scope:
        _fail("ROUTE_REPORT_BINDING_COHORT")
    if (
        report.concurrency != 48
        or report.num_experts_per_token != 6
        or report.num_routed_experts != 256
        or len(report.request_state_slots) != 48
        or len(set(report.request_state_slots)) != 48
        or len(report.recurrent_state_slots) != 48
        or len(set(report.recurrent_state_slots)) != 48
    ):
        _fail("ROUTE_REPORT_WORKLOAD")
    if (
        report.selected_completion_steps != contract.corpus.completion_steps
        or tuple(range(report.first_moe_layer, report.num_layers)) != contract.corpus.layer_indices
    ):
        _fail("ROUTE_REPORT_INVENTORY")
    expected_keys = tuple(
        (completion_step, layer_index)
        for completion_step in contract.corpus.completion_steps
        for layer_index in contract.corpus.layer_indices
    )
    observed_keys = tuple(
        (group.completion_step, group.layer_index) for group in report.group_sizes
    )
    if observed_keys != expected_keys:
        _fail("ROUTE_REPORT_ORDER")
    if any(
        len(group.group_sizes) != contract.production_abi.global_group_count
        or any(type(value) is not int or value < 0 for value in group.group_sizes)
        or sum(group.group_sizes) != contract.production_abi.m
        for group in report.group_sizes
    ):
        _fail("ROUTE_REPORT_ABI")
    corpus_payload = [group.model_dump(mode="json") for group in report.group_sizes]
    if _json_sha256(corpus_payload) != report.corpus_sha256:
        _fail("ROUTE_REPORT_CORPUS_SHA256")


def _source_environment(contract: InklingGmmTileSearchContract) -> dict[str, str]:
    return {
        "tpu_cake_git_commit": contract.tpu_cake_git_commit,
        "tpu_cake_uv_lock_sha256": contract.tpu_cake_uv_lock_sha256,
        "runner_source_sha256": contract.runner_source_sha256,
        "verifier_source_sha256": contract.verifier_source_sha256,
        "inkling_git_commit": contract.inkling_git_commit,
        "inkling_uv_lock_sha256": contract.inkling_uv_lock_sha256,
    }


def _verify_execution_target(
    contract: InklingGmmTileSearchContract,
    value: object,
) -> dict[str, str]:
    if not isinstance(value, dict):
        _fail("EXECUTION_TARGET_OBJECT")
    _exact_keys(value, _EXECUTION_TARGET_KEYS, label="EXECUTION_TARGET")
    target = contract.target_runtime
    expected = {
        "project_id": target.project_id,
        "zone": target.zone,
        "instance_name": target.instance_name,
        "accelerator_type": target.accelerator_type,
    }
    if value != expected:
        _fail("EXECUTION_TARGET_BINDING")
    return value


def _verify_runtime(contract: InklingGmmTileSearchContract, value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("RUNTIME_OBJECT")
    _exact_keys(value, _RUNTIME_KEYS, label="RUNTIME")
    target = contract.target_runtime
    if (value["jax"], value["jaxlib"], value["libtpu"]) != (
        target.jax,
        target.jaxlib,
        target.libtpu,
    ):
        _fail("RUNTIME_VERSION")
    if (
        type(value["process_count"]) is not int
        or value["process_count"] != target.host_count
        or type(value["process_index"]) is not int
        or value["process_index"] != 0
    ):
        _fail("RUNTIME_PROCESS")
    devices = value["devices"]
    if not isinstance(devices, list) or len(devices) != target.device_count:
        _fail("RUNTIME_DEVICE_COUNT")
    checked = []
    for device in devices:
        if not isinstance(device, dict):
            _fail("RUNTIME_DEVICE_OBJECT")
        _exact_keys(device, _DEVICE_KEYS, label="RUNTIME_DEVICE")
        coords = device["coords"]
        if (
            type(device["id"]) is not int
            or type(device["process_index"]) is not int
            or device["process_index"] != 0
            or device["platform"] != "tpu"
            or device["device_kind"] != "TPU7x"
            or not isinstance(coords, list)
            or len(coords) != 3
            or any(type(coordinate) is not int or coordinate < 0 for coordinate in coords)
            or type(device["core_on_chip"]) is not int
            or device["core_on_chip"] < 0
        ):
            _fail("RUNTIME_DEVICE_VALUE", device=device)
        checked.append(device)
    if {device["id"] for device in checked} != set(range(target.device_count)):
        _fail("RUNTIME_DEVICE_IDS")
    chip_coords = {tuple(device["coords"]) for device in checked}
    topology = "x".join(
        str(max(coordinates[axis] for coordinates in chip_coords) + 1) for axis in range(3)
    )
    core_inventory = {(tuple(device["coords"]), device["core_on_chip"]) for device in checked}
    if (
        topology != target.topology
        or len(chip_coords) != 4
        or len(core_inventory) != target.device_count
    ):
        _fail("RUNTIME_TOPOLOGY", observed=topology)
    return value


def _estimated_operand_bytes_per_device(contract: InklingGmmTileSearchContract) -> int:
    abi = contract.production_abi
    inputs = contract.search.layer_input_banks * abi.m * 4096 * 2
    gate_and_up = (
        2 * contract.search.layer_weight_banks * abi.local_experts_per_device * 4096 * 2048 * 2
    )
    down = contract.search.layer_weight_banks * abi.local_experts_per_device * 2048 * 4096 * 2
    return inputs + gate_and_up + down


def _verify_residency(
    contract: InklingGmmTileSearchContract,
    value: object,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("RESIDENCY_OBJECT")
    _exact_keys(value, _RESIDENCY_KEYS, label="RESIDENCY")
    expected_estimate = _estimated_operand_bytes_per_device(contract)
    if value["estimated_operand_bytes_per_device"] != expected_estimate:
        _fail("RESIDENCY_ESTIMATE")
    before_allocation = value["free_memory_before_allocation"]
    before_timing = value["free_memory_before_timing"]
    expected_count = contract.target_runtime.device_count
    if (
        not isinstance(before_allocation, list)
        or len(before_allocation) != expected_count
        or any(type(item) is not int or item <= 0 for item in before_allocation)
        or not isinstance(before_timing, list)
        or len(before_timing) != expected_count
        or any(type(item) is not int or item <= 0 for item in before_timing)
    ):
        _fail("RESIDENCY_MEMORY_INVENTORY")
    if any(item < contract.search.minimum_free_device_bytes for item in before_allocation):
        _fail("RESIDENCY_ALLOCATION_GATE")
    if expected_estimate >= min(before_allocation):
        _fail("RESIDENCY_ESTIMATE_HEADROOM")
    return value


def _screening_orders(
    contract: InklingGmmTileSearchContract,
    family: GmmSearchFamily,
) -> tuple[tuple[GmmArmName, ...], ...]:
    if family not in contract.search.families:
        _fail("SCREEN_FAMILY", family=family.value)
    names = tuple(arm.name for arm in contract.arms)
    orders = []
    for round_index in range(contract.search.screening_rounds_per_family):
        basis = names if round_index // len(names) % 2 == 0 else tuple(reversed(names))
        offset = round_index % len(names)
        orders.append(basis[offset:] + basis[:offset])
    return tuple(orders)


def _expected_policies(
    contract: InklingGmmTileSearchContract,
) -> tuple[GmmPolicyPair, ...]:
    incumbent = GmmArmName.INCUMBENT
    policies = []
    for family in contract.search.families:
        for arm in contract.arms:
            policy = (
                GmmPolicyPair(gate_up=arm.name, down=incumbent)
                if family is GmmSearchFamily.GATE_UP
                else GmmPolicyPair(gate_up=incumbent, down=arm.name)
            )
            if policy not in policies:
                policies.append(policy)
    return tuple(policies)


def _arm_tiles(
    contract: InklingGmmTileSearchContract,
    arm_name: GmmArmName,
    operation: GmmOperation,
) -> tuple[int, int, int]:
    arms = [arm for arm in contract.arms if arm.name is arm_name]
    kernels = [
        kernel for kernel in contract.production_abi.kernels if kernel.operation is operation
    ]
    if len(arms) != 1 or len(kernels) != 1:
        _fail("POLICY_ABI_INVENTORY")
    arm = arms[0]
    kernel = kernels[0]
    tile_n = kernel.n if arm.tile_n == "N" else kernel.n // 2
    return arm.tile_m, kernel.k, tile_n


def _expected_scopes(
    contract: InklingGmmTileSearchContract,
    policy: GmmPolicyPair,
) -> tuple[str, ...]:
    labels = []
    for operation, arm in (
        (GmmOperation.GATE, policy.gate_up),
        (GmmOperation.UP, policy.gate_up),
        (GmmOperation.DOWN, policy.down),
    ):
        kernels = [
            kernel for kernel in contract.production_abi.kernels if kernel.operation is operation
        ]
        if len(kernels) != 1:
            _fail("POLICY_KERNEL_INVENTORY")
        kernel = kernels[0]
        tile_m, tile_k, tile_n = _arm_tiles(contract, arm, operation)
        labels.append(
            f"gmm_v2-g_32-m_{contract.production_abi.m}-k_{kernel.k}-n_{kernel.n}"
            f"-tm_{tile_m}-tk_{tile_k}-tn_{tile_n}"
        )
    return tuple(sorted(set(labels)))


def _artifact_path(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        _fail(f"{label}_PATH")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        _fail(f"{label}_PATH")
    root = root.resolve()
    candidate = root / relative
    if any(parent.is_symlink() for parent in (candidate, *candidate.parents) if parent != root):
        _fail(f"{label}_PATH", path=value)
    path = candidate.resolve()
    if root not in path.parents or not path.is_file():
        _fail(f"{label}_PATH", path=value)
    return path


def _verify_stablehlo(path: Path, *, expected_sha256: str, expected_bytes: int) -> None:
    raw = path.read_bytes()
    if len(raw) != expected_bytes or _sha256(raw) != expected_sha256:
        _fail("STABLEHLO_CONTENT", path=path)
    try:
        text = raw.decode()
    except UnicodeDecodeError as error:
        _fail("STABLEHLO_TEXT", path=path, error=error)
    if not re.search(r"^module\s+@", text) or not re.search(
        r"^\s*func\.func\s+public\s+@main\b", text, re.MULTILINE
    ):
        _fail("STABLEHLO_STRUCTURE", path=path)


def _verify_compiler_hlo(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    expected_scopes: tuple[str, ...],
) -> None:
    raw = path.read_bytes()
    if len(raw) != expected_bytes or _sha256(raw) != expected_sha256:
        _fail("COMPILER_HLO_CONTENT", path=path)
    try:
        text = raw.decode()
    except UnicodeDecodeError as error:
        _fail("COMPILER_HLO_TEXT", path=path, error=error)
    lines = text.splitlines()
    if not lines or re.fullmatch(r"HloModule\s+\S+.*", lines[0]) is None:
        _fail("COMPILER_HLO_MODULE", path=path)
    entry_lines = [line for line in lines if re.match(r"^ENTRY\s+%?\S+", line)]
    root_lines = [line for line in lines if re.match(r"^\s*ROOT\s+%?[^=\s]+\s*=", line)]
    if len(entry_lines) != 1 or not root_lines:
        _fail("COMPILER_HLO_ENTRY", path=path)
    observed = tuple(sorted(set(_SCOPE_PATTERN.findall(text))))
    if observed != expected_scopes:
        _fail("COMPILER_HLO_SCOPES", expected=expected_scopes, observed=observed)
    scope_lines = [line for line in lines if _SCOPE_PATTERN.search(line)]
    if not scope_lines or any(_HLO_INSTRUCTION_PATTERN.match(line) is None for line in scope_lines):
        _fail("COMPILER_HLO_SCOPE_INSTRUCTION", path=path)


def _verify_compiled_policies(
    contract: InklingGmmTileSearchContract,
    value: object,
    *,
    artifact_root: Path,
) -> list[dict[str, Any]]:
    expected_policies = _expected_policies(contract)
    if not isinstance(value, list) or len(value) != len(expected_policies):
        _fail("COMPILED_POLICY_COUNT")
    verified = []
    for item, expected_policy in zip(value, expected_policies, strict=True):
        if not isinstance(item, dict):
            _fail("COMPILED_POLICY_OBJECT")
        _exact_keys(item, _COMPILED_POLICY_KEYS, label="COMPILED_POLICY")
        policy_payload = item["policy"]
        if not isinstance(policy_payload, dict):
            _fail("COMPILED_POLICY_VALUE")
        _exact_keys(policy_payload, _POLICY_KEYS, label="POLICY")
        try:
            policy = GmmPolicyPair.model_validate(
                {key: policy_payload[key] for key in ("gate_up", "down")}
            )
        except ValueError as error:
            _fail("COMPILED_POLICY_VALUE", error=error)
        if policy_payload["name"] != policy.name:
            _fail("COMPILED_POLICY_NAME", observed=policy_payload["name"])
        if policy != expected_policy:
            _fail(
                "COMPILED_POLICY_ORDER",
                expected=expected_policy.name,
                observed=policy.name,
            )
        hashes = (item["stablehlo_sha256"], item["compiler_hlo_sha256"])
        sizes = (item["stablehlo_bytes"], item["compiler_hlo_bytes"])
        if any(not _hex(value, 64) for value in hashes) or any(
            type(size) is not int or size <= 0 for size in sizes
        ):
            _fail("COMPILED_POLICY_ARTIFACT_METADATA", policy=policy.name)
        expected_scopes = _expected_scopes(contract, policy)
        if item["gmm_scope_labels"] != list(expected_scopes):
            _fail("COMPILED_POLICY_SCOPE_BINDING", policy=policy.name)
        stablehlo = _artifact_path(artifact_root, item["stablehlo_path"], label="STABLEHLO")
        compiler_hlo = _artifact_path(
            artifact_root, item["compiler_hlo_path"], label="COMPILER_HLO"
        )
        if stablehlo == compiler_hlo:
            _fail("COMPILED_POLICY_ARTIFACT_ALIAS", policy=policy.name)
        _verify_stablehlo(
            stablehlo,
            expected_sha256=item["stablehlo_sha256"],
            expected_bytes=item["stablehlo_bytes"],
        )
        _verify_compiler_hlo(
            compiler_hlo,
            expected_sha256=item["compiler_hlo_sha256"],
            expected_bytes=item["compiler_hlo_bytes"],
            expected_scopes=expected_scopes,
        )
        verified.append(
            {
                "policy": policy.model_dump(mode="json"),
                "stablehlo_sha256": item["stablehlo_sha256"],
                "compiler_hlo_sha256": item["compiler_hlo_sha256"],
                "gmm_scope_labels": list(expected_scopes),
            }
        )
    return verified


def _verify_observations(
    contract: InklingGmmTileSearchContract,
    value: object,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if not isinstance(value, list):
        _fail("OBSERVATIONS_LIST")
    expected_count = (
        len(contract.search.families)
        * contract.search.screening_rounds_per_family
        * len(contract.arms)
    )
    if len(value) != expected_count:
        _fail("OBSERVATION_COUNT", expected=expected_count, observed=len(value))
    observations = []
    for item in value:
        if not isinstance(item, dict):
            _fail("OBSERVATION_OBJECT")
        _exact_keys(item, _OBSERVATION_KEYS, label="OBSERVATION")
        try:
            observations.append(GmmScreenObservation.model_validate(item))
        except ValueError as error:
            _fail("OBSERVATION_VALUE", error=error)
    results = []
    finalists = {}
    cursor = 0
    for family in contract.search.families:
        grouped: dict[GmmArmName, list[int]] = {arm.name: [] for arm in contract.arms}
        orders = _screening_orders(contract, family)
        for round_index, order in enumerate(orders):
            for position, arm in enumerate(order):
                observation = observations[cursor]
                cursor += 1
                if (
                    observation.family is not family
                    or observation.round_index != round_index
                    or observation.position != position
                    or observation.arm is not arm
                ):
                    _fail(
                        "OBSERVATION_ORDER",
                        family=family.value,
                        round=round_index,
                        position=position,
                    )
                grouped[arm].append(observation.duration_ns)
        arms = [
            {
                "arm": arm.name.value,
                "durations_ns": grouped[arm.name],
                "median_duration_ns": statistics.median(grouped[arm.name]),
            }
            for arm in contract.arms
        ]
        declaration_order = {arm.name.value: index for index, arm in enumerate(contract.arms)}
        finalist = min(
            arms,
            key=lambda item: (
                item["median_duration_ns"],
                item["arm"] != GmmArmName.INCUMBENT.value,
                declaration_order[item["arm"]],
            ),
        )["arm"]
        finalists[family.value] = finalist
        results.append(
            {
                "family": family.value,
                "execution_orders": [[arm.value for arm in order] for order in orders],
                "arms": arms,
                "finalist": finalist,
            }
        )
    return results, finalists


def verify_screening(
    *,
    contract_path: Path,
    route_report_path: Path,
    raw_observations_path: Path,
) -> dict[str, Any]:
    contract, contract_raw = _read_contract(contract_path)
    if _sha256(Path(__file__).read_bytes()) != contract.verifier_source_sha256:
        _fail("VERIFIER_SOURCE_SHA256")
    report, report_raw = _read_report(route_report_path)
    _verify_route_report(contract, report, report_raw)
    try:
        raw_bytes = raw_observations_path.read_bytes()
    except OSError as error:
        _fail("RAW_READ", path=raw_observations_path, error=error)
    raw = _object(raw_bytes, label="RAW")
    _exact_keys(raw, _RAW_KEYS, label="RAW")
    search_id = _json_sha256(contract.model_dump(mode="json", exclude_computed_fields=True))
    if contract.search_id != search_id:
        _fail("CONTRACT_SEARCH_ID")
    if raw["schema_version"] != _RUNNER_SCHEMA:
        _fail("RAW_VERSION")
    if raw["search_id"] != search_id:
        _fail("SEARCH_ID")
    if raw["contract_sha256"] != _sha256(contract_raw):
        _fail("CONTRACT_SHA256")
    if raw["route_report_id"] != report.report_id:
        _fail("RAW_ROUTE_REPORT_ID")
    if raw["route_report_sha256"] != _sha256(report_raw):
        _fail("RAW_ROUTE_REPORT_SHA256")
    source = raw["source_environment"]
    if not isinstance(source, dict):
        _fail("SOURCE_OBJECT")
    _exact_keys(source, _SOURCE_KEYS, label="SOURCE")
    if source != _source_environment(contract):
        _fail("SOURCE_BINDING")
    execution_target = _verify_execution_target(contract, raw["execution_target"])
    runtime = _verify_runtime(contract, raw["runtime"])
    residency = _verify_residency(contract, raw["residency"])
    policies = _verify_compiled_policies(
        contract,
        raw["compiled_policies"],
        artifact_root=raw_observations_path.parent,
    )
    statistics_by_family, finalists = _verify_observations(contract, raw["screening_observations"])
    if raw["limitations"] != _LIMITATIONS:
        _fail("LIMITATIONS")
    report_payload = {
        "schema_version": _VERIFIED_SCHEMA,
        "evidence_scope": "screening-only",
        "search_id": search_id,
        "contract_sha256": _sha256(contract_raw),
        "route_report_id": report.report_id,
        "route_report_sha256": _sha256(report_raw),
        "raw_observations_sha256": _sha256(raw_bytes),
        "source_environment": source,
        "execution_target": execution_target,
        "runtime": runtime,
        "residency": residency,
        "compiled_policies": policies,
        "screening_statistics": statistics_by_family,
        "finalists": finalists,
        "claims": {
            "correctness_verified": False,
            "confirmation_run": False,
            "immutable_receipt_created": False,
            "promotion_authorized": False,
        },
        "limitations": _LIMITATIONS,
    }
    return {"verification_id": _json_sha256(report_payload), **report_payload}


def write_verified_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently verify an Inkling GMM screening-only run."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--route-report", type=Path, required=True)
    parser.add_argument("--raw-observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = verify_screening(
            contract_path=arguments.contract,
            route_report_path=arguments.route_report,
            raw_observations_path=arguments.raw_observations,
        )
        write_verified_report(arguments.output, report)
    except (OSError, ValueError) as error:
        print(error)
        return 1
    print(
        f"INKLING_GMM_SCREENING_INDEPENDENTLY_VERIFIED verification_id={report['verification_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
