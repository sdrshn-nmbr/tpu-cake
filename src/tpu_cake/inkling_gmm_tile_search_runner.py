from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import sgl_jax
from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm
from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2 import TileSizes

import tpu_cake
import tpu_cake.artifacts as artifacts_module
import tpu_cake.inkling_gmm_route_corpus as route_corpus_module
import tpu_cake.inkling_gmm_tile_search as tile_search_module
from tpu_cake.artifacts import file_sha256
from tpu_cake.inkling_gmm_route_corpus import InklingGmmRouteCorpusReport
from tpu_cake.inkling_gmm_tile_search import (
    GMM_LAYER_INDICES,
    GmmArmName,
    GmmKernelAbi,
    GmmOperation,
    GmmPolicyPair,
    GmmScreenObservation,
    GmmSearchFamily,
    GmmTileArm,
    InklingGmmTileSearchContract,
    screening_orders,
    validate_route_corpus_binding,
)

_LOGGER = logging.getLogger("tpu_cake.inkling_gmm_tile_search_runner")
_EXPERT_AXIS = "expert"
_LAYER_COUNT = len(GMM_LAYER_INDICES)
_GMM_SCOPE_PATTERN = (
    r"gmm_v2-g_[0-9]+-m_[0-9]+-k_[0-9]+-n_[0-9]+"
    r"-tm_[0-9]+-tk_[0-9]+-tn_[0-9]+"
)
_OPERAND_MINIMUM = -0.02
_OPERAND_MAXIMUM = 0.02
_METADATA_ROOT = "http://metadata.google.internal/computeMetadata/v1"


class GmmTileSearchRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScreenVariant:
    family: GmmSearchFamily
    arm: GmmArmName
    policy: GmmPolicyPair

    @property
    def name(self) -> str:
        return f"{self.family.value}/{self.arm.value}"


@dataclass(frozen=True)
class ResidentOperands:
    inputs: jax.Array
    gate_weights: jax.Array
    up_weights: jax.Array
    down_weights: jax.Array

    def executable_arguments(self) -> tuple[jax.Array, ...]:
        return self.inputs, self.gate_weights, self.up_weights, self.down_weights


@dataclass(frozen=True)
class HloArtifacts:
    stablehlo_path: Path
    stablehlo_sha256: str
    compiler_hlo_path: Path
    compiler_hlo_sha256: str
    gmm_scope_labels: tuple[str, ...]
    gmm_custom_call_counts: Mapping[str, int]


@dataclass(frozen=True)
class CompiledChain:
    policy: GmmPolicyPair
    executable: Any
    hlo: HloArtifacts


@dataclass(frozen=True)
class PreparedScreen:
    contract: InklingGmmTileSearchContract
    report: InklingGmmRouteCorpusReport
    devices: tuple[jax.Device, ...]
    operands: ResidentOperands
    routes: tuple[jax.Array, ...]
    variants: Mapping[tuple[GmmSearchFamily, GmmArmName], CompiledChain]
    free_memory_before_allocation: tuple[int, ...]
    free_memory_before_timing: tuple[int, ...]


@dataclass(frozen=True)
class SourceEnvironment:
    tpu_cake_root: Path
    inkling_root: Path
    tpu_cake_git_commit: str
    tpu_cake_uv_lock_sha256: str
    runner_source_sha256: str
    verifier_source_sha256: str
    inkling_git_commit: str
    inkling_uv_lock_sha256: str


def _fail(code: str, **context: object) -> None:
    fields = " ".join(f"{key}={value}" for key, value in sorted(context.items()))
    _LOGGER.error("INKLING_GMM_RUNNER_FAILURE code=%s %s", code, fields)
    raise GmmTileSearchRunnerError(f"INKLING_GMM_RUNNER_{code} {fields}".rstrip())


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _canonical_hlo(text: str) -> str:
    return text.rstrip("\n") + "\n"


def _arm(contract: InklingGmmTileSearchContract, name: GmmArmName) -> GmmTileArm:
    matches = tuple(arm for arm in contract.arms if arm.name is name)
    if len(matches) != 1:
        raise ValueError(f"GMM arm inventory is not unique: {name.value}")
    return matches[0]


def screen_variants(contract: InklingGmmTileSearchContract) -> tuple[ScreenVariant, ...]:
    incumbent = GmmArmName.INCUMBENT
    return tuple(
        ScreenVariant(
            family=family,
            arm=arm.name,
            policy=(
                GmmPolicyPair(gate_up=arm.name, down=incumbent)
                if family is GmmSearchFamily.GATE_UP
                else GmmPolicyPair(gate_up=incumbent, down=arm.name)
            ),
        )
        for family in contract.search.families
        for arm in contract.arms
    )


def resolved_policy_tiles(
    contract: InklingGmmTileSearchContract,
    policy: GmmPolicyPair,
) -> tuple[TileSizes, TileSizes]:
    gate_kernel = next(
        kernel
        for kernel in contract.production_abi.kernels
        if kernel.operation is GmmOperation.GATE
    )
    down_kernel = next(
        kernel
        for kernel in contract.production_abi.kernels
        if kernel.operation is GmmOperation.DOWN
    )
    return (
        TileSizes(*_arm(contract, policy.gate_up).resolve_tiles(gate_kernel)),
        TileSizes(*_arm(contract, policy.down).resolve_tiles(down_kernel)),
    )


def gmm_scope_name(kernel: GmmKernelAbi, tiles: TileSizes) -> str:
    return (
        f"gmm_v2-g_32-m_288-k_{kernel.k}-n_{kernel.n}"
        f"-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}"
    )


def expected_gmm_scope_labels(
    contract: InklingGmmTileSearchContract,
    policy: GmmPolicyPair,
) -> tuple[str, str, str]:
    gate_tiles, down_tiles = resolved_policy_tiles(contract, policy)
    kernels = {kernel.operation: kernel for kernel in contract.production_abi.kernels}
    gate = gmm_scope_name(kernels[GmmOperation.GATE], gate_tiles)
    up = gmm_scope_name(kernels[GmmOperation.UP], gate_tiles)
    down = gmm_scope_name(kernels[GmmOperation.DOWN], down_tiles)
    return gate, up, down


def extract_gmm_scope_labels(compiler_hlo: str) -> tuple[str, ...]:
    return tuple(sorted(set(re.findall(_GMM_SCOPE_PATTERN, compiler_hlo))))


def _entry_gmm_custom_call_lines(compiler_hlo: str) -> tuple[str, ...]:
    lines = compiler_hlo.splitlines()
    entry_indices = tuple(
        index for index, line in enumerate(lines) if re.match(r"^ENTRY\s+%?\S+", line)
    )
    if len(entry_indices) != 1:
        return ()
    return tuple(
        line.strip()
        for line in lines[entry_indices[0] + 1 :]
        if re.match(r"^\s*(?:ROOT\s+)?%?[^=\s]+\s*=", line)
        and re.search(_GMM_SCOPE_PATTERN, line)
        and "custom-call(" in line
        and 'custom_call_target="tpu_custom_call"' in line
    )


def expected_gmm_custom_call_counts(
    contract: InklingGmmTileSearchContract,
    policy: GmmPolicyPair,
) -> Counter[str]:
    per_layer = Counter(expected_gmm_scope_labels(contract, policy))
    return Counter({label: count * _LAYER_COUNT for label, count in per_layer.items()})


def observed_gmm_custom_call_counts(compiler_hlo: str) -> Counter[str]:
    return Counter(
        match.group()
        for line in _entry_gmm_custom_call_lines(compiler_hlo)
        if (match := re.search(_GMM_SCOPE_PATTERN, line)) is not None
    )


def validate_gmm_scope_labels(
    contract: InklingGmmTileSearchContract,
    policy: GmmPolicyPair,
    compiler_hlo: str,
) -> tuple[str, ...]:
    if not compiler_hlo.startswith("HloModule ") or "\nENTRY " not in compiler_hlo:
        _fail("COMPILED_HLO_STRUCTURE", policy=policy.name)
    observed = extract_gmm_scope_labels(compiler_hlo)
    expected = tuple(sorted(set(expected_gmm_scope_labels(contract, policy))))
    if observed != expected:
        _fail(
            "COMPILED_SCOPE_MISMATCH",
            policy=policy.name,
            expected=expected,
            observed=observed,
        )
    expected_counts = expected_gmm_custom_call_counts(contract, policy)
    observed_counts = observed_gmm_custom_call_counts(compiler_hlo)
    if observed_counts != expected_counts:
        _fail(
            "COMPILED_CUSTOM_CALL_COUNT_MISMATCH",
            policy=policy.name,
            expected=dict(expected_counts),
            observed=dict(observed_counts),
        )
    return observed


def route_matrices(
    contract: InklingGmmTileSearchContract,
    report: InklingGmmRouteCorpusReport,
) -> tuple[np.ndarray, ...]:
    groups = {
        (group.completion_step, group.layer_index): group.group_sizes
        for group in report.group_sizes
    }
    expected_keys = tuple(
        (completion_step, layer_index)
        for completion_step in contract.corpus.completion_steps
        for layer_index in contract.corpus.layer_indices
    )
    if tuple(groups) != expected_keys:
        raise ValueError("GMM route corpus order does not match the search contract")
    matrices = tuple(
        np.asarray(
            [groups[(completion_step, layer_index)] for layer_index in GMM_LAYER_INDICES],
            dtype=np.int32,
        )
        for completion_step in contract.corpus.completion_steps
    )
    expected_shape = (_LAYER_COUNT, contract.production_abi.global_group_count)
    if any(matrix.shape != expected_shape for matrix in matrices):
        raise ValueError("GMM route matrix shape mismatch")
    if any(np.any(matrix < 0) or np.any(matrix.sum(axis=1) != 288) for matrix in matrices):
        raise ValueError("GMM route matrix violates the production M dimension")
    return matrices


def estimated_resident_bytes_per_device(contract: InklingGmmTileSearchContract) -> int:
    abi = contract.production_abi
    dtype_bytes = 2
    inputs = contract.search.layer_input_banks * abi.m * 4096 * dtype_bytes
    gate_up = (
        2
        * contract.search.layer_weight_banks
        * abi.local_experts_per_device
        * 4096
        * 2048
        * dtype_bytes
    )
    down = (
        contract.search.layer_weight_banks
        * abi.local_experts_per_device
        * 2048
        * 4096
        * dtype_bytes
    )
    return inputs + gate_up + down


def _read_contract(path: Path) -> InklingGmmTileSearchContract:
    try:
        return InklingGmmTileSearchContract.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        _fail("CONTRACT_READ", path=path, error=error)


def _read_route_report(path: Path) -> tuple[InklingGmmRouteCorpusReport, bytes]:
    try:
        raw = path.read_bytes()
        return InklingGmmRouteCorpusReport.model_validate_json(raw), raw
    except (OSError, ValueError) as error:
        _fail("ROUTE_REPORT_READ", path=path, error=error)


def _git_head(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        _fail("GIT_HEAD", root=root, stderr=error.stderr.strip())


def _git_tracked_status(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        _fail("GIT_STATUS", root=root, stderr=error.stderr.strip())


def infer_inkling_root() -> Path:
    package_file = getattr(sgl_jax, "__file__", None)
    if package_file is None:
        _fail("SGL_JAX_PACKAGE_PATH")
    package_path = Path(package_file).resolve()
    candidates = tuple(
        parent
        for parent in package_path.parents
        if (parent / "engine" / "sglang-jax" / "python" / "sgl_jax").is_dir()
    )
    if len(candidates) != 1:
        _fail("INKLING_ROOT_INFERENCE", package=package_path, candidates=candidates)
    return candidates[0]


def validate_source_environment(
    contract: InklingGmmTileSearchContract,
) -> SourceEnvironment:
    runner_path = Path(__file__).resolve()
    tpu_cake_root = runner_path.parents[2]
    inkling_root = infer_inkling_root()
    expected_package_root = tpu_cake_root / "src/tpu_cake"
    imported_module_paths = tuple(
        Path(module.__file__).resolve()
        for module in (
            tpu_cake,
            artifacts_module,
            route_corpus_module,
            tile_search_module,
        )
        if module.__file__ is not None
    )
    if len(imported_module_paths) != 4 or any(
        path.parent != expected_package_root for path in imported_module_paths
    ):
        _fail(
            "TPU_CAKE_IMPORT_ROOT",
            expected=expected_package_root,
            observed=imported_module_paths,
        )
    runner_hash = file_sha256(runner_path)
    verifier_path = tpu_cake_root / "src/tpu_cake/inkling_gmm_tile_search_verifier.py"
    if runner_hash != contract.runner_source_sha256:
        _fail("RUNNER_SOURCE_HASH", path=Path(__file__))
    if not verifier_path.is_file():
        _fail("VERIFIER_SOURCE_MISSING", path=verifier_path)
    verifier_hash = file_sha256(verifier_path)
    if verifier_hash != contract.verifier_source_sha256:
        _fail("VERIFIER_SOURCE_HASH", path=verifier_path)
    tpu_cake_head = _git_head(tpu_cake_root)
    if tpu_cake_head != contract.tpu_cake_git_commit:
        _fail(
            "TPU_CAKE_COMMIT",
            expected=contract.tpu_cake_git_commit,
            observed=tpu_cake_head,
        )
    if _git_tracked_status(tpu_cake_root):
        _fail("TPU_CAKE_TRACKED_CHANGES", root=tpu_cake_root)
    tpu_cake_lock_hash = file_sha256(tpu_cake_root / "uv.lock")
    if tpu_cake_lock_hash != contract.tpu_cake_uv_lock_sha256:
        _fail("TPU_CAKE_UV_LOCK", path=tpu_cake_root / "uv.lock")
    inkling_head = _git_head(inkling_root)
    if inkling_head != contract.inkling_git_commit:
        _fail(
            "INKLING_COMMIT",
            expected=contract.inkling_git_commit,
            observed=inkling_head,
        )
    lock_path = inkling_root / "uv.lock"
    if file_sha256(lock_path) != contract.inkling_uv_lock_sha256:
        _fail("INKLING_UV_LOCK", path=lock_path)
    if _git_tracked_status(inkling_root):
        _fail("INKLING_TRACKED_CHANGES", root=inkling_root)
    for source in contract.implementation_source_manifest:
        path = inkling_root / source.path
        if not path.is_file() or file_sha256(path) != source.sha256:
            _fail("INKLING_SOURCE", path=path, expected=source.sha256)
    return SourceEnvironment(
        tpu_cake_root=tpu_cake_root,
        inkling_root=inkling_root,
        tpu_cake_git_commit=tpu_cake_head,
        tpu_cake_uv_lock_sha256=tpu_cake_lock_hash,
        runner_source_sha256=runner_hash,
        verifier_source_sha256=verifier_hash,
        inkling_git_commit=inkling_head,
        inkling_uv_lock_sha256=file_sha256(lock_path),
    )


def _metadata_text(path: str) -> str:
    request = Request(
        f"{_METADATA_ROOT}/{path}",
        headers={"Metadata-Flavor": "Google"},
    )
    try:
        with urlopen(request, timeout=3) as response:
            if response.headers.get("Metadata-Flavor") != "Google":
                _fail("METADATA_RESPONSE_HEADER", path=path)
            return response.read().decode().strip()
    except OSError as error:
        _fail("METADATA_READ", path=path, error=error)


def execution_target_metadata(contract: InklingGmmTileSearchContract) -> dict[str, str]:
    observed = {
        "project_id": _metadata_text("project/project-id"),
        "zone": _metadata_text("instance/zone").rsplit("/", maxsplit=1)[-1],
        "instance_name": _metadata_text("instance/name"),
        "accelerator_type": _metadata_text("instance/attributes/accelerator-type"),
    }
    expected = {
        "project_id": contract.target_runtime.project_id,
        "zone": contract.target_runtime.zone,
        "instance_name": contract.target_runtime.instance_name,
        "accelerator_type": contract.target_runtime.accelerator_type,
    }
    if observed != expected:
        _fail("EXECUTION_TARGET", expected=expected, observed=observed)
    return observed


def _runtime_devices(contract: InklingGmmTileSearchContract) -> tuple[jax.Device, ...]:
    devices = tuple(jax.devices())
    if len(devices) != contract.target_runtime.device_count:
        _fail(
            "DEVICE_COUNT",
            expected=contract.target_runtime.device_count,
            observed=len(devices),
        )
    if any(device.platform != "tpu" for device in devices):
        _fail("DEVICE_PLATFORM", observed=tuple(device.platform for device in devices))
    if any(device.device_kind != "TPU7x" for device in devices):
        _fail("DEVICE_KIND", observed=tuple(device.device_kind for device in devices))
    if jax.process_count() != contract.target_runtime.host_count or jax.process_index() != 0:
        _fail(
            "HOST_TOPOLOGY",
            expected_host_count=contract.target_runtime.host_count,
            observed_host_count=jax.process_count(),
            process_index=jax.process_index(),
        )
    if any(device.process_index != 0 for device in devices):
        _fail(
            "DEVICE_PROCESS_INDEX",
            observed=tuple(device.process_index for device in devices),
        )
    coords = tuple(getattr(device, "coords", None) for device in devices)
    if any(coord is None or len(coord) != 3 for coord in coords):
        _fail("DEVICE_COORDS", observed=coords)
    chip_coords = {tuple(coord) for coord in coords if coord is not None}
    observed_topology = "x".join(
        str(max(coord[axis] for coord in chip_coords) + 1) for axis in range(3)
    )
    core_inventory = {
        (tuple(device.coords), getattr(device, "core_on_chip", None)) for device in devices
    }
    if (
        observed_topology != contract.target_runtime.topology
        or len(chip_coords) != 4
        or len(core_inventory) != contract.target_runtime.device_count
    ):
        _fail(
            "DEVICE_TOPOLOGY",
            expected=contract.target_runtime.topology,
            observed=observed_topology,
            chip_count=len(chip_coords),
            core_count=len(core_inventory),
        )
    try:
        libtpu_version = version("libtpu")
    except PackageNotFoundError:
        _fail("LIBTPU_PACKAGE")
    observed_versions = (jax.__version__, jaxlib.__version__, libtpu_version)
    expected_versions = (
        contract.target_runtime.jax,
        contract.target_runtime.jaxlib,
        contract.target_runtime.libtpu,
    )
    if observed_versions != expected_versions:
        _fail("RUNTIME_VERSION", expected=expected_versions, observed=observed_versions)
    return devices


def _free_device_bytes(device: jax.Device) -> int:
    stats = device.memory_stats()
    if not isinstance(stats, dict):
        _fail("MEMORY_STATS_UNAVAILABLE", device=device)
    limit = stats.get("bytes_limit")
    used = stats.get("bytes_in_use")
    if not isinstance(limit, int) or not isinstance(used, int) or limit < used:
        _fail("MEMORY_STATS_INVALID", device=device, stats=stats)
    return limit - used


def require_allocation_headroom(
    contract: InklingGmmTileSearchContract,
    devices: Sequence[jax.Device],
) -> tuple[int, ...]:
    free = tuple(_free_device_bytes(device) for device in devices)
    minimum = contract.search.minimum_free_device_bytes
    if any(value < minimum for value in free):
        _fail("FREE_MEMORY", required=minimum, observed=free)
    estimate = estimated_resident_bytes_per_device(contract)
    if estimate >= min(free):
        _fail("OPERAND_MEMORY_ESTIMATE", estimated=estimate, available=min(free))
    _LOGGER.info(
        "INKLING_GMM_RUNNER_MEMORY_CHECK free=%s estimated_operands_per_device=%d",
        free,
        estimate,
    )
    return free


def _key_words(seed: int, stream: str, device_index: int) -> np.ndarray:
    digest = hashlib.sha256(f"{seed}:{stream}:{device_index}".encode()).digest()
    return np.frombuffer(digest[:8], dtype="<u4").copy()


def _generate_uniform_shards(
    *,
    seed: int,
    stream: str,
    local_shape: tuple[int, ...],
    devices: Sequence[jax.Device],
    replicated: bool,
) -> jax.Array:
    key_words = np.stack(
        [
            _key_words(seed, stream, 0 if replicated else device_index)
            for device_index in range(len(devices))
        ]
    )

    def generate(words: jax.Array) -> jax.Array:
        key = jax.random.fold_in(jax.random.PRNGKey(words[0]), words[1])
        return jax.random.uniform(
            key,
            local_shape,
            dtype=jnp.bfloat16,
            minval=_OPERAND_MINIMUM,
            maxval=_OPERAND_MAXIMUM,
        )

    result = jax.pmap(generate, devices=devices)(key_words)
    result.block_until_ready()
    return result


def allocate_resident_operands(
    contract: InklingGmmTileSearchContract,
    devices: Sequence[jax.Device],
) -> ResidentOperands:
    abi = contract.production_abi
    seed = contract.search.operand_seed
    shapes = {
        "inputs": (_LAYER_COUNT, abi.m, 4096),
        "gate": (_LAYER_COUNT, abi.local_experts_per_device, 4096, 2048),
        "up": (_LAYER_COUNT, abi.local_experts_per_device, 4096, 2048),
        "down": (_LAYER_COUNT, abi.local_experts_per_device, 2048, 4096),
    }
    arrays: dict[str, jax.Array] = {}
    for stream, shape in shapes.items():
        _LOGGER.info("INKLING_GMM_RUNNER_ALLOCATE stream=%s local_shape=%s", stream, shape)
        try:
            arrays[stream] = _generate_uniform_shards(
                seed=seed,
                stream=stream,
                local_shape=shape,
                devices=devices,
                replicated=stream == "inputs",
            )
        except Exception as error:
            _LOGGER.exception("INKLING_GMM_RUNNER_ALLOCATION_FAILED stream=%s", stream)
            raise GmmTileSearchRunnerError(
                f"INKLING_GMM_RUNNER_ALLOCATION stream={stream}"
            ) from error
    operands = ResidentOperands(
        inputs=arrays["inputs"],
        gate_weights=arrays["gate"],
        up_weights=arrays["up"],
        down_weights=arrays["down"],
    )
    validate_resident_operands(contract, devices, operands)
    return operands


def validate_resident_operands(
    contract: InklingGmmTileSearchContract,
    devices: Sequence[jax.Device],
    operands: ResidentOperands,
) -> None:
    abi = contract.production_abi
    expected = (
        (len(devices), _LAYER_COUNT, abi.m, 4096),
        (len(devices), _LAYER_COUNT, abi.local_experts_per_device, 4096, 2048),
        (len(devices), _LAYER_COUNT, abi.local_experts_per_device, 4096, 2048),
        (len(devices), _LAYER_COUNT, abi.local_experts_per_device, 2048, 4096),
    )
    arrays = operands.executable_arguments()
    observed_devices = set(devices)
    for name, value, shape in zip(("inputs", "gate", "up", "down"), arrays, expected, strict=True):
        if value.shape != shape or value.dtype != jnp.bfloat16:
            _fail(
                "OPERAND_ABI",
                operand=name,
                expected_shape=shape,
                observed_shape=value.shape,
                observed_dtype=value.dtype,
            )
        shard_devices = {shard.device for shard in value.addressable_shards}
        if shard_devices != observed_devices:
            _fail("OPERAND_RESIDENCY", operand=name, observed=shard_devices)


def place_route_matrices(
    matrices: Sequence[np.ndarray],
    devices: Sequence[jax.Device],
) -> tuple[jax.Array, ...]:
    place = jax.pmap(lambda value: value, devices=devices)
    routes = tuple(
        place(np.broadcast_to(matrix, (len(devices), *matrix.shape))) for matrix in matrices
    )
    jax.block_until_ready(routes)
    return routes


def _full_layer_chains(
    inputs: jax.Array,
    gate_weights: jax.Array,
    up_weights: jax.Array,
    down_weights: jax.Array,
    group_sizes: jax.Array,
    *,
    gate_up_tiles: TileSizes,
    down_tiles: TileSizes,
) -> jax.Array:
    group_offset = jax.lax.axis_index(_EXPERT_AXIS).astype(jnp.int32) * 32
    live_outputs = []
    for layer_position in range(_LAYER_COUNT):
        layer_inputs = inputs[layer_position]
        layer_groups = group_sizes[layer_position]
        common = {
            "group_sizes": layer_groups,
            "preferred_element_type": jnp.float32,
            "group_offset": group_offset,
            "interpret": False,
            "maybe_quantize_lhs": False,
            "acc_dtype": jnp.float32,
            "activation_quantized_dtype": None,
        }
        gate = gmm(
            lhs=layer_inputs,
            rhs=gate_weights[layer_position],
            zero_initialize=False,
            v2_tile_info=gate_up_tiles,
            **common,
        )
        up = gmm(
            lhs=layer_inputs,
            rhs=up_weights[layer_position],
            zero_initialize=False,
            v2_tile_info=gate_up_tiles,
            **common,
        )
        intermediate = jax.nn.silu(gate) * up
        down = gmm(
            lhs=intermediate,
            rhs=down_weights[layer_position],
            zero_initialize=True,
            v2_tile_info=down_tiles,
            **common,
        )
        first_local_group = group_offset
        active_start = jnp.sum(
            jnp.where(
                jnp.arange(layer_groups.shape[0], dtype=jnp.int32) < first_local_group,
                layer_groups,
                0,
            )
        )
        active_count = jax.lax.dynamic_slice_in_dim(
            layer_groups,
            first_local_group,
            32,
        ).sum()
        live_row = jnp.minimum(active_start, down.shape[0] - 1)
        live_outputs.append(jnp.where(active_count > 0, down[live_row, 0], jnp.float32(0.0)))
    return jnp.stack(live_outputs)


def _policy_slug(policy: GmmPolicyPair) -> str:
    return f"gate-up-{policy.gate_up.value}__down-{policy.down.value}"


def compile_chain(
    contract: InklingGmmTileSearchContract,
    policy: GmmPolicyPair,
    *,
    devices: Sequence[jax.Device],
    operands: ResidentOperands,
    example_routes: jax.Array,
    hlo_root: Path,
) -> CompiledChain:
    gate_up_tiles, down_tiles = resolved_policy_tiles(contract, policy)

    def chain(
        inputs: jax.Array,
        gate_weights: jax.Array,
        up_weights: jax.Array,
        down_weights: jax.Array,
        group_sizes: jax.Array,
    ) -> jax.Array:
        return _full_layer_chains(
            inputs,
            gate_weights,
            up_weights,
            down_weights,
            group_sizes,
            gate_up_tiles=gate_up_tiles,
            down_tiles=down_tiles,
        )

    mapped = jax.pmap(chain, axis_name=_EXPERT_AXIS, devices=devices)
    arguments = (*operands.executable_arguments(), example_routes)
    _LOGGER.info("INKLING_GMM_RUNNER_COMPILE policy=%s", policy.name)
    try:
        lowered = mapped.lower(*arguments)
        stablehlo = _canonical_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
        executable = lowered.compile()
        compiler_hlo = _canonical_hlo(executable.as_text())
    except Exception as error:
        _LOGGER.exception("INKLING_GMM_RUNNER_COMPILE_FAILED policy=%s", policy.name)
        raise GmmTileSearchRunnerError(
            f"INKLING_GMM_RUNNER_COMPILE policy={policy.name}"
        ) from error
    scopes = validate_gmm_scope_labels(contract, policy, compiler_hlo)
    custom_call_counts = observed_gmm_custom_call_counts(compiler_hlo)
    slug = _policy_slug(policy)
    stablehlo_path = hlo_root / f"{slug}.stablehlo.mlir"
    compiler_hlo_path = hlo_root / f"{slug}.compiler-hlo.txt"
    stablehlo_path.write_text(stablehlo)
    compiler_hlo_path.write_text(compiler_hlo)
    return CompiledChain(
        policy=policy,
        executable=executable,
        hlo=HloArtifacts(
            stablehlo_path=stablehlo_path,
            stablehlo_sha256=_sha256_text(stablehlo),
            compiler_hlo_path=compiler_hlo_path,
            compiler_hlo_sha256=_sha256_text(compiler_hlo),
            gmm_scope_labels=scopes,
            gmm_custom_call_counts=dict(custom_call_counts),
        ),
    )


def compile_screen_executables(
    contract: InklingGmmTileSearchContract,
    *,
    devices: Sequence[jax.Device],
    operands: ResidentOperands,
    example_routes: jax.Array,
    hlo_root: Path,
) -> Mapping[tuple[GmmSearchFamily, GmmArmName], CompiledChain]:
    compiled_by_policy: dict[str, CompiledChain] = {}
    variants: dict[tuple[GmmSearchFamily, GmmArmName], CompiledChain] = {}
    for variant in screen_variants(contract):
        compiled = compiled_by_policy.get(variant.policy.name)
        if compiled is None:
            compiled = compile_chain(
                contract,
                variant.policy,
                devices=devices,
                operands=operands,
                example_routes=example_routes,
                hlo_root=hlo_root,
            )
            compiled_by_policy[variant.policy.name] = compiled
        variants[(variant.family, variant.arm)] = compiled
    if len(variants) != 10 or len(compiled_by_policy) != 9:
        _fail(
            "EXECUTABLE_INVENTORY",
            screen_variants=len(variants),
            unique_policies=len(compiled_by_policy),
        )
    return variants


def prepare_screen(
    contract: InklingGmmTileSearchContract,
    report: InklingGmmRouteCorpusReport,
    *,
    devices: tuple[jax.Device, ...],
    hlo_root: Path,
) -> PreparedScreen:
    free_memory_before_allocation = require_allocation_headroom(contract, devices)
    matrices = route_matrices(contract, report)
    operands = allocate_resident_operands(contract, devices)
    routes = place_route_matrices(matrices, devices)
    variants = compile_screen_executables(
        contract,
        devices=devices,
        operands=operands,
        example_routes=routes[0],
        hlo_root=hlo_root,
    )
    free_memory_before_timing = tuple(_free_device_bytes(device) for device in devices)
    return PreparedScreen(
        contract=contract,
        report=report,
        devices=devices,
        operands=operands,
        routes=routes,
        variants=variants,
        free_memory_before_allocation=free_memory_before_allocation,
        free_memory_before_timing=free_memory_before_timing,
    )


def time_full_corpus_block(prepared: PreparedScreen, executable: CompiledChain) -> int:
    arguments = prepared.operands.executable_arguments()
    outputs = []
    started_ns = time.monotonic_ns()
    try:
        for routes in prepared.routes:
            outputs.append(executable.executable(*arguments, routes))
        jax.block_until_ready(outputs)
    except Exception as error:
        _LOGGER.exception("INKLING_GMM_RUNNER_EXECUTION_FAILED policy=%s", executable.policy.name)
        raise GmmTileSearchRunnerError(
            f"INKLING_GMM_RUNNER_EXECUTION policy={executable.policy.name}"
        ) from error
    duration_ns = time.monotonic_ns() - started_ns
    if duration_ns <= 0:
        _fail("TIMER", policy=executable.policy.name, duration_ns=duration_ns)
    return duration_ns


def run_screening(prepared: PreparedScreen) -> tuple[GmmScreenObservation, ...]:
    contract = prepared.contract
    for variant in screen_variants(contract):
        executable = prepared.variants[(variant.family, variant.arm)]
        for _ in range(contract.search.warmup_full_corpus_blocks_per_arm):
            time_full_corpus_block(prepared, executable)
    observations = []
    for family in contract.search.families:
        for round_index, order in enumerate(screening_orders(contract, family)):
            for position, arm in enumerate(order):
                executable = prepared.variants[(family, arm)]
                duration_ns = time_full_corpus_block(prepared, executable)
                observations.append(
                    GmmScreenObservation(
                        family=family,
                        round_index=round_index,
                        position=position,
                        arm=arm,
                        duration_ns=duration_ns,
                    )
                )
                _LOGGER.info(
                    "INKLING_GMM_RUNNER_SCREEN family=%s round=%d position=%d "
                    "arm=%s duration_ns=%d",
                    family.value,
                    round_index,
                    position,
                    arm.value,
                    duration_ns,
                )
    return tuple(observations)


def _device_observation(device: jax.Device) -> dict[str, object]:
    return {
        "id": device.id,
        "process_index": device.process_index,
        "platform": device.platform,
        "device_kind": device.device_kind,
        "coords": getattr(device, "coords", None),
        "core_on_chip": getattr(device, "core_on_chip", None),
    }


def write_raw_observations(
    path: Path,
    prepared: PreparedScreen,
    observations: Sequence[GmmScreenObservation],
    *,
    source_environment: SourceEnvironment,
    execution_target: Mapping[str, str],
    contract_path: Path,
    route_report_path: Path,
) -> None:
    unique = {compiled.policy.name: compiled for compiled in prepared.variants.values()}
    payload = {
        "schema_version": "inkling-gmm-tile-search-runner-observations-v1",
        "search_id": prepared.contract.search_id,
        "contract_sha256": file_sha256(contract_path),
        "route_report_id": prepared.report.report_id,
        "route_report_sha256": file_sha256(route_report_path),
        "source_environment": {
            "tpu_cake_git_commit": source_environment.tpu_cake_git_commit,
            "tpu_cake_uv_lock_sha256": source_environment.tpu_cake_uv_lock_sha256,
            "runner_source_sha256": source_environment.runner_source_sha256,
            "verifier_source_sha256": source_environment.verifier_source_sha256,
            "inkling_git_commit": source_environment.inkling_git_commit,
            "inkling_uv_lock_sha256": source_environment.inkling_uv_lock_sha256,
        },
        "execution_target": dict(execution_target),
        "runtime": {
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "libtpu": version("libtpu"),
            "process_count": jax.process_count(),
            "process_index": jax.process_index(),
            "devices": [_device_observation(device) for device in prepared.devices],
        },
        "residency": {
            "estimated_operand_bytes_per_device": estimated_resident_bytes_per_device(
                prepared.contract
            ),
            "free_memory_before_allocation": prepared.free_memory_before_allocation,
            "free_memory_before_timing": prepared.free_memory_before_timing,
        },
        "compiled_policies": [
            {
                "policy": compiled.policy.model_dump(mode="json"),
                "stablehlo_path": compiled.hlo.stablehlo_path.relative_to(path.parent).as_posix(),
                "stablehlo_sha256": compiled.hlo.stablehlo_sha256,
                "compiler_hlo_path": compiled.hlo.compiler_hlo_path.relative_to(
                    path.parent
                ).as_posix(),
                "compiler_hlo_sha256": compiled.hlo.compiler_hlo_sha256,
                "gmm_scope_labels": compiled.hlo.gmm_scope_labels,
                "gmm_custom_call_counts": compiled.hlo.gmm_custom_call_counts,
                "stablehlo_bytes": compiled.hlo.stablehlo_path.stat().st_size,
                "compiler_hlo_bytes": compiled.hlo.compiler_hlo_path.stat().st_size,
            }
            for compiled in unique.values()
        ],
        "screening_observations": [
            observation.model_dump(mode="json") for observation in observations
        ],
        "limitations": [
            "These are raw execution observations, not correctness evidence.",
            "This runner does not create an immutable receipt.",
            "This runner does not make or authorize a promotion decision.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run(
    *,
    contract_path: Path,
    route_report_path: Path,
    output_root: Path,
) -> Path:
    if output_root.exists():
        _fail("OUTPUT_ROOT_EXISTS", path=output_root)
    contract = _read_contract(contract_path)
    report, report_bytes = _read_route_report(route_report_path)
    try:
        validate_route_corpus_binding(contract, report, report_bytes=report_bytes)
    except ValueError as error:
        _fail("ROUTE_BINDING", error=error)
    source_environment = validate_source_environment(contract)
    execution_target = execution_target_metadata(contract)
    devices = _runtime_devices(contract)
    output_root.mkdir(parents=True)
    hlo_root = output_root / "hlo"
    hlo_root.mkdir()
    prepared = prepare_screen(
        contract,
        report,
        devices=devices,
        hlo_root=hlo_root,
    )
    observations = run_screening(prepared)
    output_path = output_root / "raw-screening-observations.json"
    write_raw_observations(
        output_path,
        prepared,
        observations,
        source_environment=source_environment,
        execution_target=execution_target,
        contract_path=contract_path,
        route_report_path=route_report_path,
    )
    _LOGGER.info("INKLING_GMM_RUNNER_COMPLETE observations=%s", output_path)
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the production-shaped Inkling TPU7x GMM tile screen."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--route-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arguments = _parser().parse_args(argv)
    try:
        output_path = run(
            contract_path=arguments.contract,
            route_report_path=arguments.route_report,
            output_root=arguments.output_root,
        )
    except GmmTileSearchRunnerError:
        return 1
    except Exception:
        _LOGGER.exception("INKLING_GMM_RUNNER_UNHANDLED_FAILURE")
        return 1
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
