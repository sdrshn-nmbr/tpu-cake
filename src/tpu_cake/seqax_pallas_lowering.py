from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, PartitionSpec
from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.parser import Parser

from tpu_cake.dialects.distributed_tensor import DistributedTensor
from tpu_cake.dialects.tpu_schedule import BufferType, MxuEinsumOp, TPUSchedule
from tpu_cake.frontend import canonical_module_text, schedule_sha256
from tpu_cake.jax_lowering import JaxTensorContract, lower_distributed_program_to_jax_mesh
from tpu_cake.seqax_physical_execution import execute_seqax_physical_program_jax
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical

SEQAX_PALLAS_EXECUTION_SCHEMA = "seqax-physical-pallas-v1"


class UnsupportedSeqaxPallasLoweringError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_manifest() -> tuple[tuple[str, str], ...]:
    package = Path(__file__).resolve().parent
    paths = (
        package / "canonical.py",
        package / "dialects" / "distributed_tensor.py",
        package / "dialects" / "tpu_schedule.py",
        package / "frontend.py",
        package / "jax_lowering.py",
        package / "lowering.py",
        package / "seqax_pallas_lowering.py",
        package / "seqax_physical_execution.py",
        package / "seqax_physical_lowering.py",
        package / "source.py",
    )
    return tuple((str(path.relative_to(package.parent)), _sha256(path)) for path in paths)


def _parse_distributed(text: str) -> ModuleOp:
    context = Context()
    context.load_dialect(Builtin)
    context.load_dialect(DistributedTensor)
    return Parser(context, text).parse_module()


def _parse_physical(text: str) -> ModuleOp:
    context = Context()
    context.load_dialect(Builtin)
    context.load_dialect(TPUSchedule)
    return Parser(context, text).parse_module()


def _names(buffer: BufferType) -> tuple[str, ...]:
    return tuple(value.data for value in buffer.shape.dimensions)


def _einsum_tiles(module: ModuleOp) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (operation.tile_m.data, operation.tile_k.data, operation.tile_n.data)
        for operation in module.walk()
        if isinstance(operation, MxuEinsumOp)
    )


def _validate_tpu_tile_shape(
    *,
    m: int,
    k: int,
    n: int,
    tile_m: int,
    tile_k: int,
    tile_n: int,
) -> None:
    if tile_m != m and tile_m % 8:
        raise UnsupportedSeqaxPallasLoweringError(
            "TPU Pallas tile M must span M or be divisible by 8"
        )
    if tile_k != k and tile_k % 128:
        raise UnsupportedSeqaxPallasLoweringError(
            "TPU Pallas tile K must span K or be divisible by 128"
        )
    if tile_n != n and tile_n % 128:
        raise UnsupportedSeqaxPallasLoweringError(
            "TPU Pallas tile N must span N or be divisible by 128"
        )


def _pallas_einsum(
    physical: MxuEinsumOp,
    lhs: jax.Array,
    rhs: jax.Array,
    *,
    interpret: bool,
    schedule_sha256_value: str,
    region_index: int = -1,
) -> jax.Array:
    lhs_type = physical.lhs.type
    rhs_type = physical.rhs.type
    result_type = physical.accumulator.type
    assert isinstance(lhs_type, BufferType)
    assert isinstance(rhs_type, BufferType)
    assert isinstance(result_type, BufferType)
    lhs_names = _names(lhs_type)
    rhs_names = _names(rhs_type)
    result_names = _names(result_type)
    contractions = tuple(value.data for value in physical.contracting_dimensions)
    batch_names = tuple(
        name for name in lhs_names if name in rhs_names and name not in contractions
    )
    lhs_contract = tuple(lhs_names.index(name) for name in contractions)
    rhs_contract = tuple(rhs_names.index(name) for name in contractions)
    lhs_batch = tuple(lhs_names.index(name) for name in batch_names)
    rhs_batch = tuple(rhs_names.index(name) for name in batch_names)
    lhs_free_names = tuple(
        name for name in lhs_names if name not in batch_names and name not in contractions
    )
    rhs_free_names = tuple(
        name for name in rhs_names if name not in batch_names and name not in contractions
    )
    lhs_free = tuple(lhs_names.index(name) for name in lhs_free_names)
    rhs_free = tuple(rhs_names.index(name) for name in rhs_free_names)
    dot_names = (*batch_names, *lhs_free_names, *rhs_free_names)
    if set(dot_names) != set(result_names):
        raise UnsupportedSeqaxPallasLoweringError(
            "physical Pallas einsum cannot reconstruct result dimensions"
        )
    permutation = tuple(dot_names.index(name) for name in result_names)
    expected_lhs = physical.lhs.type.storage.get_shape()
    expected_rhs = physical.rhs.type.storage.get_shape()
    expected_output = physical.accumulator.type.storage.get_shape()
    if tuple(lhs.shape) != expected_lhs or tuple(rhs.shape) != expected_rhs:
        raise UnsupportedSeqaxPallasLoweringError(
            "physical MXU operand shape does not match traced local shard"
        )
    batch_shape = tuple(expected_lhs[index] for index in lhs_batch)
    lhs_free_shape = tuple(expected_lhs[index] for index in lhs_free)
    rhs_free_shape = tuple(expected_rhs[index] for index in rhs_free)
    contraction_shape = tuple(expected_lhs[index] for index in lhs_contract)
    if batch_shape != tuple(expected_rhs[index] for index in rhs_batch):
        raise UnsupportedSeqaxPallasLoweringError("physical MXU batch dimensions do not match")
    if contraction_shape != tuple(expected_rhs[index] for index in rhs_contract):
        raise UnsupportedSeqaxPallasLoweringError(
            "physical MXU contraction dimensions do not match"
        )
    batch_size = math.prod(batch_shape)
    lhs_free_size = math.prod(lhs_free_shape)
    rhs_free_size = math.prod(rhs_free_shape)
    contraction_size = math.prod(contraction_shape)
    lhs_permutation = (*lhs_batch, *lhs_free, *lhs_contract)
    rhs_permutation = (*rhs_batch, *rhs_contract, *rhs_free)
    tile_m = physical.tile_m.data
    tile_k = physical.tile_k.data
    tile_n = physical.tile_n.data
    if lhs_free_size % tile_m or contraction_size % tile_k or rhs_free_size % tile_n:
        raise UnsupportedSeqaxPallasLoweringError(
            "physical MXU tiles must divide the flattened local contraction"
        )
    _validate_tpu_tile_shape(
        m=lhs_free_size,
        k=contraction_size,
        n=rhs_free_size,
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
    )
    k_steps = contraction_size // tile_k

    lhs_value = jnp.transpose(lhs, lhs_permutation).reshape(
        (batch_size, lhs_free_size, contraction_size)
    )
    rhs_value = jnp.transpose(rhs, rhs_permutation).reshape(
        (batch_size, contraction_size, rhs_free_size)
    )

    def kernel(lhs_ref, rhs_ref, output_ref, accumulator_ref) -> None:
        @pl.when(pl.program_id(3) == 0)
        def initialize() -> None:
            accumulator_ref[...] = jnp.zeros_like(accumulator_ref)

        accumulator_ref[...] = accumulator_ref[...] + jnp.dot(
            lhs_ref[0, ...],
            rhs_ref[0, ...],
            preferred_element_type=jnp.float32,
        )

        @pl.when(pl.program_id(3) == k_steps - 1)
        def store() -> None:
            output_ref[0, ...] = accumulator_ref[...]

    interpret_setting = (
        pltpu.InterpretParams(detect_races=True, out_of_bounds_reads="raise")
        if interpret
        else False
    )
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(
            (batch_size, lhs_free_size, rhs_free_size),
            jnp.float32,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(
                pl.BlockSpec(
                    (1, tile_m, tile_k),
                    lambda batch, m, _n, k: (batch, m, k),
                ),
                pl.BlockSpec(
                    (1, tile_k, tile_n),
                    lambda batch, _m, n, k: (batch, k, n),
                ),
            ),
            out_specs=pl.BlockSpec(
                (1, tile_m, tile_n),
                lambda batch, m, n, _k: (batch, m, n),
            ),
            grid=(
                batch_size,
                lhs_free_size // tile_m,
                rhs_free_size // tile_n,
                k_steps,
            ),
            scratch_shapes=(pltpu.VMEM((tile_m, tile_n), jnp.float32),),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
        ),
        interpret=interpret_setting,
        name="seqax_named_einsum",
        metadata={
            "schedule_sha256": schedule_sha256_value,
            "region_index": region_index,
            "tile_m": tile_m,
            "tile_k": tile_k,
            "tile_n": tile_n,
        },
    )
    value = call(lhs_value, rhs_value)
    value = value.reshape((*batch_shape, *lhs_free_shape, *rhs_free_shape))
    if permutation != tuple(range(len(permutation))):
        value = jnp.transpose(value, permutation)
    if tuple(value.shape) != expected_output:
        raise UnsupportedSeqaxPallasLoweringError(
            "physical MXU tiled result does not match the declared accumulator"
        )
    return value


@dataclass(frozen=True)
class SeqaxPallasPlan:
    name: str
    canonical_distributed_xdsl: str
    canonical_physical_xdsl: str
    distributed_schedule_sha256: str
    physical_schedule_sha256: str
    mesh: tuple[tuple[str, int], ...]
    input_contracts: tuple[JaxTensorContract, ...]
    output_contracts: tuple[JaxTensorContract, ...]
    pallas_region_count: int
    implementation_manifest: tuple[tuple[str, str], ...]

    @property
    def schema(self) -> str:
        return SEQAX_PALLAS_EXECUTION_SCHEMA

    @property
    def device_count(self) -> int:
        return math.prod(size for _, size in self.mesh)

    @property
    def uses_pallas(self) -> bool:
        return True

    @property
    def execution_scope(self) -> str:
        return "multi-device-local-shards-with-pallas-einsums"

    def _validated_modules(self) -> tuple[ModuleOp, ModuleOp]:
        distributed = _parse_distributed(self.canonical_distributed_xdsl)
        physical = _parse_physical(self.canonical_physical_xdsl)
        if schedule_sha256(distributed) != self.distributed_schedule_sha256:
            raise UnsupportedSeqaxPallasLoweringError("distributed schedule hash mismatch")
        if schedule_sha256(physical) != self.physical_schedule_sha256:
            raise UnsupportedSeqaxPallasLoweringError("physical schedule hash mismatch")
        regenerated = lower_seqax_forward_to_physical(
            distributed,
            einsum_tiles=_einsum_tiles(physical),
        )
        if canonical_module_text(regenerated.module) != self.canonical_physical_xdsl:
            raise UnsupportedSeqaxPallasLoweringError(
                "physical schedule is not the canonical lowering of the distributed program"
            )
        jax_plan = lower_distributed_program_to_jax_mesh(distributed)
        expected_mesh = tuple(sorted(jax_plan.mesh.items()))
        if self.mesh != expected_mesh:
            raise UnsupportedSeqaxPallasLoweringError(
                "Pallas plan mesh does not match the distributed program"
            )
        if self.input_contracts != jax_plan.input_contracts:
            raise UnsupportedSeqaxPallasLoweringError(
                "Pallas plan inputs do not match the distributed program"
            )
        if self.output_contracts != jax_plan.output_contracts:
            raise UnsupportedSeqaxPallasLoweringError(
                "Pallas plan outputs do not match the distributed program"
            )
        expected_regions = sum(isinstance(operation, MxuEinsumOp) for operation in physical.walk())
        if self.pallas_region_count != expected_regions:
            raise UnsupportedSeqaxPallasLoweringError(
                "Pallas region count does not match the physical schedule"
            )
        if self.implementation_manifest != _implementation_manifest():
            raise UnsupportedSeqaxPallasLoweringError(
                "Pallas implementation source manifest does not match the runtime"
            )
        return distributed, physical

    def _build_mapped(
        self,
        *,
        interpret: bool,
        devices,
        strict_silu_layers: int,
        checkpoint_spec: PartitionSpec | None,
    ):
        _distributed, physical = self._validated_modules()
        selected_devices = tuple(devices or jax.devices())
        if len(selected_devices) != self.device_count:
            raise ValueError(
                f"Seqax Pallas plan needs exactly {self.device_count} devices, "
                f"found {len(selected_devices)}"
            )
        axis_names = tuple(axis for axis, _ in self.mesh)
        axis_sizes = tuple(size for _, size in self.mesh)
        mesh = Mesh(
            np.asarray(selected_devices, dtype=object).reshape(axis_sizes),
            axis_names,
        )
        physical_einsums = tuple(
            operation for operation in physical.walk() if isinstance(operation, MxuEinsumOp)
        )
        if len(physical_einsums) != self.pallas_region_count:
            raise UnsupportedSeqaxPallasLoweringError(
                "physical Pallas region count does not match the plan"
            )

        def physical_call(*inputs):
            index = 0
            checkpoints: list[tuple[jax.Array, jax.Array]] | None = (
                [] if strict_silu_layers else None
            )

            def einsum(
                physical_operation: MxuEinsumOp,
                lhs: jax.Array,
                rhs: jax.Array,
            ) -> jax.Array:
                nonlocal index
                if index >= len(physical_einsums):
                    raise UnsupportedSeqaxPallasLoweringError(
                        "distributed program executed more einsums than the physical schedule"
                    )
                region_index = index
                expected_operation = physical_einsums[region_index]
                index += 1
                if physical_operation is not expected_operation:
                    raise UnsupportedSeqaxPallasLoweringError(
                        "physical execution changed the Pallas region order"
                    )
                return _pallas_einsum(
                    physical_operation,
                    lhs,
                    rhs,
                    interpret=interpret,
                    schedule_sha256_value=self.physical_schedule_sha256,
                    region_index=region_index,
                )

            outputs = execute_seqax_physical_program_jax(
                physical,
                inputs,
                einsum=einsum,
                strict_silu_checkpoints=checkpoints,
            )
            if index != len(physical_einsums):
                raise UnsupportedSeqaxPallasLoweringError(
                    "physical schedule contains unused Pallas einsum regions"
                )
            if checkpoints is None:
                return outputs
            if len(checkpoints) != strict_silu_layers:
                raise ValueError(
                    f"strict SiLU expected {strict_silu_layers} checkpoints, "
                    f"found {len(checkpoints)}"
                )
            return (*outputs, *(value for checkpoint in checkpoints for value in checkpoint))

        output_specs = tuple(contract.partition_spec() for contract in self.output_contracts)
        if strict_silu_layers:
            if checkpoint_spec is None:
                raise ValueError("strict SiLU checkpoint sharding must be explicit")
            output_specs += (checkpoint_spec,) * (2 * strict_silu_layers)

        mapped = jax.shard_map(
            physical_call,
            mesh=mesh,
            in_specs=tuple(contract.partition_spec() for contract in self.input_contracts),
            out_specs=output_specs,
            check_vma=False,
        )
        return mapped, mesh

    def build_mapped(self, *, interpret: bool = False, devices=None):
        return self._build_mapped(
            interpret=interpret,
            devices=devices,
            strict_silu_layers=0,
            checkpoint_spec=None,
        )

    def build(self, *, interpret: bool = False, devices=None):
        mapped, mesh = self.build_mapped(interpret=interpret, devices=devices)
        return jax.jit(mapped), mesh

    def build_with_strict_silu_checkpoints(
        self,
        *,
        expected_layers: int,
        checkpoint_spec: PartitionSpec,
        interpret: bool = False,
        devices=None,
    ):
        if expected_layers <= 0:
            raise ValueError("strict SiLU checkpoint count must be positive")
        mapped, mesh = self._build_mapped(
            interpret=interpret,
            devices=devices,
            strict_silu_layers=expected_layers,
            checkpoint_spec=checkpoint_spec,
        )
        return jax.jit(mapped), mesh

    def manifest(self) -> dict[str, Any]:
        mesh = dict(self.mesh)
        return {
            "schema": self.schema,
            "name": self.name,
            "distributed_schedule_sha256": self.distributed_schedule_sha256,
            "physical_schedule_sha256": self.physical_schedule_sha256,
            "mesh": mesh,
            "device_count": self.device_count,
            "execution_scope": self.execution_scope,
            "uses_pallas": self.uses_pallas,
            "pallas_region_count": self.pallas_region_count,
            "implementation_manifest": dict(self.implementation_manifest),
            "input_contracts": [contract.manifest(mesh=mesh) for contract in self.input_contracts],
            "output_contracts": [
                contract.manifest(mesh=mesh) for contract in self.output_contracts
            ],
        }

    def render_executable_source(self) -> str:
        return f"""from __future__ import annotations

from tpu_cake.jax_lowering import JaxTensorContract
from tpu_cake.seqax_pallas_lowering import SeqaxPallasPlan

PLAN = SeqaxPallasPlan(
    name={self.name!r},
    canonical_distributed_xdsl={self.canonical_distributed_xdsl!r},
    canonical_physical_xdsl={self.canonical_physical_xdsl!r},
    distributed_schedule_sha256={self.distributed_schedule_sha256!r},
    physical_schedule_sha256={self.physical_schedule_sha256!r},
    mesh={self.mesh!r},
    input_contracts={self.input_contracts!r},
    output_contracts={self.output_contracts!r},
    pallas_region_count={self.pallas_region_count},
    implementation_manifest={self.implementation_manifest!r},
)


def build(*, interpret=False, devices=None):
    return PLAN.build(interpret=interpret, devices=devices)
"""

    def source_sha256(self) -> str:
        return hashlib.sha256(self.render_executable_source().encode()).hexdigest()


def lower_seqax_physical_to_pallas(
    distributed: ModuleOp,
    physical: ModuleOp,
) -> SeqaxPallasPlan:
    distributed.verify()
    physical.verify()
    regenerated = lower_seqax_forward_to_physical(
        distributed,
        einsum_tiles=_einsum_tiles(physical),
    )
    canonical_physical = canonical_module_text(physical)
    if canonical_module_text(regenerated.module) != canonical_physical:
        raise UnsupportedSeqaxPallasLoweringError(
            "physical schedule is not the canonical lowering of the distributed program"
        )
    jax_plan = lower_distributed_program_to_jax_mesh(distributed)
    pallas_regions = tuple(
        operation for operation in physical.walk() if isinstance(operation, MxuEinsumOp)
    )
    if not pallas_regions:
        raise UnsupportedSeqaxPallasLoweringError(
            "Seqax physical schedule contains no Pallas regions"
        )
    return SeqaxPallasPlan(
        name=f"{jax_plan.name}_physical_pallas",
        canonical_distributed_xdsl=canonical_module_text(distributed),
        canonical_physical_xdsl=canonical_physical,
        distributed_schedule_sha256=schedule_sha256(distributed),
        physical_schedule_sha256=schedule_sha256(physical),
        mesh=tuple(sorted(jax_plan.mesh.items())),
        input_contracts=jax_plan.input_contracts,
        output_contracts=jax_plan.output_contracts,
        pallas_region_count=len(pallas_regions),
        implementation_manifest=_implementation_manifest(),
    )
