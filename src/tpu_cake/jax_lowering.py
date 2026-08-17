from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh, PartitionSpec
from xdsl.context import Context
from xdsl.dialects.builtin import (
    BFloat16Type,
    Builtin,
    Float16Type,
    Float32Type,
    IntegerType,
    ModuleOp,
    Signedness,
)
from xdsl.ir import Operation, SSAValue
from xdsl.parser import Parser

from tpu_cake.dialects.distributed_tensor import (
    AllGatherOp,
    AllReduceOp,
    BroadcastOp,
    CastOp,
    DistributedTensor,
    DTensorType,
    EinsumLocalOp,
    EinsumOp,
    ElementwiseOp,
    EmbeddingLookupOp,
    LayerScanOp,
    MaskedSoftmaxOp,
    PackedCausalMaskOp,
    ProgramOp,
    ReduceLocalOp,
    ReduceScatterOp,
    RenameDimensionOp,
    ReturnOp,
    RmsNormOp,
    RotaryEmbeddingOp,
    ScanYieldOp,
    SliceOp,
    TransposeOp,
)
from tpu_cake.dtensor_interpreter import (
    execute_distributed_program_jax,
    execute_distributed_program_jax_sharded,
)
from tpu_cake.frontend import canonical_module_text, schedule_sha256

JAX_LOGICAL_EXECUTION_SCHEMA = "dtensor-logical-jax-v1"
JAX_DISTRIBUTED_EXECUTION_SCHEMA = "dtensor-shard-map-jax-v1"


class UnsupportedJaxLoweringError(ValueError):
    pass


_SUPPORTED_OPERATIONS = (
    ProgramOp,
    ElementwiseOp,
    CastOp,
    RmsNormOp,
    RotaryEmbeddingOp,
    SliceOp,
    RenameDimensionOp,
    PackedCausalMaskOp,
    MaskedSoftmaxOp,
    ReduceLocalOp,
    TransposeOp,
    BroadcastOp,
    EmbeddingLookupOp,
    EinsumLocalOp,
    EinsumOp,
    AllGatherOp,
    ReduceScatterOp,
    AllReduceOp,
    LayerScanOp,
    ScanYieldOp,
    ReturnOp,
)
_SUPPORTED_ELEMENTWISE_FUNCTIONS = {"add", "multiply", "silu", "exp"}
_COLLECTIVE_OPERATIONS = (AllGatherOp, ReduceScatterOp, AllReduceOp)


def _reject(operation: Operation, message: str) -> UnsupportedJaxLoweringError:
    return UnsupportedJaxLoweringError(f"{message}: {operation.name} at {operation.location}")


def _program(module: ModuleOp) -> ProgramOp:
    top_level = tuple(module.body.block.ops)
    if len(top_level) != 1 or not isinstance(top_level[0], ProgramOp):
        operation = top_level[0] if top_level else module
        raise _reject(operation, "logical JAX lowering expects one distributed program")
    return top_level[0]


def _dtype_name(value_type: DTensorType) -> str:
    element_type = value_type.element_type
    if isinstance(element_type, BFloat16Type):
        return "bfloat16"
    if isinstance(element_type, Float16Type):
        return "float16"
    if isinstance(element_type, Float32Type):
        return "float32"
    if isinstance(element_type, IntegerType):
        width = element_type.width.data
        if width in {1, 8, 16, 32, 64}:
            if width == 1:
                return "bool"
            prefix = "u" if element_type.signedness.data is Signedness.UNSIGNED else ""
            return f"{prefix}int{width}"
    raise UnsupportedJaxLoweringError(f"unsupported logical JAX element type {element_type}")


@dataclass(frozen=True)
class JaxTensorContract:
    dtype: str
    shape: tuple[tuple[str, int], ...]
    declared_sharding: tuple[tuple[str, ...], ...]
    pending_reductions: tuple[tuple[str, str], ...]

    def local_shape(self, mesh: dict[str, int]) -> tuple[int, ...]:
        result = []
        for (_, size), axes in zip(self.shape, self.declared_sharding, strict=True):
            divisor = 1
            for axis in axes:
                divisor *= mesh[axis]
            result.append(size // divisor)
        return tuple(result)

    def partition_spec(self) -> PartitionSpec:
        entries = tuple(
            None if not axes else axes[0] if len(axes) == 1 else axes
            for axes in self.declared_sharding
        )
        return PartitionSpec(*entries)

    def manifest(self, *, mesh: dict[str, int] | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "dtype": self.dtype,
            "shape": [{"dimension": name, "size": size} for name, size in self.shape],
            "declared_sharding": [list(axes) for axes in self.declared_sharding],
            "partition_spec": [
                None if not axes else axes[0] if len(axes) == 1 else list(axes)
                for axes in self.declared_sharding
            ],
            "pending_reductions": dict(self.pending_reductions),
        }
        if mesh is not None:
            result["local_shape"] = list(self.local_shape(mesh))
        return result


def _tensor_contract(value: SSAValue) -> JaxTensorContract:
    value_type = value.type
    assert isinstance(value_type, DTensorType)
    return JaxTensorContract(
        dtype=_dtype_name(value_type),
        shape=value_type.logical_shape(),
        declared_sharding=value_type.sharding_axes(),
        pending_reductions=tuple(value_type.pending_reductions().items()),
    )


def _all_values(operation: Operation) -> tuple[SSAValue, ...]:
    values = [*operation.operands, *operation.results]
    for region in operation.regions:
        for block in region.blocks:
            values.extend(block.args)
    return tuple(values)


def _validate_supported_program(module: ModuleOp) -> ProgramOp:
    module.verify()
    program = _program(module)
    for operation in module.walk():
        if isinstance(operation, ModuleOp):
            continue
        if not isinstance(operation, _SUPPORTED_OPERATIONS):
            raise _reject(operation, "no logical JAX lowering for operation")
        for value in _all_values(operation):
            if not isinstance(value.type, DTensorType):
                raise _reject(operation, "logical JAX lowering requires distributed tensor values")
            try:
                _dtype_name(value.type)
            except UnsupportedJaxLoweringError as error:
                raise _reject(operation, str(error)) from error
        if isinstance(operation, ElementwiseOp):
            function = operation.function.data
            if function not in _SUPPORTED_ELEMENTWISE_FUNCTIONS:
                raise _reject(
                    operation,
                    f"no logical JAX lowering for elementwise function {function!r}",
                )
        if isinstance(operation, RotaryEmbeddingOp):
            value_type = operation.value.type
            assert isinstance(value_type, DTensorType)
            if value_type.logical_shape()[-1][0] != operation.head_dimension.data:
                raise _reject(operation, "logical JAX RoPE requires the head dimension last")
        if isinstance(operation, PackedCausalMaskOp):
            value_type = operation.sequence_starts.type
            assert isinstance(value_type, DTensorType)
            names = tuple(name for name, _ in value_type.logical_shape())
            if len(names) != 2 or names[1] != operation.sequence_dimension.data:
                raise _reject(
                    operation,
                    "logical JAX packed mask requires [batch, sequence] inputs",
                )
    return program


def _parse_canonical_module(text: str) -> ModuleOp:
    context = Context()
    context.load_dialect(Builtin)
    context.load_dialect(DistributedTensor)
    return Parser(context, text).parse_module()


@dataclass(frozen=True)
class JaxDistributedProgramPlan:
    """Replayable global-logical execution of verified distributed tensor xDSL.

    This is an executable semantic backend, not a physical schedule. It uses one JAX
    device, stores every tensor at its global logical shape, and therefore evaluates
    collectives as value-preserving identities. It does not use Pallas and makes no
    claim about physical sharding, communication, or kernel quality.
    """

    name: str
    canonical_xdsl: str
    schedule_sha256: str
    operation_inventory: tuple[tuple[str, int], ...]
    input_contracts: tuple[JaxTensorContract, ...]
    output_contracts: tuple[JaxTensorContract, ...]

    @property
    def schema(self) -> str:
        return JAX_LOGICAL_EXECUTION_SCHEMA

    @property
    def execution_scope(self) -> str:
        return "one-device-global-logical-tensors"

    @property
    def collective_semantics(self) -> str:
        return "global-value-identity"

    @property
    def uses_physical_collectives(self) -> bool:
        return False

    @property
    def uses_pallas(self) -> bool:
        return False

    @property
    def operation_counts(self) -> dict[str, int]:
        return dict(self.operation_inventory)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "name": self.name,
            "schedule_sha256": self.schedule_sha256,
            "execution_scope": self.execution_scope,
            "collective_semantics": self.collective_semantics,
            "uses_physical_collectives": self.uses_physical_collectives,
            "uses_pallas": self.uses_pallas,
            "operation_counts": dict(self.operation_inventory),
            "input_contracts": [value.manifest() for value in self.input_contracts],
            "output_contracts": [value.manifest() for value in self.output_contracts],
        }

    def build(self, *, device: jax.Device | None = None):
        module = _parse_canonical_module(self.canonical_xdsl)
        replayed = lower_distributed_program_to_jax(module)
        if replayed.schedule_sha256 != self.schedule_sha256:
            raise UnsupportedJaxLoweringError(
                "replayed logical JAX schedule does not match the plan hash"
            )

        def execute(*inputs):
            return execute_distributed_program_jax(module, inputs)

        compiled = jax.jit(execute)
        if device is None:
            return compiled

        def execute_on_device(*inputs):
            placed = tuple(jax.device_put(value, device) for value in inputs)
            return compiled(*placed)

        return execute_on_device

    def render_executable_source(self) -> str:
        return f"""from __future__ import annotations

from tpu_cake.jax_lowering import load_jax_distributed_plan

JAX_LOGICAL_EXECUTION_SCHEMA = {JAX_LOGICAL_EXECUTION_SCHEMA!r}
EXPECTED_SCHEDULE_SHA256 = {self.schedule_sha256!r}
CANONICAL_XDSL = {self.canonical_xdsl!r}

PLAN = load_jax_distributed_plan(
    CANONICAL_XDSL,
    expected_schedule_sha256=EXPECTED_SCHEDULE_SHA256,
)


def build(*, device=None):
    return PLAN.build(device=device)
"""

    def source_sha256(self) -> str:
        return hashlib.sha256(self.render_executable_source().encode()).hexdigest()


def _validate_physical_supported_program(module: ModuleOp) -> ProgramOp:
    program = _validate_supported_program(module)
    for operation in module.walk():
        if isinstance(operation, ReduceScatterOp):
            if operation.reducer.data != "sum":
                raise _reject(
                    operation,
                    "JAX/XLA distributed lowering supports sum reduce-scatter only",
                )
            if len(operation.scatter_dimensions) != 1:
                raise _reject(
                    operation,
                    "JAX/XLA distributed lowering supports one scatter dimension",
                )
    return program


@dataclass(frozen=True)
class JaxDistributedMeshPlan:
    """Replayable multi-device JAX/XLA lowering of distributed tensor xDSL.

    Operations execute on local shards inside ``jax.shard_map``. The plan uses real
    JAX collectives and exact input/output PartitionSpecs. It is not a hand-authored
    Pallas schedule and does not claim control over XLA's generated kernels.
    """

    name: str
    canonical_xdsl: str
    schedule_sha256: str
    mesh_axes: tuple[tuple[str, int], ...]
    operation_inventory: tuple[tuple[str, int], ...]
    input_contracts: tuple[JaxTensorContract, ...]
    output_contracts: tuple[JaxTensorContract, ...]

    @property
    def schema(self) -> str:
        return JAX_DISTRIBUTED_EXECUTION_SCHEMA

    @property
    def execution_scope(self) -> str:
        return "multi-device-local-shards"

    @property
    def collective_semantics(self) -> str:
        return "jax-lax-physical-collectives"

    @property
    def uses_physical_collectives(self) -> bool:
        return True

    @property
    def uses_pallas(self) -> bool:
        return False

    @property
    def mesh(self) -> dict[str, int]:
        return dict(self.mesh_axes)

    @property
    def device_count(self) -> int:
        count = 1
        for _, size in self.mesh_axes:
            count *= size
        return count

    @property
    def operation_counts(self) -> dict[str, int]:
        return dict(self.operation_inventory)

    @property
    def input_partition_specs(self) -> tuple[PartitionSpec, ...]:
        return tuple(value.partition_spec() for value in self.input_contracts)

    @property
    def output_partition_specs(self) -> tuple[PartitionSpec, ...]:
        return tuple(value.partition_spec() for value in self.output_contracts)

    def manifest(self) -> dict[str, Any]:
        mesh = self.mesh
        return {
            "schema": self.schema,
            "name": self.name,
            "schedule_sha256": self.schedule_sha256,
            "execution_scope": self.execution_scope,
            "collective_semantics": self.collective_semantics,
            "uses_physical_collectives": self.uses_physical_collectives,
            "uses_pallas": self.uses_pallas,
            "mesh": mesh,
            "device_count": self.device_count,
            "operation_counts": dict(self.operation_inventory),
            "input_contracts": [value.manifest(mesh=mesh) for value in self.input_contracts],
            "output_contracts": [value.manifest(mesh=mesh) for value in self.output_contracts],
        }

    def build_mapped(self, *, devices=None):
        selected_devices = tuple(devices or jax.devices())
        if len(selected_devices) != self.device_count:
            raise ValueError(
                f"distributed JAX plan needs exactly {self.device_count} devices for "
                f"mesh {self.mesh}, found {len(selected_devices)}"
            )
        module = _parse_canonical_module(self.canonical_xdsl)
        replayed = lower_distributed_program_to_jax_mesh(module)
        if replayed.schedule_sha256 != self.schedule_sha256:
            raise UnsupportedJaxLoweringError(
                "replayed distributed JAX schedule does not match the plan hash"
            )
        axis_names = tuple(axis for axis, _ in self.mesh_axes)
        axis_shape = tuple(size for _, size in self.mesh_axes)
        mesh = Mesh(np.asarray(selected_devices, dtype=object).reshape(axis_shape), axis_names)

        def execute(*inputs):
            return execute_distributed_program_jax_sharded(module, inputs)

        mapped = jax.shard_map(
            execute,
            mesh=mesh,
            in_specs=self.input_partition_specs,
            out_specs=self.output_partition_specs,
            check_vma=False,
        )
        return mapped, mesh

    def build(self, *, devices=None):
        mapped, mesh = self.build_mapped(devices=devices)
        return jax.jit(mapped), mesh

    def render_executable_source(self) -> str:
        return f"""from __future__ import annotations

from tpu_cake.jax_lowering import load_jax_distributed_mesh_plan

JAX_DISTRIBUTED_EXECUTION_SCHEMA = {JAX_DISTRIBUTED_EXECUTION_SCHEMA!r}
EXPECTED_SCHEDULE_SHA256 = {self.schedule_sha256!r}
CANONICAL_XDSL = {self.canonical_xdsl!r}

PLAN = load_jax_distributed_mesh_plan(
    CANONICAL_XDSL,
    expected_schedule_sha256=EXPECTED_SCHEDULE_SHA256,
)


def build(*, devices=None):
    return PLAN.build(devices=devices)
"""

    def source_sha256(self) -> str:
        return hashlib.sha256(self.render_executable_source().encode()).hexdigest()


def lower_distributed_program_to_jax(module: ModuleOp) -> JaxDistributedProgramPlan:
    _validate_supported_program(module)
    canonical_xdsl = canonical_module_text(module)
    canonical_module = _parse_canonical_module(canonical_xdsl)
    program = _validate_supported_program(canonical_module)
    counts = Counter(operation.name for operation in canonical_module.walk())
    outputs = tuple(program.body.block.last_op.values)
    assert isinstance(program.body.block.last_op, ReturnOp)
    return JaxDistributedProgramPlan(
        name=program.sym_name.data,
        canonical_xdsl=canonical_xdsl,
        schedule_sha256=schedule_sha256(canonical_module),
        operation_inventory=tuple(sorted(counts.items())),
        input_contracts=tuple(_tensor_contract(value) for value in program.body.block.args),
        output_contracts=tuple(_tensor_contract(value) for value in outputs),
    )


def load_jax_distributed_plan(
    canonical_xdsl: str,
    *,
    expected_schedule_sha256: str,
) -> JaxDistributedProgramPlan:
    module = _parse_canonical_module(canonical_xdsl)
    plan = lower_distributed_program_to_jax(module)
    if plan.canonical_xdsl != canonical_xdsl:
        raise UnsupportedJaxLoweringError("logical JAX replay source is not canonical xDSL")
    if plan.schedule_sha256 != expected_schedule_sha256:
        raise UnsupportedJaxLoweringError(
            "logical JAX schedule hash mismatch: "
            f"expected {expected_schedule_sha256}, got {plan.schedule_sha256}"
        )
    return plan


def lower_distributed_program_to_jax_mesh(module: ModuleOp) -> JaxDistributedMeshPlan:
    _validate_physical_supported_program(module)
    canonical_xdsl = canonical_module_text(module)
    canonical_module = _parse_canonical_module(canonical_xdsl)
    program = _validate_physical_supported_program(canonical_module)
    counts = Counter(operation.name for operation in canonical_module.walk())
    outputs = tuple(program.body.block.last_op.values)
    assert isinstance(program.body.block.last_op, ReturnOp)
    return JaxDistributedMeshPlan(
        name=program.sym_name.data,
        canonical_xdsl=canonical_xdsl,
        schedule_sha256=schedule_sha256(canonical_module),
        mesh_axes=tuple(program.mesh.sizes().items()),
        operation_inventory=tuple(sorted(counts.items())),
        input_contracts=tuple(_tensor_contract(value) for value in program.body.block.args),
        output_contracts=tuple(_tensor_contract(value) for value in outputs),
    )


def load_jax_distributed_mesh_plan(
    canonical_xdsl: str,
    *,
    expected_schedule_sha256: str,
) -> JaxDistributedMeshPlan:
    module = _parse_canonical_module(canonical_xdsl)
    plan = lower_distributed_program_to_jax_mesh(module)
    if plan.canonical_xdsl != canonical_xdsl:
        raise UnsupportedJaxLoweringError("distributed JAX replay source is not canonical xDSL")
    if plan.schedule_sha256 != expected_schedule_sha256:
        raise UnsupportedJaxLoweringError(
            "distributed JAX schedule hash mismatch: "
            f"expected {expected_schedule_sha256}, got {plan.schedule_sha256}"
        )
    return plan
