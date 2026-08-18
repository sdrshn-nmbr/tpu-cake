from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue

from tpu_cake.cost_model import HardwareRateModel, tpu7x_tensorcore_rates
from tpu_cake.dialects.tpu_schedule import (
    AllocOp,
    BufferType,
    CollectiveOp,
    DmaStartOp,
    KernelOp,
    MxuEinsumOp,
    VectorComputeOp,
    ViewOp,
    buffer_bytes,
)
from tpu_cake.frontend import schedule_sha256
from tpu_cake.physical_cost_model import (
    PhysicalKernelResourceReport,
    analyze_physical_kernel,
)
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.seqax_surface import seqax_forward_workload_surface
from tpu_cake.workloads.seqax_forward import (
    SeqaxFeedForwardFusion,
    SeqaxNumericalSemantics,
    seqax_forward_schedule,
)

PHYSICAL_FUSION_COMPARISON_SCHEMA = "physical-fusion-comparison-v1"
SEQAX_SILU_MULTIPLY_FUSION_SCHEMA = "seqax-silu-multiply-fusion-v1"


class UnsupportedPhysicalFusionError(ValueError):
    pass


class PhysicalFusionRewrite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    baseline_silu_operation_id: str = Field(min_length=1)
    baseline_multiply_operation_id: str = Field(min_length=1)
    candidate_fused_operation_id: str = Field(min_length=1)
    eliminated_intermediate_vmem_bytes_per_device: int = Field(gt=0)


class PhysicalFusionComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^physical-fusion-comparison-v1$")
    transformation: str = Field(pattern=r"^silu_multiply$")
    baseline: PhysicalKernelResourceReport
    candidate: PhysicalKernelResourceReport
    rewrites: tuple[PhysicalFusionRewrite, ...] = Field(min_length=1)
    eliminated_vector_operations: int = Field(gt=0)
    eliminated_intermediate_vmem_bytes_per_device: int = Field(gt=0)
    allocated_vmem_savings_bytes_per_device: int = Field(gt=0)
    peak_live_vmem_savings_bytes_per_device: int = Field(ge=0)
    declared_work_is_equal: bool
    candidate_is_static_resource_preference: bool
    measured_performance_winner: None = None
    predictive_validation: bool = False
    assumptions: tuple[str, ...]
    omissions: tuple[str, ...]

    @model_validator(mode="after")
    def deltas_are_consistent(self) -> PhysicalFusionComparison:
        rewrite_count = len(self.rewrites)
        if self.baseline.physical_schedule_sha256 == self.candidate.physical_schedule_sha256:
            raise ValueError("PHYSICAL_FUSION_SCHEDULES_MUST_DIFFER")
        if self.eliminated_vector_operations != rewrite_count:
            raise ValueError("PHYSICAL_FUSION_OPERATION_DELTA_MISMATCH")
        intermediate_bytes = sum(
            value.eliminated_intermediate_vmem_bytes_per_device for value in self.rewrites
        )
        if self.eliminated_intermediate_vmem_bytes_per_device != intermediate_bytes:
            raise ValueError("PHYSICAL_FUSION_INTERMEDIATE_BYTES_MISMATCH")
        allocated_savings = (
            self.baseline.memory.allocated_vmem_bytes_per_device
            - self.candidate.memory.allocated_vmem_bytes_per_device
        )
        peak_savings = (
            self.baseline.memory.peak_live_vmem_bytes_per_device
            - self.candidate.memory.peak_live_vmem_bytes_per_device
        )
        if (
            self.allocated_vmem_savings_bytes_per_device != allocated_savings
            or allocated_savings != intermediate_bytes
            or self.peak_live_vmem_savings_bytes_per_device != peak_savings
            or peak_savings < 0
        ):
            raise ValueError("PHYSICAL_FUSION_MEMORY_DELTA_MISMATCH")
        if not self.declared_work_is_equal:
            raise ValueError("PHYSICAL_FUSION_DECLARED_WORK_MUST_MATCH")
        expected_preference = allocated_savings > 0 and peak_savings >= 0
        if self.candidate_is_static_resource_preference != expected_preference:
            raise ValueError("PHYSICAL_FUSION_STATIC_PREFERENCE_MISMATCH")
        if self.predictive_validation:
            raise ValueError("PHYSICAL_FUSION_HAS_NO_PREDICTIVE_VALIDATION")
        return self


class SeqaxSiluMultiplyFusionContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^seqax-silu-multiply-fusion-v1$")
    surface_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_names: tuple[str, ...] = Field(min_length=1)
    target: str = Field(pattern=r"^tpu7x$")
    numerical_semantics: SeqaxNumericalSemantics
    baseline: SeqaxFeedForwardFusion
    candidate: SeqaxFeedForwardFusion

    @model_validator(mode="after")
    def is_canonical(self) -> SeqaxSiluMultiplyFusionContract:
        surface = seqax_forward_workload_surface()
        expected = (
            SEQAX_SILU_MULTIPLY_FUSION_SCHEMA,
            surface.surface_id,
            tuple(scenario.name for scenario in surface.scenarios),
            "tpu7x",
            SeqaxNumericalSemantics.LEGACY_FUSED_V0,
            SeqaxFeedForwardFusion.SEPARATE,
            SeqaxFeedForwardFusion.SILU_MULTIPLY,
        )
        observed = (
            self.schema_version,
            self.surface_id,
            self.scenario_names,
            self.target,
            self.numerical_semantics,
            self.baseline,
            self.candidate,
        )
        if observed != expected:
            raise ValueError("SEQAX_SILU_MULTIPLY_FUSION_CONTRACT_NOT_CANONICAL")
        return self

    @computed_field
    @property
    def contract_id(self) -> str:
        payload = self.model_dump(mode="json", exclude_computed_fields=True)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class SeqaxFusionScenarioComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_name: str = Field(min_length=1)
    baseline_distributed_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_distributed_schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    comparison: PhysicalFusionComparison


class SeqaxSiluMultiplyFusionReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^seqax-silu-multiply-fusion-v1$")
    contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    surface_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenarios: tuple[SeqaxFusionScenarioComparison, ...] = Field(min_length=1)
    semantic_authority: str = Field(pattern=r"^canonical_seqax_regeneration$")
    measured_performance_winner: None = None
    predictive_validation: bool = False

    @model_validator(mode="after")
    def remains_static(self) -> SeqaxSiluMultiplyFusionReport:
        if self.predictive_validation:
            raise ValueError("SEQAX_FUSION_REPORT_HAS_NO_PREDICTIVE_VALIDATION")
        return self


def default_seqax_silu_multiply_fusion_contract() -> SeqaxSiluMultiplyFusionContract:
    surface = seqax_forward_workload_surface()
    return SeqaxSiluMultiplyFusionContract(
        schema_version=SEQAX_SILU_MULTIPLY_FUSION_SCHEMA,
        surface_id=surface.surface_id,
        scenario_names=tuple(scenario.name for scenario in surface.scenarios),
        target="tpu7x",
        numerical_semantics=SeqaxNumericalSemantics.LEGACY_FUSED_V0,
        baseline=SeqaxFeedForwardFusion.SEPARATE,
        candidate=SeqaxFeedForwardFusion.SILU_MULTIPLY,
    )


def _root(value: SSAValue) -> SSAValue:
    while isinstance(value.owner, ViewOp):
        value = value.owner.base
    return value


def _operation_ids(module: ModuleOp) -> dict[Operation, str]:
    return {
        operation: f"{index:04d}:{operation.name}" for index, operation in enumerate(module.walk())
    }


def _lineage_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _vector_input_lineages(module: ModuleOp) -> dict[VectorComputeOp, tuple[str, ...]]:
    kernel = next(
        (operation for operation in module.walk() if isinstance(operation, KernelOp)),
        None,
    )
    if kernel is None:
        raise UnsupportedPhysicalFusionError("fusion comparison requires one physical kernel")
    contents: dict[SSAValue, str] = {
        argument: _lineage_hash(("argument", index, _value_contract(argument)))
        for index, argument in enumerate(kernel.body.block.args)
    }
    vector_inputs: dict[VectorComputeOp, tuple[str, ...]] = {}
    silu_inputs: dict[SSAValue, str] = {}

    def lineage(value: SSAValue) -> str:
        if isinstance(value.owner, ViewOp):
            return _lineage_hash(
                (
                    "view",
                    lineage(value.owner.base),
                    str(value.owner.region),
                    _value_contract(value),
                )
            )
        root = _root(value)
        try:
            return contents[root]
        except KeyError as error:
            raise UnsupportedPhysicalFusionError(
                "fusion comparison cannot derive complete physical value lineage"
            ) from error

    for operation in kernel.body.block.ops:
        if isinstance(operation, AllocOp | ViewOp):
            continue
        if isinstance(operation, DmaStartOp):
            contents[_root(operation.destination)] = lineage(operation.source)
            continue
        if isinstance(operation, MxuEinsumOp):
            contents[_root(operation.accumulator)] = _lineage_hash(
                (
                    "mxu_einsum",
                    tuple(lineage(value) for value in (operation.lhs, operation.rhs)),
                    tuple(value.data for value in operation.contracting_dimensions),
                    tuple(value.data for value in operation.pending_reduction_axes),
                )
            )
            continue
        if isinstance(operation, CollectiveOp):
            contents[_root(operation.destination)] = _lineage_hash(
                (
                    "collective",
                    lineage(operation.source),
                    operation.kind.data.value,
                    operation.mesh_axis.data,
                    operation.group_size.data,
                    operation.split_dimension.data,
                    operation.concat_dimension.data,
                    operation.reducer.data,
                )
            )
            continue
        if isinstance(operation, VectorComputeOp):
            inputs = tuple(lineage(value) for value in operation.inputs)
            vector_inputs[operation] = inputs
            function = operation.function.data
            semantic_inputs = inputs
            if function == "silu":
                silu_inputs[_root(operation.output)] = inputs[0]
            elif function == "multiply":
                silu_positions = tuple(
                    index
                    for index, value in enumerate(operation.inputs)
                    if _root(value) in silu_inputs
                )
                if len(silu_positions) == 1:
                    silu_position = silu_positions[0]
                    function = "silu_multiply"
                    semantic_inputs = (
                        silu_inputs[_root(operation.inputs[silu_position])],
                        inputs[1 - silu_position],
                    )
            contents[_root(operation.output)] = _lineage_hash(
                (
                    "vector_compute",
                    function,
                    semantic_inputs,
                    tuple(value.data for value in operation.configuration),
                    tuple(value.data for value in operation.pending_reduction_axes),
                    None
                    if operation.materialization is None
                    else operation.materialization.data.value,
                )
            )
    return vector_inputs


def _value_contract(value: SSAValue) -> tuple[object, ...]:
    buffer = value.type
    assert isinstance(buffer, BufferType)
    return (
        buffer.storage.get_shape(),
        str(buffer.storage.element_type),
        tuple(item.data for item in buffer.shape.dimensions),
        buffer.space.data.value,
        tuple(item.data for item in buffer.sharding.axes),
        tuple(item.data for item in buffer.layout.order),
        buffer.ownership.data.value,
    )


def _vectors(module: ModuleOp, function: str) -> tuple[VectorComputeOp, ...]:
    return tuple(
        operation
        for operation in module.walk()
        if isinstance(operation, VectorComputeOp) and operation.function.data == function
    )


def _normalized_vector_functions(module: ModuleOp) -> Counter[str]:
    functions: Counter[str] = Counter()
    for operation in module.walk():
        if not isinstance(operation, VectorComputeOp):
            continue
        if operation.function.data == "silu_multiply":
            functions.update(("silu", "multiply"))
        else:
            functions[operation.function.data] += 1
    return functions


def _work_signature(report: PhysicalKernelResourceReport) -> tuple[object, ...]:
    return (
        report.target,
        report.mesh_axes,
        report.device_count,
        report.hardware,
        tuple(
            (
                value.input_dtype,
                value.batch,
                value.m,
                value.k,
                value.n,
                value.tile_m,
                value.tile_k,
                value.tile_n,
                value.executions,
                value.pending_reduction_axes,
            )
            for value in report.mxu_regions
        ),
        sum(value.scalar_flops for value in report.vector_work),
        sum(value.special_function_ops for value in report.vector_work),
        sum(value.index_and_compare_ops for value in report.vector_work),
        report.memory.explicit_hbm_dma_read_bytes_per_device,
        report.memory.explicit_hbm_dma_write_bytes_per_device,
        report.memory.explicit_local_dma_bytes_per_device,
        tuple(
            (
                value.kind,
                value.executions,
                value.mesh_axis,
                value.group_size,
                value.payload_bytes_per_device,
                value.total_ring_equivalent_bidirectional_bytes_per_device,
            )
            for value in report.collectives
        ),
        tuple(
            (
                value.executions,
                value.transfer_plan,
                value.payload_bytes_per_route,
                value.route_count,
                value.aggregate_link_bytes,
            )
            for value in report.remote_dmas
        ),
        report.priced_compute_time_floor_ns,
        report.priced_hbm_time_floor_ns,
        report.collective_ring_equivalent_time_scenario_ns,
        report.remote_dma_exact_endpoint_time_floor_ns,
        report.remote_dma_exact_link_time_floor_ns,
        report.combined_ici_injection_time_scenario_ns,
        report.priced_ici_time_scenario_ns,
    )


def compare_physical_silu_multiply_fusion(
    baseline_module: ModuleOp,
    candidate_module: ModuleOp,
    *,
    hardware: HardwareRateModel,
) -> PhysicalFusionComparison:
    if hardware != tpu7x_tensorcore_rates():
        raise UnsupportedPhysicalFusionError(
            "physical-fusion-comparison-v1 requires the exact TPU7x rate authority"
        )
    baseline = analyze_physical_kernel(baseline_module, hardware=hardware)
    candidate = analyze_physical_kernel(candidate_module, hardware=hardware)
    baseline_silu = _vectors(baseline_module, "silu")
    baseline_multiply = _vectors(baseline_module, "multiply")
    candidate_fused = _vectors(candidate_module, "silu_multiply")
    if (
        not baseline_silu
        or len(baseline_silu) != len(baseline_multiply)
        or len(baseline_silu) != len(candidate_fused)
        or _vectors(baseline_module, "silu_multiply")
        or _vectors(candidate_module, "silu")
        or _vectors(candidate_module, "multiply")
    ):
        raise UnsupportedPhysicalFusionError(
            "fusion comparison requires exact separate and fused SiLU-multiply inventories"
        )
    if _normalized_vector_functions(baseline_module) != _normalized_vector_functions(
        candidate_module
    ):
        raise UnsupportedPhysicalFusionError("fusion changes the normalized vector work inventory")
    baseline_ids = _operation_ids(baseline_module)
    candidate_ids = _operation_ids(candidate_module)
    baseline_lineages = _vector_input_lineages(baseline_module)
    candidate_lineages = _vector_input_lineages(candidate_module)
    rewrites: list[PhysicalFusionRewrite] = []
    for index, (silu, multiply, fused) in enumerate(
        zip(baseline_silu, baseline_multiply, candidate_fused, strict=True)
    ):
        silu_uses = tuple(
            operand_index
            for operand_index, operand in enumerate(multiply.inputs)
            if operand == silu.output
        )
        if len(silu_uses) != 1:
            raise UnsupportedPhysicalFusionError(
                "baseline multiply must consume the corresponding SiLU intermediate"
            )
        other_input = multiply.inputs[1 - silu_uses[0]]
        if not (
            silu.materialization is None
            and _value_contract(silu.inputs[0]) == _value_contract(fused.inputs[0])
            and _value_contract(other_input) == _value_contract(fused.inputs[1])
            and baseline_lineages[silu][0] == candidate_lineages[fused][0]
            and baseline_lineages[multiply][1 - silu_uses[0]] == candidate_lineages[fused][1]
            and _value_contract(multiply.output) == _value_contract(fused.output)
            and tuple(value.data for value in silu.pending_reduction_axes)
            == tuple(value.data for value in multiply.pending_reduction_axes)
            == tuple(value.data for value in fused.pending_reduction_axes)
        ):
            raise UnsupportedPhysicalFusionError(
                "fused SiLU multiply does not preserve baseline value contracts or producer lineage"
            )
        intermediate = _root(silu.output).type
        assert isinstance(intermediate, BufferType)
        rewrites.append(
            PhysicalFusionRewrite(
                index=index,
                baseline_silu_operation_id=baseline_ids[silu],
                baseline_multiply_operation_id=baseline_ids[multiply],
                candidate_fused_operation_id=candidate_ids[fused],
                eliminated_intermediate_vmem_bytes_per_device=buffer_bytes(intermediate),
            )
        )
    baseline_static = Counter(dict(baseline.canonical_operation_inventory))
    candidate_static = Counter(dict(candidate.canonical_operation_inventory))
    expected_static = baseline_static.copy()
    expected_static["tpu_schedule.vector_compute"] -= len(rewrites)
    expected_static["tpu_schedule.alloc"] -= len(rewrites)
    if +expected_static != +candidate_static:
        raise UnsupportedPhysicalFusionError(
            "fusion changes physical operation classes beyond vector and intermediate allocation"
        )
    declared_work_equal = _work_signature(baseline) == _work_signature(candidate)
    if not declared_work_equal:
        raise UnsupportedPhysicalFusionError("fusion changes declared physical work or traffic")
    intermediate_bytes = sum(
        value.eliminated_intermediate_vmem_bytes_per_device for value in rewrites
    )
    allocated_savings = (
        baseline.memory.allocated_vmem_bytes_per_device
        - candidate.memory.allocated_vmem_bytes_per_device
    )
    peak_savings = (
        baseline.memory.peak_live_vmem_bytes_per_device
        - candidate.memory.peak_live_vmem_bytes_per_device
    )
    return PhysicalFusionComparison(
        schema_version=PHYSICAL_FUSION_COMPARISON_SCHEMA,
        transformation="silu_multiply",
        baseline=baseline,
        candidate=candidate,
        rewrites=tuple(rewrites),
        eliminated_vector_operations=len(rewrites),
        eliminated_intermediate_vmem_bytes_per_device=intermediate_bytes,
        allocated_vmem_savings_bytes_per_device=allocated_savings,
        peak_live_vmem_savings_bytes_per_device=peak_savings,
        declared_work_is_equal=True,
        candidate_is_static_resource_preference=allocated_savings > 0 and peak_savings >= 0,
        measured_performance_winner=None,
        predictive_validation=False,
        assumptions=(
            "The candidate physical IR selects the full-local Pallas implementation for each fused vector operation.",
            "Static preference means equal declared work with fewer operations and no worse declared memory.",
            "Memory savings are declared schedule-model savings, not measured physical VMEM savings.",
        ),
        omissions=(
            "The generic pair comparator does not establish program provenance or semantic equivalence.",
            "This static report does not bind the Pallas lowering implementation, compiled source, or executable HLO.",
            "No launch, vector, or special-function rate is priced.",
            "No TPU Mosaic compile, warm timing, trace, hardware counters, or predictive calibration is attached.",
            "Static preference is not a performance promotion.",
        ),
    )


def derive_seqax_silu_multiply_fusion_report(
    contract: SeqaxSiluMultiplyFusionContract,
) -> SeqaxSiluMultiplyFusionReport:
    if contract != default_seqax_silu_multiply_fusion_contract():
        raise ValueError("SEQAX_SILU_MULTIPLY_FUSION_CONTRACT_MISMATCH")
    hardware = tpu7x_tensorcore_rates()
    comparisons: list[SeqaxFusionScenarioComparison] = []
    surface = seqax_forward_workload_surface()
    for scenario in surface.scenarios:
        parameters = scenario.parameters()
        baseline_distributed = seqax_forward_schedule(
            **parameters,
            numerical_semantics=contract.numerical_semantics,
            feed_forward_fusion=contract.baseline,
        )
        candidate_distributed = seqax_forward_schedule(
            **parameters,
            numerical_semantics=contract.numerical_semantics,
            feed_forward_fusion=contract.candidate,
        )
        baseline = lower_seqax_forward_to_physical(baseline_distributed).module
        candidate = lower_seqax_forward_to_physical(candidate_distributed).module
        comparisons.append(
            SeqaxFusionScenarioComparison(
                scenario_name=scenario.name,
                baseline_distributed_schedule_sha256=schedule_sha256(baseline_distributed),
                candidate_distributed_schedule_sha256=schedule_sha256(candidate_distributed),
                comparison=compare_physical_silu_multiply_fusion(
                    baseline,
                    candidate,
                    hardware=hardware,
                ),
            )
        )
    return SeqaxSiluMultiplyFusionReport(
        schema_version=SEQAX_SILU_MULTIPLY_FUSION_SCHEMA,
        contract_id=contract.contract_id,
        surface_id=contract.surface_id,
        scenarios=tuple(comparisons),
        semantic_authority="canonical_seqax_regeneration",
        measured_performance_winner=None,
        predictive_validation=False,
    )


def validate_seqax_silu_multiply_fusion_report(
    report: SeqaxSiluMultiplyFusionReport,
    *,
    contract: SeqaxSiluMultiplyFusionContract,
) -> None:
    expected = derive_seqax_silu_multiply_fusion_report(contract)
    if report != expected:
        raise ValueError("SEQAX_SILU_MULTIPLY_FUSION_REPORT_REPLAY_MISMATCH")


def write_seqax_silu_multiply_fusion_report(
    output: Path,
    *,
    contract: SeqaxSiluMultiplyFusionContract,
) -> SeqaxSiluMultiplyFusionReport:
    output = output.resolve()
    if output.exists():
        raise ValueError("SEQAX_SILU_MULTIPLY_FUSION_OUTPUT_EXISTS")
    report = derive_seqax_silu_multiply_fusion_report(contract)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}")
    payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return report
