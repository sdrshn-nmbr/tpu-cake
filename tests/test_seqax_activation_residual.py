import inspect
import os
from pathlib import Path

import pytest

import tpu_cake.seqax_activation_residual_runner as activation_runner
from tpu_cake.compiler_analysis import (
    CompilerCollectiveAnalysis,
    CompilerCostMetric,
    CompilerExecutableAnalysis,
    CompilerMemoryAnalysis,
)
from tpu_cake.cost_model import tpu7x_tensorcore_rates
from tpu_cake.identity import json_sha256
from tpu_cake.physical_cost_model import analyze_physical_kernel
from tpu_cake.runner import _runtime_identity
from tpu_cake.seqax_activation_residual import (
    SeqaxActivationResidualDesignContract,
    default_seqax_activation_residual_design_contract,
)
from tpu_cake.seqax_activation_residual_runner import (
    SeqaxActivationResidualCompilerCandidate,
    _compile_candidate,
    _compiler_environment,
    _prepare_candidates,
    _require_safe_new_root,
    _validate_pallas_collectives,
    analyze_activation_residual_boundary,
)
from tpu_cake.seqax_large_residual_qualification import (
    default_seqax_large_residual_qualification_failure_record,
)
from tpu_cake.seqax_pallas_lowering import lower_seqax_physical_to_pallas
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.workloads.seqax_forward import (
    SeqaxNumericalSemantics,
    SeqaxResidualNormStrategy,
    seqax_forward_schedule,
)


def test_external_activation_residual_design_is_canonical() -> None:
    saved = SeqaxActivationResidualDesignContract.model_validate_json(
        Path("contracts/seqax-activation-residual-v1.json").read_text()
    )

    assert saved == default_seqax_activation_residual_design_contract(saved.runtime)
    assert saved.source_remote_url == "https://github.com/sdrshn-nmbr/tpu-cake.git"
    assert saved.source_branch == "main"
    assert saved.compiler_environment == {
        "LIBTPU_INIT_ARGS": " --xla_tpu_use_enhanced_launch_barrier=true",
        "TPU_LIBRARY_PATH": "/home/sudarshan/tpu-cake-main/.venv/lib/python3.12/site-packages/libtpu/libtpu.so",
    }
    assert saved.compile_input_mode == "abstract-only"
    assert saved.full_activation_bf16_bytes_per_data_shard == 1_048_576
    assert saved.full_activation_bf16_bytes_per_data_shard > (
        saved.illustrative_latency_crossover_bytes
    )
    assert saved.latency_crossover_role == ("illustrative-hardware-specific-not-a-tpu7x-threshold")
    assert saved.predicted_ring_savings_bytes_per_device == 3_145_728
    assert saved.predicted_ring_savings_fraction > 0.19
    assert saved.parameters["model"] == 256
    assert saved.parameters["batch"] == 64
    assert saved.parameters["sequence"] == 64
    assert saved.failed_workload_boundary_relation == (
        "same-one-mib-boundary-with-sixteen-times-shorter-model-axis-reductions"
    )
    assert saved.boundary_reduce_scatter_input == "f32[32,64,256]"
    assert saved.boundary_reduce_scatter_output == "f32[32,64,64]"
    assert saved.boundary_residual_output == "bf16[32,64,64]"
    assert saved.boundary_all_gather_output == "bf16[32,64,256]"
    assert saved.top1_role == "diagnostic-only"
    assert (
        saved.parent_failure_record_id
        == default_seqax_large_residual_qualification_failure_record().record_id
    )
    assert saved.correctness_policy_status == "pending-calibration"
    assert not saved.timing_authorized


def test_activation_residual_static_plans_replay() -> None:
    contract = default_seqax_activation_residual_design_contract(_runtime_identity())
    parameters = dict(contract.parameters)
    parameters["numerical_semantics"] = SeqaxNumericalSemantics(parameters["numerical_semantics"])

    for expected in contract.candidates:
        distributed = seqax_forward_schedule(
            **parameters,
            residual_norm_strategy=expected.candidate,
        )
        physical = lower_seqax_forward_to_physical(distributed).module
        plan = lower_seqax_physical_to_pallas(distributed, physical)
        report = analyze_physical_kernel(physical, hardware=tpu7x_tensorcore_rates())

        assert plan.distributed_schedule_sha256 == expected.distributed_schedule_sha256
        assert plan.physical_schedule_sha256 == expected.physical_schedule_sha256
        assert plan.source_sha256() == expected.pallas_source_sha256
        assert json_sha256(plan.manifest()) == expected.pallas_manifest_sha256
        assert plan.pallas_region_count == expected.expected_pallas_regions
        assert (
            report.devices[0].collective_ring_equivalent_bytes
            == expected.expected_ring_equivalent_ici_bytes_per_device
        )
        assert (
            report.memory.peak_live_vmem_bytes_per_device
            == expected.expected_peak_vmem_bytes_per_device
        )


def _boundary_hlo(
    *,
    dead: bool = False,
    live: bool = True,
    broken_edge: bool = False,
) -> str:
    chains = []
    fusion_computations = []
    for index in range(2):
        gather_input = f"convert_add_fusion{index}" if not broken_edge or index == 0 else "wrong"
        chains.append(f"  %residual{index} = bf16[32,64,64] parameter({index})")
        chains.extend(
            (
                (
                    f"  %rs{index} = ((f32[32,64,256], token[]), f32[32,64,64]) "
                    f"call-start(%input{index}, %token{index}), "
                    'async_execution_thread="sparsecore", '
                    'metadata={op_name="jit(physical_call)/shard_map/reduce_scatter"}, '
                    'backend_config={"sparse_core_config":{"offload":"OFFLOAD_COLLECTIVE"}}'
                ),
                (
                    f"  %done{index} = f32[32,64,64] "
                    f"call-done(%rs{index}, %token{index}), "
                    'metadata={op_name="jit(physical_call)/shard_map/reduce_scatter"}'
                ),
                (
                    f"  %convert_add_fusion{index} = bf16[32,64,64] "
                    f"fusion(%done{index}, %residual{index}), "
                    f"calls=%residual_add{index}, "
                    'metadata={op_name="jit(physical_call)/shard_map/add"}'
                ),
                (
                    f"  %ag{index} = ((bf16[32,64,64], token[]), bf16[32,64,256]) "
                    f"call-start(%{gather_input}, %token{index}), "
                    'async_execution_thread="sparsecore", '
                    'metadata={op_name="jit(physical_call)/shard_map/all_gather"}, '
                    'backend_config={"sparse_core_config":{"offload":"OFFLOAD_COLLECTIVE"}}'
                ),
            )
        )
        fusion_computations.append(
            f"%residual_add{index} (p0: bf16[32,64,64], p1: f32[32,64,64]) "
            "-> bf16[32,64,64] {\n"
            f"  %p0_{index} = bf16[32,64,64] parameter(0)\n"
            f"  %p1_{index} = f32[32,64,64] parameter(1)\n"
            f"  %converted_{index} = bf16[32,64,64] convert(%p1_{index})\n"
            f"  ROOT %add_{index} = bf16[32,64,64] "
            f"add(%p0_{index}, %converted_{index})\n"
            "}"
        )
    root = "  ROOT %result = tuple(%ag0, %ag1)" if live else "  ROOT %constant = f32[] constant(0)"
    body = "\n".join((*chains, root))
    if dead:
        return (
            f"HloModule test\n\n{'\n\n'.join(fusion_computations)}\n\ndead {{\n{body}\n}}\n\n"
            "ENTRY main {\n  ROOT %constant = f32[] constant(0)\n}\n"
        )
    return f"HloModule test\n\n{'\n\n'.join(fusion_computations)}\n\nENTRY main {{\n{body}\n}}\n"


def _collectives(*, reduce_scatters: int = 3) -> CompilerCollectiveAnalysis:
    return CompilerCollectiveAnalysis(
        stablehlo_reduce_scatter_count=reduce_scatters,
        stablehlo_all_gather_count=17,
        compiler_reduce_scatter_count=reduce_scatters,
        compiler_all_reduce_count=0,
        compiler_all_gather_count=17,
        sparse_core_reduce_scatter_count=reduce_scatters,
        sparse_core_all_gather_count=17,
    )


def _analysis(
    *, stable: str = "1", compiler: str = "2", peak: int = 10
) -> CompilerExecutableAnalysis:
    return CompilerExecutableAnalysis(
        stablehlo_sha256=stable * 64,
        compiler_hlo_sha256=compiler * 64,
        cost_metrics=(CompilerCostMetric(name="flops", raw_value=1.0, value=1.0, available=True),),
        memory=CompilerMemoryAnalysis(
            generated_code_size_in_bytes=1,
            argument_size_in_bytes=1,
            output_size_in_bytes=1,
            alias_size_in_bytes=0,
            temp_size_in_bytes=1,
            host_generated_code_size_in_bytes=0,
            host_argument_size_in_bytes=0,
            host_output_size_in_bytes=0,
            host_alias_size_in_bytes=0,
            host_temp_size_in_bytes=0,
            peak_memory_in_bytes=peak,
            buffer_assignment_available=False,
            buffer_assignment_size_bytes=0,
        ),
        collectives=_collectives(),
    )


def _compiler_candidate() -> SeqaxActivationResidualCompilerCandidate:
    boundary = analyze_activation_residual_boundary(_boundary_hlo())
    return SeqaxActivationResidualCompilerCandidate(
        candidate=SeqaxResidualNormStrategy.STANDARD,
        distributed_schedule_sha256="3" * 64,
        physical_schedule_sha256="4" * 64,
        pallas_source_sha256="5" * 64,
        pallas_manifest_sha256="6" * 64,
        pallas_analysis=_analysis(),
        control_analysis=_analysis(compiler="7"),
        pallas_reachable_collectives=_collectives(),
        control_reachable_collectives=_collectives(),
        pallas_boundary=boundary,
        physical_peak_vmem_bytes_per_device=1,
        ring_equivalent_ici_bytes_per_device=1,
    )


def test_activation_boundary_requires_two_entry_reachable_chains() -> None:
    analysis = analyze_activation_residual_boundary(_boundary_hlo())

    assert analysis.chain_count == 2
    assert len({trace.reduce_scatter_start for trace in analysis.traces}) == 2
    assert (
        analyze_activation_residual_boundary(
            _boundary_hlo().replace("convert_add_fusion", "renamed_add")
        ).semantic_id
        == analysis.semantic_id
    )
    assert (
        analyze_activation_residual_boundary(
            _boundary_hlo().replace("ENTRY main {", "ENTRY %main (arg: f32[]) -> tuple() {")
        ).semantic_id
        == analysis.semantic_id
    )
    with pytest.raises(ValueError):
        analyze_activation_residual_boundary(_boundary_hlo(dead=True))
    with pytest.raises(ValueError):
        analyze_activation_residual_boundary(_boundary_hlo(live=False))
    with pytest.raises(ValueError):
        analyze_activation_residual_boundary(_boundary_hlo(broken_edge=True))
    with pytest.raises(ValueError):
        analyze_activation_residual_boundary(_boundary_hlo().replace(" convert(", " copy("))
    with pytest.raises(ValueError):
        analyze_activation_residual_boundary(
            _boundary_hlo().replace('"offload":"OFFLOAD_COLLECTIVE"', '"offload":"NONE"')
        )


def test_capture_root_rejects_lexical_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "evidence"
    repository = tmp_path / "repository"
    evidence.mkdir()
    repository.mkdir()
    monkeypatch.setattr(activation_runner, "_EVIDENCE_ROOT", evidence)
    escaped = evidence / ".." / repository.name / "seqax-activation-residual-compiler-bad"

    with pytest.raises(ValueError, match="ROOT_NOT_NORMALIZED"):
        _require_safe_new_root(escaped, repository)


def test_compiler_semantics_ignore_raw_hlo_and_names_but_not_stable_memory_or_collectives() -> None:
    candidate = _compiler_candidate()
    raw_hlo_changed = candidate.model_copy(update={"pallas_analysis": _analysis(compiler="8")})
    renamed_boundary = candidate.pallas_boundary.model_copy(
        update={
            "traces": tuple(
                trace.model_copy(update={"reduce_scatter_start": f"renamed{trace.ordinal}"})
                for trace in candidate.pallas_boundary.traces
            )
        }
    )

    assert raw_hlo_changed.semantic_id == candidate.semantic_id
    assert candidate.model_copy(update={"pallas_boundary": renamed_boundary}).semantic_id == (
        candidate.semantic_id
    )
    assert (
        candidate.model_copy(update={"pallas_analysis": _analysis(stable="9")}).semantic_id
        != candidate.semantic_id
    )
    assert (
        candidate.model_copy(update={"pallas_analysis": _analysis(peak=11)}).semantic_id
        != candidate.semantic_id
    )
    assert (
        candidate.model_copy(
            update={"pallas_reachable_collectives": _collectives(reduce_scatters=2)}
        ).semantic_id
        != candidate.semantic_id
    )


def test_compiler_environment_rejects_undeclared_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    design = SeqaxActivationResidualDesignContract.model_validate_json(
        Path("contracts/seqax-activation-residual-v1.json").read_text()
    )
    for key in tuple(os.environ):
        if key == "TPU_LIBRARY_PATH" or key.startswith(("JAX_", "XLA_", "PJRT_", "LIBTPU_")):
            monkeypatch.delenv(key)
    for key, value in design.compiler_environment.items():
        monkeypatch.setenv(key, value)

    assert _compiler_environment(design) == design.compiler_environment
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    with pytest.raises(ValueError, match="COMPILER_ENVIRONMENT_MISMATCH"):
        _compiler_environment(design)


def test_pallas_collective_gate_rejects_stablehlo_tampering() -> None:
    design = default_seqax_activation_residual_design_contract(_runtime_identity())
    tampered = _collectives().model_copy(
        update={
            "stablehlo_reduce_scatter_count": 0,
            "stablehlo_all_gather_count": 0,
        }
    )

    with pytest.raises(ValueError, match="NATIVE_COLLECTIVE_MISMATCH"):
        _validate_pallas_collectives(design.candidates[0], tampered)


def test_compile_path_uses_abstract_inputs_and_has_shared_abi() -> None:
    source = inspect.getsource(_compile_candidate)
    module_source = Path("src/tpu_cake/seqax_activation_residual_runner.py").read_text()
    design = default_seqax_activation_residual_design_contract(_runtime_identity())
    prepared = _prepare_candidates(design)

    assert "ShapeDtypeStruct" in module_source
    assert "seqax_forward_inputs" not in module_source
    assert "jax.device_put(" not in module_source
    assert ".block_until_ready(" not in module_source
    assert "device_get(" not in module_source
    assert "pallas_executable(" not in source
    assert "control_executable(" not in source
    assert len(prepared[0].plan.input_contracts) == 13
    assert prepared[0].plan.input_contracts == prepared[1].plan.input_contracts
    assert prepared[0].plan.output_contracts == prepared[1].plan.output_contracts
