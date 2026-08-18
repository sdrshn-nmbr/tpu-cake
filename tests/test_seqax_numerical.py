import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from pydantic import ValidationError
from xdsl.utils.exceptions import VerifyException

from tpu_cake.frontend import canonical_module_text
from tpu_cake.seqax_numerical import (
    BF16_UNIT_ROUNDOFF,
    SEQAX_BF16_FORWARD_NUMERICAL_SCHEMA,
    SeqaxBf16NumericalScenario,
    SeqaxBf16ValidationContract,
    SeqaxInputMutation,
    SeqaxNumericalDiscriminator,
    assess_seqax_bf16_forward,
    decode_seqax_bf16_checkpoint,
    default_seqax_bf16_validation_contract,
    encode_seqax_bf16_checkpoint,
    mutate_seqax_forward_inputs,
    rounded_mathematical_silu_bf16,
    seqax_discriminator_clause,
    validate_seqax_numerical_inputs,
    validate_strict_silu_stablehlo,
)
from tpu_cake.seqax_pallas_lowering import (
    _parse_physical,
    lower_seqax_physical_to_pallas,
)
from tpu_cake.seqax_pallas_search import SEQAX_PALLAS_CORRECTNESS_SEEDS
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.workloads.seqax_forward import (
    SeqaxNumericalSemantics,
    seqax_forward_schedule,
)
from tpu_cake.workloads.seqax_oracle import (
    seqax_forward_canonical_reference,
    seqax_forward_inputs,
)


def _calibration_evidence(
    seed_index: int = 0,
) -> tuple[
    SeqaxBf16ValidationContract,
    SeqaxBf16NumericalScenario,
    tuple[np.ndarray, ...],
    np.ndarray,
    dict[str, object],
]:
    contract = default_seqax_bf16_validation_contract()
    scenario = contract.scenarios[0]
    seed = scenario.seeds[seed_index]
    inputs = seqax_forward_inputs(seed=seed, **scenario.parameters.model_dump())
    reference = seqax_forward_canonical_reference(inputs, **scenario.parameters.model_dump())
    gate = np.full(scenario.gate_checkpoints[0].shape, 0.625, dtype=ml_dtypes.bfloat16)
    silu = rounded_mathematical_silu_bf16(gate)
    evidence: dict[str, object] = {
        "seed": seed,
        "inputs": inputs,
        "pallas_gate_checkpoints": (gate,),
        "control_gate_checkpoints": (gate.copy(),),
        "pallas_silu_checkpoints": (silu,),
        "control_silu_checkpoints": (silu.copy(),),
    }
    return contract, scenario, inputs, reference, evidence


def test_mathematical_silu_reference_rounds_once_to_bf16() -> None:
    value = np.asarray(
        [-3.625, -2.125, -0.3046875, 0.0, 0.8828125, 2.125, 2.96875, 3.375],
        dtype=ml_dtypes.bfloat16,
    )

    actual = rounded_mathematical_silu_bf16(value)

    np.testing.assert_array_equal(
        actual,
        np.asarray(
            [
                -0.09423828125,
                -0.2265625,
                -0.12890625,
                0.0,
                0.625,
                1.8984375,
                2.828125,
                3.265625,
            ],
            dtype=ml_dtypes.bfloat16,
        ),
    )
    assert actual.dtype == np.dtype(ml_dtypes.bfloat16)
    assert BF16_UNIT_ROUNDOFF == 0.00390625
    assert SEQAX_BF16_FORWARD_NUMERICAL_SCHEMA == "bf16-forward-numerical-v1"


def test_mathematical_silu_reference_rejects_non_bf16_or_nonfinite_input() -> None:
    with pytest.raises(TypeError, match="requires BF16"):
        rounded_mathematical_silu_bf16(np.asarray([1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="requires finite"):
        rounded_mathematical_silu_bf16(np.asarray([np.inf, np.nan], dtype=ml_dtypes.bfloat16))


def test_strict_silu_stablehlo_requires_barrier_dataflow_into_multiply() -> None:
    stablehlo = """module {
      func.func private @silu(tensor<1x4xbf16>) -> tensor<1x4xbf16>
      func.func @main(
        %arg0: tensor<1x4xbf16>, %other: tensor<1x4xbf16>
      ) -> tensor<1x4xbf16> {
        %1 = stablehlo.optimization_barrier %arg0 : tensor<1x4xbf16>
        %2 = func.call @silu(%1) : (tensor<1x4xbf16>) -> tensor<1x4xbf16>
        %3 = stablehlo.optimization_barrier %2 : tensor<1x4xbf16>
        %4 = stablehlo.multiply %other, %3 : tensor<1x4xbf16>
        return %4 : tensor<1x4xbf16>
      }
    }"""

    validate_strict_silu_stablehlo(stablehlo, expected_count=1)

    with pytest.raises(ValueError, match="input barrier"):
        validate_strict_silu_stablehlo(
            stablehlo.replace("func.call @silu(%1)", "func.call @silu(%arg0)"),
            expected_count=1,
        )
    with pytest.raises(ValueError, match="result must feed only"):
        validate_strict_silu_stablehlo(
            stablehlo.replace("optimization_barrier %2", "optimization_barrier %arg0"),
            expected_count=1,
        )
    with pytest.raises(ValueError, match="must feed exactly one"):
        validate_strict_silu_stablehlo(
            stablehlo.replace("multiply %other, %3", "multiply %other, %arg0"),
            expected_count=1,
        )
    with pytest.raises(ValueError, match="result must feed only"):
        validate_strict_silu_stablehlo(
            stablehlo.replace(
                "return %4 : tensor<1x4xbf16>",
                "%5 = stablehlo.multiply %other, %2 : tensor<1x4xbf16>\n"
                "        return %4 : tensor<1x4xbf16>",
            ),
            expected_count=1,
        )
    with pytest.raises(ValueError, match="must reach its function return"):
        validate_strict_silu_stablehlo(
            stablehlo.replace(
                "return %4 : tensor<1x4xbf16>",
                "%5 = stablehlo.add %4, %other : tensor<1x4xbf16>\n"
                "        %6 = stablehlo.multiply %other, %arg0 : tensor<1x4xbf16>\n"
                "        return %6 : tensor<1x4xbf16>",
            ),
            expected_count=1,
        )
    with pytest.raises(ValueError, match="expected 2 calls"):
        validate_strict_silu_stablehlo(stablehlo, expected_count=2)


def test_strict_silu_stablehlo_scopes_ssa_values_per_function() -> None:
    function = """
      func.func @{name}(
        %arg0: tensor<1x4xbf16>, %other: tensor<1x4xbf16>
      ) -> tensor<1x4xbf16> {{
        %0 = stablehlo.optimization_barrier %arg0 : tensor<1x4xbf16>
        %1 = func.call @silu(%0) : (tensor<1x4xbf16>) -> tensor<1x4xbf16>
        %2 = stablehlo.optimization_barrier %1 : tensor<1x4xbf16>
        %3 = stablehlo.multiply %other, %2 : tensor<1x4xbf16>
        return %3 : tensor<1x4xbf16>
      }}
    """
    stablehlo = (
        """module {
      func.func private @silu(tensor<1x4xbf16>) -> tensor<1x4xbf16>
    """
        + function.format(name="first")
        + function.format(name="second")
        + "}"
    )

    validate_strict_silu_stablehlo(stablehlo, expected_count=2)


def test_bf16_forward_contract_binds_surface_abi_and_held_out_seeds() -> None:
    contract = default_seqax_bf16_validation_contract()
    encoded = contract.model_dump_json(indent=2, exclude_computed_fields=True)

    assert SeqaxBf16ValidationContract.model_validate_json(encoded) == contract
    assert contract.scenarios[0].seeds == SEQAX_PALLAS_CORRECTNESS_SEEDS
    assert contract.scenarios[0].role.value == "calibration"
    assert tuple(scenario.role.value for scenario in contract.scenarios[1:]) == (
        "held_out",
        "held_out",
        "held_out",
    )
    assert not set(contract.scenarios[0].seeds).intersection(
        seed for scenario in contract.scenarios[1:] for seed in scenario.seeds
    )
    assert contract.required_discriminators == tuple(SeqaxNumericalDiscriminator)
    assert tuple(tensor.name for tensor in contract.scenarios[0].inputs) == (
        "tokens",
        "sequence_starts",
        "embedding",
        "layer_norm_1",
        "layer_norm_2",
        "query_weights",
        "key_value_weights",
        "output_weights",
        "gate_weights",
        "up_weights",
        "down_weights",
        "final_layer_norm",
        "unembedding",
    )
    assert contract.scenarios[1].output.shape == (2, 3, 32)
    assert tuple(value.shape for value in contract.scenarios[1].silu_checkpoints) == (
        (2, 3, 24),
        (2, 3, 24),
    )


def test_bf16_forward_external_contract_is_canonical() -> None:
    saved = SeqaxBf16ValidationContract.model_validate_json(
        Path("contracts/seqax-bf16-forward-numerical-v1.json").read_text()
    )

    assert saved == default_seqax_bf16_validation_contract()
    assert saved.contract_id == "07a9f56c80b3019ee30aa30253e24256448f4bbe4b8add70ccbf6dec7d21135a"
    assert saved.acceptance_authority == "authenticated-runner-and-relocated-public-replay"
    assert saved.require_instrumented_output_parity
    assert saved.require_discriminator_artifact_replay


def test_bf16_forward_scenario_abis_match_strict_physical_plans() -> None:
    for scenario in default_seqax_bf16_validation_contract().scenarios:
        distributed = seqax_forward_schedule(
            **scenario.parameters.model_dump(),
            numerical_semantics=SeqaxNumericalSemantics.TYPED_BF16_V1,
        )
        physical = lower_seqax_forward_to_physical(distributed).module
        plan = lower_seqax_physical_to_pallas(distributed, physical)

        assert tuple(
            (tuple(size for _, size in value.shape), value.dtype) for value in plan.input_contracts
        ) == tuple((value.shape, value.dtype) for value in scenario.inputs)
        assert len(plan.output_contracts) == 1
        assert (
            tuple(size for _, size in plan.output_contracts[0].shape),
            plan.output_contracts[0].dtype,
        ) == (scenario.output.shape, scenario.output.dtype)


def test_bf16_forward_input_abi_validation_is_exact() -> None:
    scenario = default_seqax_bf16_validation_contract().scenarios[0]
    parameters = scenario.parameters.model_dump()
    inputs = seqax_forward_inputs(seed=scenario.seeds[0], **parameters)
    validate_seqax_numerical_inputs(inputs, scenario)

    wrong_dtype = list(inputs)
    wrong_dtype[0] = wrong_dtype[0].astype(np.int32)
    with pytest.raises(TypeError, match="dtype mismatch: tokens"):
        validate_seqax_numerical_inputs(tuple(wrong_dtype), scenario)

    wrong_shape = list(inputs)
    wrong_shape[2] = wrong_shape[2][:-1]
    with pytest.raises(ValueError, match="shape mismatch: embedding"):
        validate_seqax_numerical_inputs(tuple(wrong_shape), scenario)

    with pytest.raises(ValueError, match="input count"):
        validate_seqax_numerical_inputs(inputs[:-1], scenario)


def test_bf16_checkpoint_uint16_codec_preserves_exact_logical_bits() -> None:
    checkpoint_contract = default_seqax_bf16_validation_contract().scenarios[0].silu_checkpoints[0]
    values = np.linspace(-3.0, 3.0, num=np.prod(checkpoint_contract.shape)).reshape(
        checkpoint_contract.shape
    )
    logical = np.asarray(values, dtype=ml_dtypes.bfloat16)
    logical.reshape(-1)[0] = np.asarray(-0.0, dtype=ml_dtypes.bfloat16)
    logical.reshape(-1)[1] = np.asarray(0.0, dtype=ml_dtypes.bfloat16)

    stored = encode_seqax_bf16_checkpoint(logical, checkpoint_contract)
    decoded = decode_seqax_bf16_checkpoint(stored, checkpoint_contract)

    assert stored.dtype == np.dtype(np.uint16)
    np.testing.assert_array_equal(decoded.view(np.uint16), logical.view(np.uint16))
    with pytest.raises(TypeError, match="storage must use uint16"):
        decode_seqax_bf16_checkpoint(stored.astype(np.int16), checkpoint_contract)
    with pytest.raises(ValueError, match="shape does not match"):
        decode_seqax_bf16_checkpoint(stored[:, :, :-1], checkpoint_contract)
    nonfinite = logical.copy()
    nonfinite.reshape(-1)[0] = np.asarray(np.nan, dtype=ml_dtypes.bfloat16)
    with pytest.raises(ValueError, match="must be finite"):
        encode_seqax_bf16_checkpoint(nonfinite, checkpoint_contract)
    with pytest.raises(ValueError, match="must be finite"):
        decode_seqax_bf16_checkpoint(nonfinite.view(np.uint16), checkpoint_contract)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("device_count",), True, "device_count"),
        (("policy", "cpu_relative_l2_units"), 2.1, "not canonical"),
        (("acceptance_authority",), "pure-evaluator", "acceptance authority"),
        (("scenarios", 0, "parameters", "model"), "256", "model"),
        (("scenarios", 0, "inputs", 0, "dtype"), "int32", "input ABI"),
        (
            ("required_discriminators",),
            [value.value for value in tuple(SeqaxNumericalDiscriminator)[:-1]],
            "discriminators",
        ),
    ),
)
def test_bf16_forward_contract_rejects_noncanonical_values(
    path: tuple[str | int, ...], value: object, message: str
) -> None:
    payload = default_seqax_bf16_validation_contract().model_dump(
        mode="json", exclude={"contract_id"}
    )
    target: object = payload
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValidationError, match=message):
        SeqaxBf16ValidationContract.model_validate_json(json.dumps(payload))


def test_bf16_forward_policy_distinguishes_row_spikes_and_distributed_drift() -> None:
    contract, scenario, _inputs, reference, evidence = _calibration_evidence()
    policy = contract.policy
    localized = reference.copy()
    row_scale = max(float(np.max(np.abs(reference[0, 0]))), 1.0)
    localized[0, 0, 0] += np.float32(4.5 * BF16_UNIT_ROUNDOFF * row_scale)
    distributed = reference * np.float32(1.0 + 2.1 * BF16_UNIT_ROUNDOFF)

    localized_assessment = assess_seqax_bf16_forward(
        localized,
        localized,
        **evidence,
        policy=policy,
        scenario=scenario,
    )
    distributed_assessment = assess_seqax_bf16_forward(
        distributed,
        distributed,
        **evidence,
        policy=policy,
        scenario=scenario,
    )

    assert localized_assessment.cpu_pallas_relative_l2 < 2 * BF16_UNIT_ROUNDOFF
    assert localized_assessment.cpu_pallas_row_scaled_max > 4 * BF16_UNIT_ROUNDOFF
    assert not localized_assessment.final_outputs_satisfy_policy
    assert localized_assessment.checkpoint_values_consistent
    assert distributed_assessment.cpu_pallas_relative_l2 > 2 * BF16_UNIT_ROUNDOFF
    assert distributed_assessment.cpu_pallas_row_scaled_max < 4 * BF16_UNIT_ROUNDOFF
    assert not distributed_assessment.final_outputs_satisfy_policy
    assert distributed_assessment.checkpoint_values_consistent


def test_bf16_forward_policy_reports_top1_without_using_it_as_the_oracle() -> None:
    contract, scenario, _inputs, reference, evidence = _calibration_evidence(-1)
    policy = contract.policy
    actual = reference.copy()
    rows = actual.reshape(-1, actual.shape[-1])
    reference_rows = reference.reshape(-1, reference.shape[-1])
    gaps = []
    for row_index, row in enumerate(reference_rows):
        order = np.argsort(row)
        gaps.append((row[order[-1]] - row[order[-2]], row_index, order[-1], order[-2]))
    _gap, row_index, top, second = min(gaps)
    rows[row_index, top] -= np.float32(0.02)
    rows[row_index, second] += np.float32(0.02)

    assessment = assess_seqax_bf16_forward(
        actual,
        actual,
        **evidence,
        policy=policy,
        scenario=scenario,
    )

    assert assessment.final_outputs_satisfy_policy
    assert assessment.checkpoint_values_consistent
    assert "passed" not in type(assessment).model_fields
    assert not assessment.pallas_top1_matches_cpu
    assert not assessment.control_top1_matches_cpu
    assert assessment.pallas_top1_matches_control


@pytest.mark.parametrize("mutation", tuple(SeqaxInputMutation))
def test_bf16_forward_semantic_input_mutations_fail_the_policy(
    mutation: SeqaxInputMutation,
) -> None:
    contract = default_seqax_bf16_validation_contract()
    scenario = contract.scenarios[0]
    parameters = scenario.parameters.model_dump()
    inputs = seqax_forward_inputs(seed=scenario.seeds[0], **parameters)
    mutated = seqax_forward_canonical_reference(
        mutate_seqax_forward_inputs(inputs, mutation), **parameters
    )

    assessment = assess_seqax_bf16_forward(
        mutated,
        mutated,
        **_calibration_evidence()[4],
        policy=contract.policy,
        scenario=scenario,
    )

    assert not assessment.final_outputs_satisfy_policy
    assert assessment.checkpoint_values_consistent


def test_bf16_forward_evaluator_regenerates_oracles_and_rejects_forged_evidence() -> None:
    contract, scenario, _inputs, _reference, evidence = _calibration_evidence()
    fake_output = np.full(scenario.output.shape, 42.0, dtype=np.float32)
    fake_gate = np.full(scenario.gate_checkpoints[0].shape, 0.625, dtype=ml_dtypes.bfloat16)
    fake_silu = np.full_like(fake_gate, 7.0)

    assessment = assess_seqax_bf16_forward(
        fake_output,
        fake_output,
        seed=evidence["seed"],  # type: ignore[arg-type]
        inputs=evidence["inputs"],  # type: ignore[arg-type]
        pallas_gate_checkpoints=(fake_gate,),
        control_gate_checkpoints=(fake_gate,),
        pallas_silu_checkpoints=(fake_silu,),
        control_silu_checkpoints=(fake_silu,),
        policy=contract.policy,
        scenario=scenario,
    )

    assert assessment.cpu_pallas_relative_l2 > 2 * BF16_UNIT_ROUNDOFF
    assert not assessment.pallas_silu_matches_mathematical
    assert not assessment.control_silu_matches_mathematical
    assert not assessment.final_outputs_satisfy_policy
    assert not assessment.checkpoint_values_consistent

    corrupted_inputs = list(evidence["inputs"])  # type: ignore[arg-type]
    corrupted_inputs[2] = corrupted_inputs[2].copy()
    corrupted_inputs[2].reshape(-1)[0] += np.float32(1.0)
    with pytest.raises(ValueError, match="deterministic input mismatch: embedding"):
        assess_seqax_bf16_forward(
            fake_output,
            fake_output,
            seed=evidence["seed"],  # type: ignore[arg-type]
            inputs=tuple(corrupted_inputs),
            pallas_gate_checkpoints=(fake_gate,),
            control_gate_checkpoints=(fake_gate,),
            pallas_silu_checkpoints=(fake_silu,),
            control_silu_checkpoints=(fake_silu,),
            policy=contract.policy,
            scenario=scenario,
        )


@pytest.mark.parametrize(
    "value",
    (
        np.asarray([[np.nan, 0.0]], dtype=np.float32),
        np.asarray([[np.inf, 0.0]], dtype=np.float32),
    ),
)
def test_bf16_forward_policy_rejects_nonfinite_output(value: np.ndarray) -> None:
    contract, scenario, _inputs, reference, evidence = _calibration_evidence()
    policy = contract.policy
    invalid = reference.copy()
    invalid.reshape(-1)[: value.size] = value.reshape(-1)
    with pytest.raises(ValueError, match="must be finite"):
        assess_seqax_bf16_forward(
            invalid,
            reference,
            **evidence,
            policy=policy,
            scenario=scenario,
        )


def test_bf16_forward_policy_rejects_dtype_shape_and_checkpoint_failures() -> None:
    contract, scenario, _inputs, reference, evidence = _calibration_evidence()
    policy = contract.policy
    with pytest.raises(TypeError, match="must use float32"):
        assess_seqax_bf16_forward(
            reference.astype(np.float64),
            reference,
            **evidence,
            policy=policy,
            scenario=scenario,
        )
    with pytest.raises(ValueError, match="shape does not match the contract"):
        assess_seqax_bf16_forward(
            reference[:, :-1],
            reference,
            **evidence,
            policy=policy,
            scenario=scenario,
        )
    gate = evidence["pallas_gate_checkpoints"][0]  # type: ignore[index]
    mathematical = evidence["pallas_silu_checkpoints"][0]  # type: ignore[index]
    wrong = np.zeros_like(mathematical)
    checkpoint_failure = assess_seqax_bf16_forward(
        reference,
        reference,
        seed=evidence["seed"],  # type: ignore[arg-type]
        inputs=evidence["inputs"],  # type: ignore[arg-type]
        pallas_gate_checkpoints=(gate,),
        control_gate_checkpoints=(gate,),
        pallas_silu_checkpoints=(wrong,),
        control_silu_checkpoints=(mathematical,),
        policy=policy,
        scenario=scenario,
    )
    assert not checkpoint_failure.pallas_silu_matches_mathematical
    assert checkpoint_failure.control_silu_matches_mathematical
    assert not checkpoint_failure.silu_cross_path_exact
    assert checkpoint_failure.final_outputs_satisfy_policy
    assert not checkpoint_failure.checkpoint_values_consistent
    with pytest.raises(TypeError, match="must use bfloat16"):
        assess_seqax_bf16_forward(
            reference,
            reference,
            seed=evidence["seed"],  # type: ignore[arg-type]
            inputs=evidence["inputs"],  # type: ignore[arg-type]
            pallas_gate_checkpoints=(gate,),
            control_gate_checkpoints=(gate,),
            pallas_silu_checkpoints=(mathematical.astype(np.float32),),
            control_silu_checkpoints=(mathematical,),
            policy=policy,
            scenario=scenario,
        )
    with pytest.raises(ValueError, match="checkpoint shape does not match"):
        assess_seqax_bf16_forward(
            reference,
            reference,
            seed=evidence["seed"],  # type: ignore[arg-type]
            inputs=evidence["inputs"],  # type: ignore[arg-type]
            pallas_gate_checkpoints=(gate,),
            control_gate_checkpoints=(gate,),
            pallas_silu_checkpoints=(mathematical[:, :0],),
            control_silu_checkpoints=(mathematical,),
            policy=policy,
            scenario=scenario,
        )
    with pytest.raises(ValueError, match="gate checkpoint shape does not match"):
        assess_seqax_bf16_forward(
            reference,
            reference,
            seed=evidence["seed"],  # type: ignore[arg-type]
            inputs=evidence["inputs"],  # type: ignore[arg-type]
            pallas_gate_checkpoints=(gate[:, :0],),
            control_gate_checkpoints=(gate,),
            pallas_silu_checkpoints=(mathematical,),
            control_silu_checkpoints=(mathematical,),
            policy=policy,
            scenario=scenario,
        )


def test_strict_silu_stablehlo_rejects_identity_and_relu_substitutions() -> None:
    identity = """module {
      func.func @main(
        %arg0: tensor<1x4xbf16>, %other: tensor<1x4xbf16>
      ) -> tensor<1x4xbf16> {
        %1 = stablehlo.optimization_barrier %arg0 : tensor<1x4xbf16>
        %2 = stablehlo.optimization_barrier %1 : tensor<1x4xbf16>
        %3 = stablehlo.multiply %other, %2 : tensor<1x4xbf16>
        return %3 : tensor<1x4xbf16>
      }
    }"""
    relu = """module {
      func.func private @relu(tensor<1x4xbf16>) -> tensor<1x4xbf16>
      func.func @main(
        %arg0: tensor<1x4xbf16>, %other: tensor<1x4xbf16>
      ) -> tensor<1x4xbf16> {
        %1 = stablehlo.optimization_barrier %arg0 : tensor<1x4xbf16>
        %2 = func.call @relu(%1) : (tensor<1x4xbf16>) -> tensor<1x4xbf16>
        %3 = stablehlo.optimization_barrier %2 : tensor<1x4xbf16>
        %4 = stablehlo.multiply %other, %3 : tensor<1x4xbf16>
        return %4 : tensor<1x4xbf16>
      }
    }"""

    with pytest.raises(ValueError, match="expected 1 calls"):
        validate_strict_silu_stablehlo(identity, expected_count=1)
    with pytest.raises(ValueError, match="expected 1 calls"):
        validate_strict_silu_stablehlo(relu, expected_count=1)


def test_bf16_forward_collective_drop_mutation_fails_physical_verification() -> None:
    scenario = default_seqax_bf16_validation_contract().scenarios[0]
    distributed = seqax_forward_schedule(
        **scenario.parameters.model_dump(),
        numerical_semantics=SeqaxNumericalSemantics.TYPED_BF16_V1,
    )
    physical = canonical_module_text(lower_seqax_forward_to_physical(distributed).module)
    removed = False
    mutant_lines = []
    for line in physical.splitlines():
        if not removed and '"tpu_schedule.collective"' in line and "reduce_scatter" in line:
            removed = True
            continue
        mutant_lines.append(line)
    mutant = "\n".join(mutant_lines) + "\n"

    assert removed
    with pytest.raises(
        VerifyException, match="reads a buffer before its producing operation completes"
    ):
        _parse_physical(mutant).verify()


def test_bf16_forward_discriminator_clause_inventory_is_complete_and_typed() -> None:
    contract = default_seqax_bf16_validation_contract()
    clauses = tuple(
        seqax_discriminator_clause(discriminator)
        for discriminator in contract.required_discriminators
    )

    assert len(clauses) == len(contract.required_discriminators)
    assert clauses[0].value == "strict_hlo_structure"
    assert clauses[4].value == "physical_schedule_verification"
    assert clauses[-1].value == "output_shape"
    with pytest.raises(TypeError, match="must be typed"):
        seqax_discriminator_clause("localized_spike")  # type: ignore[arg-type]
