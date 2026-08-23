from __future__ import annotations

import ast
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from pydantic import ValidationError

from tpu_cake.seqax_numerical import (
    SeqaxBf16CheckpointContract,
    SeqaxBf16ScenarioParameters,
    _assess_seqax_bf16_outputs,
    default_seqax_bf16_validation_contract,
    rounded_mathematical_silu_bf16,
    seqax_bf16_checkpoint_contract,
)
from tpu_cake.seqax_silu_fusion_correctness import (
    SeqaxSiluFusionCorrectnessContract,
)
from tpu_cake.seqax_silu_fusion_correctness_worker import (
    _boundary_discriminator,
    _silu_float32,
)
from tpu_cake.workloads.seqax_oracle import (
    seqax_forward_canonical_reference,
    seqax_forward_inputs,
)

_ROOT = Path(__file__).resolve().parents[1]


def _parameters() -> SeqaxBf16ScenarioParameters:
    contract = SeqaxSiluFusionCorrectnessContract.model_validate_json(
        (_ROOT / "contracts/seqax-silu-fusion-correctness-v1.json").read_text()
    )
    values = dict(contract.parameters)
    values.pop("numerical_semantics")
    return SeqaxBf16ScenarioParameters.model_validate(values)


def _boundary_checkpoints(
    *,
    gate: float,
    up: float,
) -> tuple[tuple[np.ndarray, ...], ...]:
    bfloat16 = np.dtype(ml_dtypes.bfloat16)
    gate_value = np.asarray([gate], dtype=bfloat16)
    up_value = np.asarray([up], dtype=bfloat16)
    silu = rounded_mathematical_silu_bf16(gate_value)
    hidden = np.asarray(
        silu.astype(np.float32) * up_value.astype(np.float32),
        dtype=bfloat16,
    )
    values = [np.asarray([0.0], dtype=bfloat16) for _ in range(13)]
    values[6] = gate_value
    values[7] = silu
    values[9] = up_value
    values[10] = hidden
    return tuple((value,) for value in values)


def test_generic_checkpoint_contract_has_exact_silu_fusion_abi() -> None:
    contract = seqax_bf16_checkpoint_contract(_parameters())

    assert contract.output.shape == (256, 1, 16)
    assert contract.output.dtype == "float32"
    assert contract.gate_checkpoints[0].shape == (256, 1, 4096)
    assert contract.hidden_checkpoints[0].dtype == "bfloat16"
    assert contract.down_bfloat16_checkpoints[0].shape == (256, 1, 32)


def test_generic_checkpoint_contract_rejects_shape_mutation() -> None:
    contract = seqax_bf16_checkpoint_contract(_parameters())
    payload = contract.model_dump()
    payload["hidden_checkpoints"][0]["shape"] = (256, 1, 2048)

    with pytest.raises(ValidationError, match="checkpoint ABI mismatch"):
        SeqaxBf16CheckpointContract.model_validate(payload)


def test_generic_checkpoint_assessment_requires_and_accepts_declared_seed() -> None:
    parameters = SeqaxBf16ScenarioParameters(
        batch=2,
        sequence=1,
        model=8,
        vocabulary=16,
        feed_forward=16,
        query_groups=2,
        key_value_heads=4,
        head=4,
        layers=1,
        data_mesh=2,
        tensor_mesh=4,
        rope_max_timescale=256,
    )
    contract = seqax_bf16_checkpoint_contract(parameters)
    seed = 9173
    inputs = tuple(
        np.asarray(value) for value in seqax_forward_inputs(seed=seed, **parameters.model_dump())
    )
    output = np.asarray(
        seqax_forward_canonical_reference(
            inputs,
            quantization_decimals=6,
            **parameters.model_dump(),
        )
    )
    policy = default_seqax_bf16_validation_contract().policy

    assessment, expected_inputs = _assess_seqax_bf16_outputs(
        output,
        output,
        seed=seed,
        inputs=inputs,
        policy=policy,
        scenario=contract,
        declared_seeds=(seed,),
    )

    assert assessment.final_outputs_satisfy_policy
    assert all(
        np.array_equal(actual, expected)
        for actual, expected in zip(inputs, expected_inputs, strict=True)
    )
    with pytest.raises(ValueError, match="seed is not declared"):
        _assess_seqax_bf16_outputs(
            output,
            output,
            seed=seed,
            inputs=inputs,
            policy=policy,
            scenario=contract,
            declared_seeds=(seed + 1,),
        )


def test_boundary_discriminator_proves_intermediate_bf16_round(tmp_path: Path) -> None:
    checkpoints = _boundary_checkpoints(gate=-1.640625, up=-0.89453125)

    difference_count = _boundary_discriminator(
        tmp_path,
        (checkpoints, checkpoints),
    )

    assert difference_count == 1
    assert (tmp_path / "strict_hidden.npy").is_file()
    assert (tmp_path / "mutant_hidden.npy").is_file()


def test_boundary_discriminator_rejects_nondiscriminating_input(tmp_path: Path) -> None:
    checkpoints = _boundary_checkpoints(gate=0.0, up=1.0)

    with pytest.raises(ValueError, match="BOUNDARY_DISCRIMINATOR_FAILED"):
        _boundary_discriminator(tmp_path, (checkpoints, checkpoints))


def test_boundary_mutant_uses_unrounded_float32_silu() -> None:
    gate = np.asarray([-1.640625], dtype=ml_dtypes.bfloat16)

    assert _silu_float32(gate).dtype == np.float32
    assert not np.array_equal(
        np.asarray(_silu_float32(gate), dtype=ml_dtypes.bfloat16),
        _silu_float32(gate),
    )


def test_worker_source_has_no_timing_or_profile_calls() -> None:
    path = _ROOT / "src/tpu_cake/seqax_silu_fusion_correctness_worker.py"
    tree = ast.parse(path.read_text())
    imported_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "time" not in imported_roots
    assert "perf_counter" not in called_names
    assert "profiler" not in path.read_text()
