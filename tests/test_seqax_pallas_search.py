from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from tpu_cake.contracts import RuntimeIdentity
from tpu_cake.dialects.tpu_schedule import MxuEinsumOp
from tpu_cake.seqax_pallas_search import (
    SeqaxPallasRoundObservation,
    candidate_statistics,
    candidate_tiles,
    confirmation_statistics,
    default_seqax_pallas_search_contract,
    execution_orders,
)
from tpu_cake.seqax_pallas_search_runner import (
    _compiler_tile_metadata,
    _load_primitive_input,
    _regenerate_primitive_operands,
    _save_primitive_input,
    _validate_output_abi,
    prepare_seqax_pallas_candidates,
)


def _runtime() -> RuntimeIdentity:
    return RuntimeIdentity(
        python="3.12.3",
        jax="0.11.0",
        jaxlib="0.11.0",
        libtpu="0.0.44.1",
        xla=" --xla_tpu_use_enhanced_launch_barrier=true",
    )


def _observation(
    round_index: int,
    position: int,
    candidate: str,
    median: int,
) -> SeqaxPallasRoundObservation:
    return SeqaxPallasRoundObservation(
        round_index=round_index,
        position=position,
        candidate=candidate,
        samples_ns=(median - 2, median - 1, median, median + 1, median + 2),
        median_ns=float(median),
    )


def test_seqax_pallas_search_contract_is_canonical_and_balanced() -> None:
    contract = default_seqax_pallas_search_contract(_runtime())
    orders = execution_orders(contract)

    assert len(orders) == 8
    assert (
        contract.search_id
        == contract.model_validate_json(
            contract.model_dump_json(exclude_computed_fields=True)
        ).search_id
    )
    assert all(
        [order[position] for order in orders].count(candidate.name) == 2
        for position in range(4)
        for candidate in contract.candidates
    )
    assert contract.timing_seed in contract.correctness_seeds


def test_checked_in_seqax_pallas_search_contract_is_the_canonical_contract() -> None:
    saved = default_seqax_pallas_search_contract(_runtime()).model_validate_json(
        Path("contracts/seqax-physical-pallas-tile-search.json").read_text()
    )

    assert saved == default_seqax_pallas_search_contract(_runtime())


def test_seqax_pallas_search_rejects_a_noncanonical_protocol() -> None:
    contract = default_seqax_pallas_search_contract(_runtime())

    with pytest.raises(ValidationError, match="measurement protocol"):
        contract.model_validate(
            {**contract.model_dump(exclude_computed_fields=True), "measured_iterations": 4}
        )


def test_seqax_pallas_search_rejects_a_candidate_subset() -> None:
    contract = default_seqax_pallas_search_contract(_runtime())

    with pytest.raises(ValidationError, match="candidate set"):
        contract.model_validate(
            {
                **contract.model_dump(exclude_computed_fields=True),
                "candidates": contract.model_dump(exclude_computed_fields=True)["candidates"][:2],
            }
        )


def test_seqax_pallas_candidate_policies_create_distinct_realized_schedules() -> None:
    contract = default_seqax_pallas_search_contract(_runtime())
    full = (
        (1, 256, 8),
        (1, 8, 256),
        (1, 256, 256),
        (1, 4, 4),
    )
    realized = tuple(candidate_tiles(full, candidate) for candidate in contract.candidates)

    assert len(set(realized)) == 4
    assert realized[0] == full
    assert realized[1][0] == (1, 128, 8)
    assert realized[2][1] == (1, 8, 128)
    assert realized[3][2] == (1, 128, 128)


def test_seqax_pallas_search_prepares_the_exact_physical_candidate_set() -> None:
    contract = default_seqax_pallas_search_contract(_runtime())

    _distributed, prepared = prepare_seqax_pallas_candidates(contract)

    assert tuple(value.candidate.name for value in prepared) == (
        "incumbent",
        "split-k",
        "split-n",
        "split-kn",
    )
    assert tuple(value.candidate.expected_changed_regions for value in prepared) == (
        0,
        5,
        2,
        7,
    )
    assert len({value.plan.physical_schedule_sha256 for value in prepared}) == 4
    assert len({value.plan.source_sha256() for value in prepared}) == 4
    assert all(len(value.tiles) == 9 for value in prepared)


def test_seqax_pallas_primitive_operands_are_seeded_and_typed() -> None:
    contract = default_seqax_pallas_search_contract(_runtime())
    _distributed, prepared = prepare_seqax_pallas_candidates(contract)
    operation = next(
        value for value in prepared[-1].physical.walk() if isinstance(value, MxuEinsumOp)
    )

    lhs, rhs = _regenerate_primitive_operands(
        operation,
        contract.correctness_seeds[0],
    )
    repeated_lhs, repeated_rhs = _regenerate_primitive_operands(
        operation,
        contract.correctness_seeds[0],
    )

    assert str(lhs.dtype) == "bfloat16"
    assert str(rhs.dtype) == "bfloat16"
    assert np.array_equal(lhs, repeated_lhs)
    assert np.array_equal(rhs, repeated_rhs)


def test_seqax_pallas_primitive_bf16_storage_round_trips_exactly(tmp_path: Path) -> None:
    contract = default_seqax_pallas_search_contract(_runtime())
    _distributed, prepared = prepare_seqax_pallas_candidates(contract)
    operation = next(
        value for value in prepared[-1].physical.walk() if isinstance(value, MxuEinsumOp)
    )
    value, _rhs = _regenerate_primitive_operands(operation, contract.correctness_seeds[0])
    path = tmp_path / "input.npy"

    _save_primitive_input(path, value, "bf16")

    stored = np.load(path, allow_pickle=False)
    restored = _load_primitive_input(path, "bf16")
    assert stored.dtype == np.dtype(np.uint16)
    assert restored.dtype == value.dtype
    assert np.array_equal(restored, value)

    np.save(path, value.astype(np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match="PRIMITIVE_STORAGE_DTYPE"):
        _load_primitive_input(path, "bf16")


def test_seqax_pallas_output_must_match_the_plan_abi() -> None:
    contract = default_seqax_pallas_search_contract(_runtime())
    _distributed, prepared = prepare_seqax_pallas_candidates(contract)
    output_contract = prepared[0].plan.output_contracts[0]
    shape = tuple(size for _, size in output_contract.shape)

    _validate_output_abi(np.zeros(shape, dtype=np.float32), output_contract, "incumbent")
    with pytest.raises(ValueError, match="OUTPUT_ABI_MISMATCH"):
        _validate_output_abi(np.zeros(shape, dtype=np.float64), output_contract, "incumbent")


def test_seqax_pallas_compiler_tiles_are_scoped_to_custom_calls() -> None:
    schedule = "a" * 64
    compiler_hlo = f'''HloModule test
ENTRY main {{
  pallas_call.0 = f32[1] custom-call(), custom_call_target="tpu_custom_call", frontend_attributes={{kernel_metadata={{
"region_index":0,
"schedule_sha256":"{schedule}",
"tile_k":128,
"tile_m":8,
"tile_n":128
}}}}, backend_config={{}}
}}
'''

    assert _compiler_tile_metadata(compiler_hlo) == ((0, schedule, 8, 128, 128),)
    assert (
        _compiler_tile_metadata(
            'comment: "region_index":0, "schedule_sha256":"'
            + schedule
            + '", "tile_k":128, "tile_m":8, "tile_n":128'
        )
        == ()
    )


def test_seqax_pallas_compiler_tiles_use_semantic_region_order() -> None:
    schedule = "a" * 64

    def custom_call(region: int, tile_k: int, *, root: bool = False) -> str:
        prefix = "ROOT " if root else ""
        return f'''  {prefix}pallas_call.{region + 20} = f32[1] custom-call(), custom_call_target="tpu_custom_call", frontend_attributes={{kernel_metadata={{
"region_index":{region},
"schedule_sha256":"{schedule}",
"tile_k":{tile_k},
"tile_m":8,
"tile_n":128
}}}}, backend_config={{}}'''

    compiler_hlo = "\n".join(
        (
            "HloModule test",
            "ENTRY main {",
            custom_call(1, 256),
            custom_call(0, 128, root=True),
            "}",
        ),
    )

    assert _compiler_tile_metadata(compiler_hlo) == (
        (0, schedule, 8, 128, 128),
        (1, schedule, 8, 256, 128),
    )

    duplicate = compiler_hlo.replace(custom_call(1, 256), custom_call(0, 256))
    with pytest.raises(ValueError, match="DUPLICATE_COMPILER_REGION"):
        _compiler_tile_metadata(duplicate)


def test_seqax_pallas_statistics_promote_only_a_clear_matched_winner() -> None:
    contract = default_seqax_pallas_search_contract(_runtime())
    orders = execution_orders(contract)
    durations = {
        "incumbent": (100, 101, 99, 100, 102, 98, 100, 101),
        "split-k": (80, 81, 79, 80, 82, 78, 80, 81),
        "split-n": (99, 102, 100, 101, 103, 98, 100, 102),
        "split-kn": (95, 105, 94, 106, 95, 105, 94, 106),
    }
    observations = tuple(
        _observation(round_index, position, name, durations[name][round_index])
        for round_index, order in enumerate(orders)
        for position, name in enumerate(order)
    )

    by_name = {item.name: item for item in candidate_statistics(contract, observations)}

    assert by_name["split-k"].promotable is True
    assert by_name["split-k"].improvement_confidence_interval[0] > 0.03
    assert by_name["split-n"].promotable is False
    assert by_name["split-kn"].promotable is False


def test_seqax_pallas_statistics_reject_missing_rounds() -> None:
    contract = default_seqax_pallas_search_contract(_runtime())
    observations = tuple(
        _observation(round_index, position, name, 100)
        for round_index, order in enumerate(execution_orders(contract))
        for position, name in enumerate(order)
    )[:-1]

    with pytest.raises(ValueError, match="execution order"):
        candidate_statistics(contract, observations)


def test_seqax_pallas_statistics_reject_reordered_or_short_samples() -> None:
    contract = default_seqax_pallas_search_contract(_runtime())
    observations = tuple(
        _observation(round_index, position, name, 100)
        for round_index, order in enumerate(execution_orders(contract))
        for position, name in enumerate(order)
    )
    reordered = (observations[1], observations[0], *observations[2:])
    short = observations[0].model_copy(update={"samples_ns": (99, 100, 101), "median_ns": 100.0})

    with pytest.raises(ValueError, match="execution order"):
        candidate_statistics(contract, reordered)
    with pytest.raises(ValueError, match="sample count"):
        candidate_statistics(contract, (short, *observations[1:]))


def test_seqax_pallas_confirmation_is_fresh_and_order_balanced() -> None:
    contract = default_seqax_pallas_search_contract(_runtime())
    observations = tuple(
        _observation(
            round_index,
            position,
            name,
            100 if name == contract.baseline else 80,
        )
        for round_index in range(contract.confirmation_rounds)
        for position, name in enumerate(
            (contract.baseline, "split-k")
            if round_index % 2 == 0
            else ("split-k", contract.baseline)
        )
    )

    result = confirmation_statistics(contract, "split-k", observations)

    assert result.confirmed
    assert result.improvement_confidence_interval[0] > 0.03
    assert {order for order in result.execution_orders} == {
        ("incumbent", "split-k"),
        ("split-k", "incumbent"),
    }
