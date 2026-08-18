from __future__ import annotations

import pytest

from tpu_cake.contracts import RuntimeIdentity
from tpu_cake.seqax_pallas_lowering import lower_seqax_physical_to_pallas
from tpu_cake.seqax_pallas_search import SEQAX_PALLAS_SEARCH_PARAMETERS
from tpu_cake.seqax_physical_lowering import lower_seqax_forward_to_physical
from tpu_cake.seqax_weight_placement import (
    SeqaxWeightPlacementContract,
    SeqaxWeightPlacementName,
    SeqaxWeightPlacementPolicy,
    default_seqax_weight_placement_contract,
    parameter_residency_bytes_per_device,
)
from tpu_cake.workloads.seqax_forward import (
    REPLICATED_EMBEDDING_FEED_FORWARD_WEIGHT_DATA,
    SHARDED_WEIGHT_DATA,
    SeqaxDataAxisPlacement,
    seqax_forward_schedule,
)


def _runtime() -> RuntimeIdentity:
    return RuntimeIdentity(
        python="3.12.3",
        jax="0.11.0",
        jaxlib="0.11.0",
        libtpu="0.0.44.1",
        xla="--xla_tpu_use_enhanced_launch_barrier=true",
    )


def _plan(placement):
    distributed = seqax_forward_schedule(
        **SEQAX_PALLAS_SEARCH_PARAMETERS,
        weight_data_placement=placement,
    )
    physical = lower_seqax_forward_to_physical(distributed).module
    return lower_seqax_physical_to_pallas(distributed, physical)


def test_weight_placement_contract_is_canonical_and_stable() -> None:
    first = default_seqax_weight_placement_contract(_runtime())
    second = SeqaxWeightPlacementContract.model_validate_json(
        first.model_dump_json(exclude_computed_fields=True)
    )

    assert first == second
    assert first.search_id == second.search_id
    assert tuple(candidate.name for candidate in first.candidates) == (
        SeqaxWeightPlacementName.SHARDED,
        SeqaxWeightPlacementName.EMBEDDING_MLP,
    )
    assert first.candidates[1].policy == SeqaxWeightPlacementPolicy(
        embedding=SeqaxDataAxisPlacement.REPLICATED,
        attention=SeqaxDataAxisPlacement.SHARDED,
        feed_forward=SeqaxDataAxisPlacement.REPLICATED,
    )


def test_weight_placement_contract_rejects_policy_and_protocol_drift() -> None:
    contract = default_seqax_weight_placement_contract(_runtime())
    payload = contract.model_dump(mode="python", exclude_computed_fields=True)
    payload["rounds"] = 8
    with pytest.raises(ValueError, match="measurement protocol"):
        SeqaxWeightPlacementContract.model_validate(payload)

    payload = contract.model_dump(mode="python", exclude_computed_fields=True)
    payload["candidates"][1]["policy"]["attention"] = "replicated"
    with pytest.raises(ValueError, match="candidate contracts"):
        SeqaxWeightPlacementContract.model_validate(payload)


def test_parameter_residency_is_derived_from_exact_local_input_contracts() -> None:
    sharded = _plan(SHARDED_WEIGHT_DATA)
    candidate = _plan(REPLICATED_EMBEDDING_FEED_FORWARD_WEIGHT_DATA)

    sharded_bytes = parameter_residency_bytes_per_device(
        sharded.input_contracts,
        mesh=dict(sharded.mesh),
    )
    candidate_bytes = parameter_residency_bytes_per_device(
        candidate.input_contracts,
        mesh=dict(candidate.mesh),
    )

    assert sharded_bytes == 22_912
    assert candidate_bytes == 33_152
    assert candidate_bytes - sharded_bytes == 10_240


def test_parameter_residency_rejects_an_incomplete_abi() -> None:
    plan = _plan(SHARDED_WEIGHT_DATA)
    with pytest.raises(ValueError, match="input contract count"):
        parameter_residency_bytes_per_device(
            plan.input_contracts[:-1],
            mesh=dict(plan.mesh),
        )
