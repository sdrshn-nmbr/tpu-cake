import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tpu_cake.rpa_surface import (
    INKLING_SHARDED_RPA_CORRECTNESS_SEEDS,
    InklingShardedRpaSurfaceContract,
    default_inkling_sharded_rpa_surface_contract,
)

_CALIBRATION_SEEDS = {20260820, 20260821, 20260822, 20260823, 20260824, 20260825}


def _payload() -> dict:
    return json.loads(Path("contracts/inkling-sharded-rpa-surface.json").read_text())


def test_sharded_rpa_surface_contract_is_external_and_canonical() -> None:
    saved = InklingShardedRpaSurfaceContract.model_validate_json(
        Path("contracts/inkling-sharded-rpa-surface.json").read_text()
    )
    generated = default_inkling_sharded_rpa_surface_contract()

    assert saved == generated
    assert saved.surface_id == ("641219820406fb73c1adff4da86482e7effcc236e1b8b1111dc2a3d00882620d")
    assert not set(INKLING_SHARDED_RPA_CORRECTNESS_SEEDS) & _CALIBRATION_SEEDS
    assert saved.plan.compiler_hlo_authority == (
        "receipt-bound-raw-bytes-not-reproducible-identity"
    )


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("hlo_identity_status",), "pending"),
        (("output_relative_l2_error",), 0.007),
        (("plan", "stablehlo_sha256"), "0" * 64),
        (("plan", "mesh_shape"), [1, 8]),
        (("runtime", "jax"), "0.11.1"),
    ),
)
def test_sharded_rpa_surface_contract_rejects_coordinated_policy_drift(
    path: tuple[str, ...],
    value: object,
) -> None:
    payload = _payload()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises((ValidationError, ValueError)):
        InklingShardedRpaSurfaceContract.model_validate_json(json.dumps(payload))
