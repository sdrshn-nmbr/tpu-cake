import numpy as np
import pytest

from tpu_cake.contracts import (
    PHASE_REQUIRED_ROLES_BY_PROFILE,
    ArtifactRole,
    EvidenceProfile,
)
from tpu_cake.seqax_bundle import (
    _counter_experiment,
    _expected_result_role,
    _trusted_experiment,
    _validate_array_contract,
)


def test_seqax_receipt_profile_has_complete_phase_contracts() -> None:
    roles = PHASE_REQUIRED_ROLES_BY_PROFILE[
        EvidenceProfile.SEQAX_DISTRIBUTED_FORWARD
    ]

    assert len(roles) == 5
    assert all(phase_roles for phase_roles in roles.values())


def test_seqax_profile_binds_one_distributed_timed_program() -> None:
    experiment = _trusted_experiment()

    assert experiment.profile.required_timed_hlo_markers == (
        "all-gather",
        "reduce_scatter",
        "dot_general",
    )
    assert experiment.profile.minimum_tpu_device_planes == 8


def test_seqax_counter_capture_requires_hardware_counter_families() -> None:
    experiment = _counter_experiment(_trusted_experiment())

    assert experiment.profile.require_hbm_read_counters
    assert experiment.profile.require_hbm_write_counters
    assert experiment.profile.require_cycle_counters
    assert experiment.profile.minimum_counter_device_planes == 4


def test_seqax_array_contract_rejects_equal_values_with_the_wrong_dtype() -> None:
    with pytest.raises(ValueError, match="SEQAX_TENSOR_CONTRACT_MISMATCH"):
        _validate_array_contract(
            np.asarray([1, 2], dtype=np.int32),
            name="tokens",
            shape=(2,),
            dtype="uint32",
            phase="timing",
        )


def test_seqax_result_artifact_roles_come_from_trusted_paths() -> None:
    assert _expected_result_role("experiment.json") is ArtifactRole.EXPERIMENT
    assert _expected_result_role("inputs/12.npy") is ArtifactRole.CORRECTNESS_INPUT
    with pytest.raises(ValueError, match="PATH_UNRECOGNIZED"):
        _expected_result_role("renamed-experiment.json")
