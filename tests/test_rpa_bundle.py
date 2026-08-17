import pytest

from tpu_cake.contracts import RuntimeIdentity
from tpu_cake.rpa_bundle import (
    _require_shared_canonical_identity,
    _require_timing_sample_protocol,
)
from tpu_cake.rpa_runner import FusedRpaRunResult
from tpu_cake.workloads import inkling_fused_rpa_experiment


def _result(**updates) -> FusedRpaRunResult:
    experiment = inkling_fused_rpa_experiment()
    values = {
        "schedule_sha256": experiment.schedule_sha256,
        "pallas_source_sha256": "1" * 64,
        "stablehlo_sha256": "6" * 64,
        "compiler_hlo_sha256": "7" * 64,
        "input_sha256": ("2" * 64,),
        "output_sha256": ("3" * 64,),
        "oracle_sha256": ("4" * 64,),
        "backend_manifest": (),
        "backend_executor": "module.ragged_paged_attention",
        "backend_executor_sha256": "8" * 64,
        "runtime": RuntimeIdentity(python="3.13", jax="0.11.0"),
        "backend": "tpu",
        "device_kind": "TPU7x",
        "device_count": 8,
        "execution_scope": "local-shard-caller-owned-sharding",
    }
    values.update(updates)
    return FusedRpaRunResult.model_construct(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schedule_sha256", "a" * 64),
        ("pallas_source_sha256", "b" * 64),
        ("input_sha256", ("c" * 64,)),
        ("output_sha256", ("d" * 64,)),
        ("oracle_sha256", ("e" * 64,)),
        ("device_count", 4),
    ),
)
def test_cross_phase_identity_rejects_one_changed_claim(field: str, value: object) -> None:
    experiment = inkling_fused_rpa_experiment()

    with pytest.raises(ValueError, match="CANONICAL_EXECUTION_IDENTITY"):
        _require_shared_canonical_identity(
            (_result(), _result(**{field: value}), _result()),
            experiment,
        )


def test_cross_phase_identity_requires_current_canonical_schedule() -> None:
    experiment = inkling_fused_rpa_experiment()
    stale = _result(schedule_sha256="a" * 64)

    with pytest.raises(ValueError, match="CANONICAL_EXECUTION_IDENTITY"):
        _require_shared_canonical_identity((stale, stale, stale), experiment)


@pytest.mark.parametrize("samples", ((1,), (1,) * 49, (1,) * 51, (0,) * 50))
def test_timing_sample_protocol_requires_exactly_fifty_positive_samples(
    samples: tuple[int, ...],
) -> None:
    result = _result(samples_ns=samples, measured_iterations=50)

    with pytest.raises(ValueError, match="TIMING_SAMPLE_PROTOCOL"):
        _require_timing_sample_protocol(result)


def test_timing_sample_protocol_accepts_declared_run() -> None:
    _require_timing_sample_protocol(_result(samples_ns=(1,) * 50, measured_iterations=50))
