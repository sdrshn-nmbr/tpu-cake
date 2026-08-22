from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from tpu_cake import xprof_evidence
from tpu_cake.contracts import ArtifactReference, ArtifactRole, RuntimeIdentity
from tpu_cake.rpa_bundle import (
    _decode_custom_call_durations,
    _load_declared_array,
    _require_shared_canonical_identity,
    _require_timing_sample_protocol,
    _timing_metrics,
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


def test_bfloat16_artifact_round_trip_restores_declared_dtype(tmp_path: Path) -> None:
    path = tmp_path / "values.npy"
    expected = np.asarray(jnp.arange(4, dtype=jnp.bfloat16))
    np.save(path, expected, allow_pickle=False)

    observed = _load_declared_array(path, "bfloat16")

    assert str(observed.dtype) == "bfloat16"
    np.testing.assert_array_equal(observed, expected)


def test_decode_custom_call_duration_requires_fifty_tpu_xla_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xplane = tmp_path / "profile/capture.xplane.pb"
    xplane.parent.mkdir()
    xplane.write_bytes(b"xplane")
    event_name = (
        '%RPAd-p_16-bq_8_8-bkv_128_128.1 custom-call(), custom_call_target="tpu_custom_call"'
    )
    colliding_event_name = (
        '%RPAd-p_16-bq_8_8-bkv_128_1280.1 custom-call(), custom_call_target="tpu_custom_call"'
    )

    class FakeProfile:
        planes = (
            SimpleNamespace(
                name="/device:TPU:0",
                stats={},
                lines=(
                    SimpleNamespace(
                        name="XLA Ops",
                        events=tuple(
                            SimpleNamespace(
                                name=event_name,
                                start_ns=index,
                                duration_ns=float(index + 1),
                            )
                            for index in range(50)
                        )
                        + (
                            SimpleNamespace(
                                name=colliding_event_name,
                                start_ns=51,
                                duration_ns=999.0,
                            ),
                        ),
                    ),
                ),
            ),
        )

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        xprof_evidence.profile_data.ProfileData,
        "from_file",
        lambda _: FakeProfile(),
    )

    assert _decode_custom_call_durations(tmp_path, (8, 128, 8, 128), 50) == tuple(
        float(index + 1) for index in range(50)
    )
    with pytest.raises(ValueError, match="DECODE_CUSTOM_CALL_PROTOCOL"):
        _decode_custom_call_durations(tmp_path, (8, 128, 8, 128), 49)


def test_timing_metric_preserves_half_nanosecond_median(tmp_path: Path) -> None:
    result_path = tmp_path / "timing/result.json"
    result_path.parent.mkdir()
    result_path.write_text("{}")
    result = _result(
        samples_ns=tuple(range(1, 51)),
        median_ns=25,
        p90_ns=45,
        coefficient_of_variation=0.1,
        artifacts=(
            ArtifactReference(
                path="result.json",
                size_bytes=2,
                sha256="0" * 64,
                role=ArtifactRole.TIMING_SAMPLES,
            ),
        ),
    )

    median_metric = _timing_metrics(tmp_path, result)[0]

    assert str(median_metric.quantity.value) == "25.5"
