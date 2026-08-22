from __future__ import annotations

from pathlib import Path

import jax
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import Mesh

from tpu_cake.matmul_collective_surface_correctness import make_correctness_operand_shard
from tpu_cake.matmul_collective_surface_correctness_executor import (
    SurfaceCorrectnessWorkerRequest,
)
from tpu_cake.matmul_collective_surface_correctness_protocol import (
    MatmulCollectiveSurfaceCorrectnessProtocol,
)
from tpu_cake.matmul_collective_surface_correctness_worker import (
    _canonical_slice,
    _device_sentinel_hex,
    _error_metrics,
    _execution_order,
    _materialize_operand,
    _OperandCallback,
    _parent_consensus,
    _save_array_exclusive,
    _validate_empty_compilation_cache,
    _validate_worker_authorization,
    _verify_resident_sentinels,
)
from tpu_cake.matmul_collective_surface_prediction import (
    MatmulCollectiveSurfaceDesignContract,
    MatmulCollectiveSurfaceScenario,
    MatmulCollectiveSurfaceSplit,
)
from tpu_cake.matmul_collective_surface_runner import (
    CompileCaptureRecord,
    MatmulCollectiveSurfaceCompileReport,
    SurfaceCompileStatus,
)

PROTOCOL = MatmulCollectiveSurfaceCorrectnessProtocol.model_validate_json(
    Path("contracts/matmul-collective-surface-correctness-v1.json").read_text()
)
DESIGN = MatmulCollectiveSurfaceDesignContract.model_validate_json(
    Path("contracts/matmul-collective-surface-design-v1.json").read_text()
)


def _request() -> SurfaceCorrectnessWorkerRequest:
    return SurfaceCorrectnessWorkerRequest(
        attempt_id="a" * 64,
        split=MatmulCollectiveSurfaceSplit.CALIBRATION,
        invocation_nonce="b" * 64,
        execution_authority_sha256="c" * 64,
        parent_snapshot_path="/evidence/parent_compile",
        protocol=PROTOCOL,
        design=DESIGN,
    )


def _small_scenario() -> MatmulCollectiveSurfaceScenario:
    return MatmulCollectiveSurfaceScenario(
        name="calibration-0",
        split=MatmulCollectiveSurfaceSplit.CALIBRATION,
        m=16,
        k=1024,
        n=1024,
        tile_m=16,
        tile_n=128,
    )


def _capture(
    repetition: int,
    *,
    semantic_compiler_hlo_sha256: str = "8" * 64,
) -> CompileCaptureRecord:
    return CompileCaptureRecord(
        scenario_name="calibration-0",
        strategy=PROTOCOL.strategies[0],
        repetition=repetition,
        input_contract_sha256="1" * 64,
        distributed_schedule_sha256="2" * 64,
        physical_schedule_sha256="3" * 64,
        pallas_source_sha256="4" * 64,
        status=SurfaceCompileStatus.SUCCEEDED,
        stablehlo="stable\n",
        compiler_hlo="compiler\n",
        stablehlo_sha256="5" * 64,
        semantic_stablehlo_sha256="6" * 64,
        compiler_hlo_sha256="7" * 64,
        semantic_compiler_hlo_sha256=semantic_compiler_hlo_sha256,
    )


def test_operand_callback_captures_full_host_payload_and_semantic_sentinels() -> None:
    request = _request()
    scenario = _small_scenario()
    callback = _OperandCallback(
        request=request,
        scenario=scenario,
        pattern="signed-periodic",
        role="lhs",
    )

    shards = tuple(
        callback((slice(0, 16), slice(device * 128, (device + 1) * 128))) for device in range(8)
    )

    assert len(callback.captures) == 8
    assert all(value.shape == (16, 128) for value in shards)
    assert all(value.dtype == np.dtype(ml_dtypes.bfloat16) for value in shards)
    assert all(
        value.host_callback_payload_nbytes == 16 * 128 * 2 for value in callback.captures.values()
    )
    assert all(len(value.expected_sentinel_hex) == 32 for value in callback.captures.values())
    first = callback.captures[((0, 16), (0, 128))]
    assert len(first.host_callback_payload_sha256) == 64

    with pytest.raises(ValueError, match="SHARD_REPEATED"):
        callback((slice(0, 16), slice(0, 128)))


@pytest.mark.parametrize(
    "index",
    (
        (0, slice(None)),
        (slice(None), slice(None, None, 2)),
        (slice(-1, 4), slice(None)),
        (slice(0, 0), slice(None)),
    ),
)
def test_canonical_slice_rejects_ambiguous_callback_indexes(index) -> None:
    with pytest.raises((TypeError, ValueError), match="CALLBACK_INDEX_INVALID"):
        _canonical_slice(index, (16, 128))


def test_device_sentinel_read_preserves_bfloat16_payload_bits() -> None:
    value = np.arange(64, dtype=np.float32).reshape(8, 8).astype(ml_dtypes.bfloat16)
    coordinates = tuple((index // 8, index % 8) for index in range(32))

    observed = _device_sentinel_hex(value, coordinates)
    expected = tuple(value[coordinate].tobytes().hex() for coordinate in coordinates)

    assert observed == expected


def test_compilation_cache_must_be_absolute_existing_and_empty(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(cache))
    assert _validate_empty_compilation_cache() == cache

    (cache / "reused").write_text("compiled")
    with pytest.raises(ValueError, match="CACHE_NOT_EMPTY"):
        _validate_empty_compilation_cache()


def test_worker_authorization_rejects_substituted_request_path(tmp_path) -> None:
    canonical = tmp_path / "worker-request.json"
    substituted = tmp_path / "substituted.json"
    canonical.write_text("{}")
    substituted.write_text("{}")

    with pytest.raises(ValueError, match="REQUEST_PATH_MISMATCH"):
        _validate_worker_authorization(tmp_path, substituted, _request())


@pytest.mark.parametrize("role", ("lhs", "rhs"))
def test_materialize_operand_binds_host_payload_to_actual_device_sentinels(
    monkeypatch,
    role: str,
) -> None:
    request = _request()
    scenario = _small_scenario()

    class Device:
        def __init__(self, device_id: int) -> None:
            self.id = device_id
            self.process_index = 0

    class Shard:
        def __init__(self, device_id: int, index, data) -> None:
            self.device = Device(device_id)
            self.index = index
            self.data = data

    class Array:
        def __init__(self, shards) -> None:
            self.addressable_shards = shards

    def fake_make_array(shape, _sharding, callback):
        shards = []
        for device in range(8):
            index = (
                (slice(0, shape[0]), slice(device * 128, (device + 1) * 128))
                if role == "lhs"
                else (slice(device * 128, (device + 1) * 128), slice(0, shape[1]))
            )
            shards.append(Shard(device, index, callback(index)))
        return Array(tuple(shards))

    monkeypatch.setattr(jax, "make_array_from_callback", fake_make_array)
    mesh = Mesh(np.asarray([jax.devices()[0]]), ("t",))

    resident, identities = _materialize_operand(
        request,
        scenario,
        "block-diagonal",
        role,
        mesh,
    )

    assert tuple(value.device_id for value in identities) == tuple(range(8))
    assert all(
        sentinel.observed_bfloat16_hex == sentinel.expected_bfloat16_hex
        for identity in identities
        for sentinel in identity.sentinels
    )
    expected = make_correctness_operand_shard(
        "block-diagonal",
        role,
        m=16,
        k=1024,
        n=1024,
        k_start=0,
        k_stop=128,
    )
    assert identities[0].host_callback_payload_sha256 != "0" * 64
    assert identities[0].host_callback_payload_nbytes == expected.nbytes
    _verify_resident_sentinels(resident, identities)

    resident.addressable_shards[0].data[identities[0].sentinels[0].local_coordinate] = 7
    with pytest.raises(ValueError, match="RESIDENT_SENTINEL_CHANGED"):
        _verify_resident_sentinels(resident, identities)


def test_saved_arrays_are_exclusive_and_hash_the_loaded_payload(tmp_path: Path) -> None:
    value = np.arange(32, dtype=np.float32).reshape(4, 8)

    saved = _save_array_exclusive(tmp_path, "outputs/case/oracle.npy", value)
    loaded = np.load(tmp_path / saved.path, allow_pickle=False)

    assert loaded.dtype == np.dtype("<f4")
    np.testing.assert_array_equal(loaded, value)
    assert saved.file_sha256 != saved.array_sha256
    with pytest.raises(FileExistsError):
        _save_array_exclusive(tmp_path, "outputs/case/oracle.npy", value)


def test_error_metrics_use_the_declared_combined_tolerance() -> None:
    oracle = np.ascontiguousarray(np.asarray([[0.0, 10.0]], dtype=np.float32))
    passing = np.ascontiguousarray(np.asarray([[0.001, 10.011]], dtype=np.float32))
    failing = np.ascontiguousarray(np.asarray([[0.00101, 10.0111]], dtype=np.float32))

    assert (
        _error_metrics(
            passing,
            oracle,
            absolute_tolerance=0.001,
            relative_tolerance=0.001,
        )[0]
        == 0
    )
    assert (
        _error_metrics(
            failing,
            oracle,
            absolute_tolerance=0.001,
            relative_tolerance=0.001,
        )[0]
        == 2
    )


def test_execution_order_is_exact_abba_then_baab() -> None:
    first, second = PROTOCOL.strategies

    assert _execution_order(PROTOCOL.strategies, 0) == (first, second, second, first)
    assert _execution_order(PROTOCOL.strategies, 1) == (second, first, first, second)


def test_parent_consensus_rejects_a_semantic_compiler_disagreement() -> None:
    valid = MatmulCollectiveSurfaceCompileReport(
        design_id="a" * 64,
        source_authority_sha256="b" * 64,
        execution_authority_sha256="c" * 64,
        captures=(_capture(1), _capture(2)),
    )
    assert _parent_consensus(valid, "calibration-0", PROTOCOL.strategies[0]).repetition == 1

    broken = valid.model_copy(
        update={"captures": (_capture(1), _capture(2, semantic_compiler_hlo_sha256="9" * 64))}
    )
    with pytest.raises(ValueError, match="PARENT_CONSENSUS_INVALID"):
        _parent_consensus(broken, "calibration-0", PROTOCOL.strategies[0])
