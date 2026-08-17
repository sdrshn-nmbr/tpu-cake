from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tpu_cake import rpa_search
from tpu_cake.contracts import RuntimeIdentity
from tpu_cake.rpa_search import (
    RpaDeviceTiming,
    RpaSearchCandidate,
    RpaSearchContract,
    RpaSearchProfilerAdvancedConfiguration,
    RpaSearchProfilerContract,
    _archive_incomplete_run,
    _confirmation_statistics,
    _device_timing,
    _execution_orders,
    _incomplete_attempts,
    _statistics,
    _validated_profiler_config_sha256,
    validate_rpa_search_result,
)


def _candidate(name: str, blocks: tuple[int, int, int, int]) -> RpaSearchCandidate:
    return RpaSearchCandidate(
        name=name,
        query_block_size=blocks[0],
        kv_block_size=blocks[1],
        query_cluster_size=blocks[2],
        kv_cluster_size=blocks[3],
    )


def _contract() -> RpaSearchContract:
    return RpaSearchContract(
        baseline="incumbent",
        rounds=6,
        confirmation_rounds=6,
        bootstrap_samples=1_000,
        runtime=RuntimeIdentity(
            python="3.12.3",
            jax="0.11.0",
            jaxlib="0.11.0",
            libtpu="0.0.44.1",
            xla="test",
        ),
        backend="tpu",
        device_kind="TPU7x",
        device_count=8,
        profiler=RpaSearchProfilerContract(
            mode="trace",
            raise_error_on_start_failure=True,
            enable_hlo_proto=True,
            host_tracer_level=1,
            python_tracer_level=0,
            advanced_configuration=RpaSearchProfilerAdvancedConfiguration(
                tpu_num_chips_to_profile_per_task=4
            ),
            libtpu_init_args="test",
        ),
        candidates=(
            _candidate("incumbent", (8, 128, 8, 128)),
            _candidate("kv-64", (8, 64, 8, 64)),
            _candidate("kv-256", (8, 256, 8, 128)),
        ),
    )


def _result(median_ns: int) -> RpaDeviceTiming:
    return RpaDeviceTiming.model_construct(
        median_ns=median_ns,
        durations_ns=(float(median_ns),) * 50,
    )


def test_rpa_search_rejects_illegal_page_or_cluster_blocks() -> None:
    with pytest.raises(ValidationError, match="page size 16"):
        _candidate("bad-page", (8, 40, 8, 20))
    with pytest.raises(ValidationError, match="divide query block"):
        _candidate("bad-query", (8, 128, 3, 128))


def test_rpa_search_order_is_deterministic_and_balanced() -> None:
    orders = _execution_orders(_contract())

    assert orders == _execution_orders(_contract())
    assert len(orders) == 6
    assert all(set(order) == {"incumbent", "kv-64", "kv-256"} for order in orders)
    assert all(
        [order[position] for order in orders].count(candidate) == 2
        for position in range(3)
        for candidate in ("incumbent", "kv-64", "kv-256")
    )


def test_rpa_search_promotes_from_paired_run_level_measurements() -> None:
    contract = _contract()
    results = {
        "incumbent": [_result(value) for value in (100, 102, 98, 101, 99, 100)],
        "kv-64": [_result(value) for value in (80, 82, 78, 81, 79, 80)],
        "kv-256": [_result(value) for value in (101, 100, 100, 102, 99, 101)],
    }

    statistics = {item.name: item for item in _statistics(contract, results)}

    assert statistics["kv-64"].promotable is True
    assert statistics["kv-64"].run_count == 6
    assert statistics["kv-64"].sample_count == 300
    assert statistics["kv-64"].improvement_confidence_interval[0] > 0.01
    assert statistics["kv-256"].promotable is False


def test_rpa_search_requires_fresh_confirmation_for_promotion() -> None:
    contract = _contract()
    orders = tuple(
        (("incumbent", "kv-64") if index % 2 == 0 else ("kv-64", "incumbent"))
        for index in range(contract.confirmation_rounds)
    )
    confirmation = _confirmation_statistics(
        contract,
        "kv-64",
        orders,
        [_result(value) for value in (100, 102, 98, 101, 99, 100)],
        [_result(value) for value in (80, 82, 78, 81, 79, 80)],
    )

    assert confirmation.confirmed is True
    assert confirmation.run_count == contract.confirmation_rounds
    assert confirmation.execution_orders == orders


def test_checked_in_rpa_search_contract_is_valid() -> None:
    path = Path("contracts/inkling-fused-rpa-block-search.json")
    contract = RpaSearchContract.model_validate_json(path.read_text())

    assert contract.baseline == "incumbent"
    assert len(contract.candidates) == 5
    assert contract.search_id == RpaSearchContract.model_validate_json(
        contract.model_dump_json(exclude_computed_fields=True)
    ).search_id
    orders = _execution_orders(contract)
    assert len(orders) == 10
    assert all(
        [order[position] for order in orders].count(candidate.name) == 2
        for position in range(5)
        for candidate in contract.candidates
    )


def test_rpa_search_replay_rejects_a_self_declared_contract(
    tmp_path: Path,
) -> None:
    saved = _contract()
    expected = saved.model_copy(update={"minimum_practical_improvement": 0.02})
    (tmp_path / "contract.json").write_text(
        saved.model_dump_json(exclude_computed_fields=True)
    )

    with pytest.raises(ValueError, match="RPA_SEARCH_CONTRACT_MISMATCH"):
        validate_rpa_search_result(tmp_path, expected)


def test_incomplete_search_run_is_preserved_before_retry(tmp_path: Path) -> None:
    run_root = tmp_path / "round-00" / "00-incumbent"
    run_root.mkdir(parents=True)
    (run_root / "failure.log").write_text("compile failed\n")

    archive = _archive_incomplete_run(tmp_path, run_root, "compile failed")

    assert not run_root.exists()
    assert archive == tmp_path / "incomplete/round-00/00-incumbent-attempt-00"
    assert (archive / "failure.log").read_text() == "compile failed\n"
    assert "compile failed" in (archive / "failure.json").read_text()
    evidence = _incomplete_attempts(tmp_path)
    assert len(evidence) == 1
    assert evidence[0].path == "incomplete/round-00/00-incumbent-attempt-00"
    original_manifest = evidence[0].manifest_sha256
    (archive / "failure.log").write_text("changed\n")
    assert _incomplete_attempts(tmp_path)[0].manifest_sha256 != original_manifest


def test_rpa_search_requires_the_declared_profiler_configuration(
    tmp_path: Path,
) -> None:
    contract = _contract()
    path = tmp_path / "profiler_config.json"
    path.write_text(contract.profiler.model_dump_json())
    expected_sha256 = _validated_profiler_config_sha256(path, contract)
    changed = contract.profiler.model_copy(update={"host_tracer_level": 99})
    path.write_text(changed.model_dump_json())

    with pytest.raises(ValueError, match="RPA_SEARCH_PROFILER_CONFIG_MISMATCH"):
        _validated_profiler_config_sha256(path, contract)
    assert len(expected_sha256) == 64


def test_device_timing_accepts_only_tpu_xla_custom_call_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xplane = tmp_path / "profile/capture.xplane.pb"
    xplane.parent.mkdir()
    xplane.write_bytes(b"xplane")
    candidate = _candidate("incumbent", (8, 128, 8, 128))
    event_name = (
        '%RPAd-p_16-bq_8_8-bkv_128_128.1 custom-call(), '
        'custom_call_target="tpu_custom_call"'
    )

    def event(duration: float) -> SimpleNamespace:
        return SimpleNamespace(name=event_name, duration_ns=duration)

    class FakeProfile:
        planes = (
            SimpleNamespace(
                name="/host:CPU",
                lines=(SimpleNamespace(name="XLA Ops", events=(event(1.0),)),),
            ),
            SimpleNamespace(
                name="/device:TPU:0",
                lines=(
                    SimpleNamespace(name="Host Threads", events=(event(2.0),)),
                    SimpleNamespace(
                        name="XLA Ops", events=tuple(event(10.0) for _ in range(50))
                    ),
                ),
            ),
        )

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        rpa_search.profile_data.ProfileData,
        "from_file",
        lambda _: FakeProfile(),
    )

    timing = _device_timing(tmp_path, candidate, 50)

    assert timing.durations_ns == (10.0,) * 50
    assert timing.median_ns == 10.0
