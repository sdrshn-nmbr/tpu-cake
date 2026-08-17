from decimal import Decimal

import pytest

from tpu_cake.surfaces import (
    AttentionScenario,
    AttentionWorkloadSurface,
    OutputEquivalencePolicy,
    ScenarioObservation,
    SeqaxForwardScenario,
    SeqaxForwardWorkloadSurface,
    SurfaceCandidateObservation,
    compare_surface_candidates,
)


def _three_scenario_surface() -> SeqaxForwardWorkloadSurface:
    common = {
        "batch": 2,
        "sequence": 4,
        "model": 8,
        "vocabulary": 16,
        "feed_forward": 16,
        "query_groups": 2,
        "key_value_heads": 4,
        "head": 4,
        "layers": 2,
        "data_mesh": 2,
        "tensor_mesh": 4,
        "rope_max_timescale": 256,
        "weight": Decimal(1),
    }
    return SeqaxForwardWorkloadSurface(
        name="independent-scenario-bootstrap",
        scenarios=tuple(
            SeqaxForwardScenario(name=name, **common)
            for name in ("tiny", "wider", "deeper")
        ),
        minimum_practical_improvement=Decimal("0.03"),
        bootstrap_samples=10_000,
    )


def _three_scenario_candidate(
    name: str,
    values: dict[str, tuple[int, ...]],
    *,
    starts_first: bool,
) -> SurfaceCandidateObservation:
    return SurfaceCandidateObservation(
        candidate=name,
        scenarios=tuple(
            ScenarioObservation(
                scenario=scenario,
                round_medians_ns=values[scenario],
                ran_first=tuple(
                    starts_first if index % 2 == 0 else not starts_first
                    for index in range(len(values[scenario]))
                ),
                input_sha256="1" * 64,
                output_sha256="2" * 64,
                runtime_sha256="3" * 64,
                profiled=False,
                passed=True,
            )
            for scenario in ("tiny", "wider", "deeper")
        ),
    )


def _surface() -> AttentionWorkloadSurface:
    return AttentionWorkloadSurface(
        name="inkling-rpa",
        scenarios=(
            AttentionScenario(
                name="decode-ragged",
                stage="steady_decode",
                batch_size=4,
                query_tokens_per_request=1,
                context_lengths=(17, 33, 65, 127),
                page_size=16,
                dtype="bf16",
                sharding=("d", "t"),
                weight=Decimal("0.7"),
            ),
            AttentionScenario(
                name="prefill-tail",
                stage="prefill",
                batch_size=2,
                query_tokens_per_request=33,
                context_lengths=(33, 61),
                page_size=16,
                dtype="bf16",
                sharding=("d", "t"),
                weight=Decimal("0.3"),
            ),
        ),
        minimum_practical_improvement=Decimal("0.03"),
        maximum_scenario_regression=Decimal("0.01"),
        bootstrap_samples=1_000,
    )


def _candidate(name: str, decode: tuple[int, ...], prefill: tuple[int, ...]):
    return SurfaceCandidateObservation(
        candidate=name,
        scenarios=(
            ScenarioObservation(
                scenario="decode-ragged",
                round_medians_ns=decode,
                ran_first=(True, False, True, False, True),
                input_sha256="1" * 64,
                output_sha256="a" * 64,
                runtime_sha256="2" * 64,
                profiled=False,
                passed=True,
            ),
            ScenarioObservation(
                scenario="prefill-tail",
                round_medians_ns=prefill,
                ran_first=(True, False, True, False, True),
                input_sha256="3" * 64,
                output_sha256="b" * 64,
                runtime_sha256="2" * 64,
                profiled=False,
                passed=True,
            ),
        ),
    )


def test_attention_surface_tracks_raggedness_and_page_occupancy() -> None:
    surface = _surface()
    scenario = surface.scenarios[0]

    assert scenario.allocated_pages == 18
    assert scenario.page_occupancy == Decimal(242) / Decimal(288)
    assert len(surface.surface_id) == 64


def test_seqax_surface_exposes_complete_forward_parameters() -> None:
    scenario = SeqaxForwardScenario(
        name="small",
        batch=2,
        sequence=4,
        model=8,
        vocabulary=16,
        feed_forward=16,
        query_groups=2,
        key_value_heads=4,
        head=4,
        layers=2,
        data_mesh=2,
        tensor_mesh=4,
        rope_max_timescale=256,
        weight=Decimal(1),
    )
    surface = SeqaxForwardWorkloadSurface(
        name="seqax-forward",
        scenarios=(scenario, scenario.model_copy(update={"name": "second"})),
        minimum_practical_improvement=Decimal("0.03"),
        bootstrap_samples=1_000,
    )

    assert scenario.parameters()["rope_max_timescale"] == 256
    assert "weight" not in scenario.parameters()
    assert len(surface.surface_id) == 64
    assert surface.surface_id != surface.model_copy(
        update={"oracle_quantization_decimals": 4}
    ).surface_id


def test_seqax_surface_rejects_dimensions_that_do_not_fit_the_mesh() -> None:
    with pytest.raises(ValueError, match="vocabulary and feed-forward"):
        SeqaxForwardScenario(
            name="bad",
            batch=2,
            sequence=4,
            model=8,
            vocabulary=15,
            feed_forward=16,
            query_groups=2,
            key_value_heads=4,
            head=4,
            layers=2,
            data_mesh=2,
            tensor_mesh=4,
            rope_max_timescale=256,
            weight=Decimal(1),
        )


def test_surface_promotion_uses_matched_rounds_and_rejects_regressions() -> None:
    baseline = _candidate("baseline", (100, 102, 98, 101, 99), (200, 204, 196, 202, 198))
    faster = _candidate("faster", (90, 91, 89, 90, 88), (180, 183, 177, 181, 179)).model_copy(
        update={
            "scenarios": tuple(
                value.model_copy(update={"ran_first": (False, True, False, True, False)})
                for value in _candidate(
                    "faster",
                    (90, 91, 89, 90, 88),
                    (180, 183, 177, 181, 179),
                ).scenarios
            )
        }
    )
    regressed = _candidate("regressed", (80, 81, 79, 80, 78), (210, 214, 206, 212, 208)).model_copy(
        update={
            "scenarios": tuple(
                value.model_copy(update={"ran_first": (False, True, False, True, False)})
                for value in _candidate(
                    "regressed",
                    (80, 81, 79, 80, 78),
                    (210, 214, 206, 212, 208),
                ).scenarios
            )
        }
    )

    assert compare_surface_candidates(_surface(), baseline, faster).promotable
    rejected = compare_surface_candidates(_surface(), baseline, regressed)
    assert not rejected.promotable
    assert rejected.scenario_improvements["prefill-tail"] < 0


def test_surface_comparison_requires_each_candidate_to_pass() -> None:
    baseline = _candidate("baseline", (100,) * 5, (200,) * 5)
    raw_candidate = _candidate("candidate", (90,) * 5, (180,) * 5)
    candidate = raw_candidate.model_copy(
        update={
            "scenarios": tuple(
                value.model_copy(update={"ran_first": (False, True, False, True, False)})
                for value in raw_candidate.scenarios
            )
        }
    )
    corrupted = candidate.model_copy(
        update={
            "scenarios": (
                candidate.scenarios[0].model_copy(update={"passed": False}),
                candidate.scenarios[1],
            )
        }
    )

    with pytest.raises(ValueError, match="must pass their numerical contracts"):
        compare_surface_candidates(_surface(), baseline, corrupted)


def test_surface_comparison_allows_numerically_valid_cross_mode_rounding() -> None:
    baseline = _candidate("baseline", (100,) * 5, (200,) * 5)
    raw_candidate = _candidate("candidate", (90,) * 5, (180,) * 5)
    candidate = raw_candidate.model_copy(
        update={
            "scenarios": tuple(
                value.model_copy(
                    update={
                        "ran_first": (False, True, False, True, False),
                        "output_sha256": "c" * 64,
                    }
                )
                for value in raw_candidate.scenarios
            )
        }
    )

    tolerant_surface = _surface().model_copy(
        update={
            "output_equivalence": (
                OutputEquivalencePolicy.INDEPENDENT_ORACLE_AND_CROSS_MODE_TOLERANCE
            )
        }
    )

    assert compare_surface_candidates(tolerant_surface, baseline, candidate).promotable


def test_surface_comparison_rejects_unmatched_benchmark_context() -> None:
    baseline = _candidate("baseline", (100,) * 5, (200,) * 5)
    raw_candidate = _candidate("candidate", (90,) * 5, (180,) * 5)
    candidate = raw_candidate.model_copy(
        update={
            "scenarios": tuple(
                value.model_copy(update={"ran_first": (False, True, False, True, False)})
                for value in raw_candidate.scenarios
            )
        }
    )

    wrong_input = candidate.model_copy(
        update={
            "scenarios": (
                candidate.scenarios[0].model_copy(update={"input_sha256": "4" * 64}),
                candidate.scenarios[1],
            )
        }
    )
    with pytest.raises(ValueError, match="matched inputs and runtimes"):
        compare_surface_candidates(_surface(), baseline, wrong_input)

    profiled = candidate.model_copy(
        update={
            "scenarios": (
                candidate.scenarios[0].model_copy(update={"profiled": True}),
                candidate.scenarios[1],
            )
        }
    )
    with pytest.raises(ValueError, match="unprofiled timing"):
        compare_surface_candidates(_surface(), baseline, profiled)

    biased_order = candidate.model_copy(
        update={
            "scenarios": (
                candidate.scenarios[0].model_copy(update={"ran_first": (True,) * 5}),
                candidate.scenarios[1],
            )
        }
    )
    with pytest.raises(ValueError, match="alternating matched order"):
        compare_surface_candidates(_surface(), baseline, biased_order)


def test_surface_bootstraps_sequential_scenarios_independently() -> None:
    surface = _three_scenario_surface()
    baseline = _three_scenario_candidate(
        "baseline",
        {name: (100,) * 10 for name in ("tiny", "wider", "deeper")},
        starts_first=True,
    )
    candidate = _three_scenario_candidate(
        "candidate",
        {
            "tiny": (100, 92) * 5,
            "wider": (92, 100) * 5,
            "deeper": (96,) * 10,
        },
        starts_first=False,
    )

    comparison = compare_surface_candidates(surface, baseline, candidate)

    assert not comparison.promotable
    assert comparison.confidence_interval[0] < surface.minimum_practical_improvement
