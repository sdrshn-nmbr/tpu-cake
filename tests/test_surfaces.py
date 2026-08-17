from decimal import Decimal

import pytest

from tpu_cake.surfaces import (
    AttentionScenario,
    AttentionWorkloadSurface,
    ScenarioObservation,
    SurfaceCandidateObservation,
    compare_surface_candidates,
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


def test_surface_comparison_requires_matched_correct_outputs() -> None:
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
                candidate.scenarios[0].model_copy(update={"output_sha256": "c" * 64}),
                candidate.scenarios[1],
            )
        }
    )

    with pytest.raises(ValueError, match="matched correct outputs"):
        compare_surface_candidates(_surface(), baseline, corrupted)


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
