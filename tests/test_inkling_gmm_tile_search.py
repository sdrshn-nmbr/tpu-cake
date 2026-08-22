import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tpu_cake.contracts import SourceFileContract
from tpu_cake.identity import model_identity_sha256
from tpu_cake.inkling_gmm_route_corpus import (
    InklingGmmRouteCorpusReport,
    RouteGroupSizes,
    _json_sha256,
)
from tpu_cake.inkling_gmm_tile_search import (
    GMM_CORRECTNESS_SEEDS,
    GMM_IMPLEMENTATION_SOURCE_PATHS,
    GMM_OPERAND_SEED,
    GmmArmName,
    GmmConfirmationObservation,
    GmmCorrectnessProfileReason,
    GmmOperation,
    GmmPolicyPair,
    GmmScreenObservation,
    GmmSearchFamily,
    InklingGmmTileSearchContract,
    confirmation_orders,
    confirmation_statistics,
    default_gmm_numerical_contract,
    default_gmm_tile_search_contract,
    local_active_span,
    screening_orders,
    screening_statistics,
    select_correctness_profiles,
    select_cpu_oracle_rows,
    validate_route_corpus_binding,
)


def _source_manifest() -> tuple[SourceFileContract, ...]:
    return tuple(
        SourceFileContract(path=path, sha256=f"{index + 1:064x}")
        for index, path in enumerate(GMM_IMPLEMENTATION_SOURCE_PATHS)
    )


def _route_report() -> tuple[InklingGmmRouteCorpusReport, bytes]:
    groups = []
    counts = (33,) + (1,) * 255
    for completion_step in range(2, 66):
        for layer_index in range(2, 42):
            groups.append(
                RouteGroupSizes(
                    completion_step=completion_step,
                    layer_index=layer_index,
                    group_sizes=counts,
                )
            )
    provisional = InklingGmmRouteCorpusReport(
        report_id="0" * 64,
        contract_id="1" * 64,
        capture_id="2" * 64,
        capture_sha256="3" * 64,
        producer_source_sha256="4" * 64,
        verifier_source_sha256="5" * 64,
        server_launch_receipt_id="6" * 64,
        server_launch_receipt_sha256="7" * 64,
        request_sha256="8" * 64,
        model_weight_manifest_sha256="9" * 64,
        concurrency=48,
        selected_completion_steps=tuple(range(2, 66)),
        first_moe_layer=2,
        num_layers=42,
        num_experts_per_token=6,
        num_routed_experts=256,
        request_state_slots=tuple(range(48)),
        recurrent_state_slots=tuple(range(48, 96)),
        group_sizes=tuple(groups),
        corpus_sha256=_json_sha256([group.model_dump(mode="json") for group in groups]),
    )
    report = provisional.model_copy(
        update={"report_id": model_identity_sha256(provisional, exclude={"report_id"})}
    )
    raw = (json.dumps(report.model_dump(mode="json"), sort_keys=True) + "\n").encode()
    return report, raw


def _contract() -> tuple[InklingGmmTileSearchContract, InklingGmmRouteCorpusReport]:
    report, raw = _route_report()
    contract = default_gmm_tile_search_contract(
        accepted_route_report_id=report.report_id,
        accepted_route_report_sha256=hashlib.sha256(raw).hexdigest(),
        accepted_route_corpus_sha256=report.corpus_sha256,
        accepted_route_report=report,
        tpu_cake_git_commit="d" * 40,
        tpu_cake_uv_lock_sha256="e" * 64,
        runner_source_sha256="f" * 64,
        verifier_source_sha256="0" * 64,
        confirmation_verifier_source_sha256="1" * 64,
        inkling_git_commit="a" * 40,
        inkling_uv_lock_sha256="b" * 64,
        implementation_source_manifest=_source_manifest(),
    )
    return contract, report


def test_default_contract_fixes_the_production_abi_and_protocol() -> None:
    contract, _ = _contract()

    assert tuple(source.path for source in contract.implementation_source_manifest) == (
        GMM_IMPLEMENTATION_SOURCE_PATHS
    )
    assert (contract.production_abi.m, contract.production_abi.global_group_count) == (288, 256)
    assert (
        contract.production_abi.device_count,
        contract.production_abi.local_experts_per_device,
    ) == (8, 32)
    assert contract.production_abi.group_offset_rule == "device_index*32"
    assert contract.production_abi.expert_location == "trivial-identity-no-redundant-experts"
    assert contract.production_abi.lhs_distribution == "same-global-expert-sorted-lhs-per-device"
    assert contract.target_runtime.project_id == "astral-medley-465922-b2"
    assert contract.target_runtime.zone == "us-central1-c"
    assert contract.target_runtime.instance_name == "tpu-cake-v7x-rsag-wx7r"
    assert contract.target_runtime.device_type == "TPU v7x"
    assert contract.target_runtime.accelerator_type == "tpu7x-8"
    assert contract.target_runtime.topology == "2x2x1"
    assert contract.target_runtime.host_count == 1
    assert (
        contract.target_runtime.server_tp_size,
        contract.target_runtime.server_ep_size,
        contract.target_runtime.gmm_expert_axis_size,
        contract.target_runtime.gmm_tensor_axis_size,
    ) == (8, 8, 8, 1)

    gate, up, down = contract.production_abi.kernels
    assert (gate.operation, up.operation, down.operation) == (
        GmmOperation.GATE,
        GmmOperation.UP,
        GmmOperation.DOWN,
    )
    assert (gate.lhs_dtype, gate.rhs_dtype, gate.accumulator_dtype, gate.output_dtype) == (
        "bf16",
        "bf16",
        "fp32",
        "fp32",
    )
    assert (gate.k, gate.n, gate.zero_initialize) == (4096, 2048, False)
    assert (up.k, up.n, up.zero_initialize) == (4096, 2048, False)
    assert (down.lhs_dtype, down.rhs_dtype, down.accumulator_dtype, down.output_dtype) == (
        "fp32",
        "bf16",
        "fp32",
        "fp32",
    )
    assert (down.k, down.n, down.zero_initialize) == (2048, 4096, True)
    assert all(
        not kernel.quantized and not kernel.has_scale and not kernel.has_bias
        for kernel in contract.production_abi.kernels
    )

    assert contract.corpus.completion_steps == tuple(range(2, 66))
    assert contract.corpus.layer_indices == tuple(range(2, 42))
    assert contract.corpus.group_count == 64 * 40
    assert contract.corpus.timing_unit == "one-ordered-64-step-by-40-layer-corpus-block"
    assert contract.corpus.groups_are_independent_samples is False
    assert tuple(arm.name for arm in contract.arms) == tuple(GmmArmName)
    assert tuple((arm.tile_m, arm.tile_k, arm.tile_n) for arm in contract.arms) == (
        (128, "K", "N"),
        (64, "K", "N"),
        (32, "K", "N"),
        (128, "K", "N/2"),
        (64, "K", "N/2"),
    )

    assert contract.search.can_promote is False
    assert contract.search.claim_scope == "best-validated-result-from-two-one-factor-screens"
    assert contract.search.families == (GmmSearchFamily.GATE_UP, GmmSearchFamily.DOWN)
    assert contract.search.gate_up_screen == (
        "candidate-gate-up-silu-multiply-incumbent-down-full-chain"
    )
    assert contract.search.down_screen == (
        "incumbent-gate-up-silu-multiply-candidate-down-full-chain"
    )
    assert contract.search.warmup_full_corpus_blocks_per_arm == 1
    assert contract.search.screening_rounds_per_family == 10
    assert contract.search.order == "balanced-forward-reverse-latin-square"
    assert contract.search.score == "median-full-corpus-duration"
    assert contract.search.finalist_rule == "lowest-family-median"
    assert contract.search.tie_rule == "incumbent-then-declaration-order"
    assert contract.search.layer_weight_banks == 40
    assert contract.search.layer_weight_banks_are_distinct is True
    assert contract.search.layer_input_banks == 40
    assert contract.search.layer_input_banks_are_distinct is True
    assert contract.search.layer_inputs_reused_across_completion_steps is True
    assert contract.search.operand_seed == GMM_OPERAND_SEED
    assert contract.search.operand_generation == "jax-stateless-uniform-v1"
    assert contract.search.minimum_free_device_bytes == 80 * 1024**3
    assert contract.search.executables_resident is True
    assert contract.search.operands_resident is True
    assert contract.search.external_operands_shared_across_arms is True
    assert contract.search.candidate_intermediates_shared_across_arms is False
    assert contract.search.free_memory_checked_before_allocation is True
    assert contract.search.residency_checked_before_timing is True
    assert contract.search.compilation_excluded_from_timing is True
    assert contract.search.dispatch == (
        "64-host-dispatched-completion-executions-with-40-unrolled-layer-chains"
    )
    assert contract.search.synchronization == "block-until-ready-on-chain-liveness-output"
    assert contract.search.output_liveness == "one-down-output-scalar-per-layer"
    assert contract.search.compiler_preflight == (
        "reachable-exact-gmm-v2-scope-label-per-operation"
    )
    assert contract.confirmation.paired_rounds == 32
    assert contract.confirmation.samples_per_arm_per_round == 5
    assert contract.confirmation.bootstrap_samples == 100_000
    assert contract.confirmation.bootstrap_seed_rule == "semantic-seed(search-id,finalist)"
    assert (
        contract.confirmation.within_round_reduction
        == "median-of-five-synchronized-full-corpus-blocks"
    )
    assert contract.confirmation.confidence_level == 0.99
    assert contract.confirmation.minimum_practical_improvement == 0.03
    assert contract.confirmation.lower_bound_must_exceed_threshold is True
    assert contract.confirmation.allow_early_stopping is False
    assert contract.confirmation.allow_retry is False
    assert contract.confirmation.executables_resident is True
    assert contract.confirmation.operands_resident is True
    assert contract.confirmation.candidate == "combined-gate-up-and-down-family-finalists"
    assert contract.confirmation.combined_failure_rule == "no-promotion-no-fallback"
    assert contract.confirmation.screening_samples_reused is False
    assert contract.correctness.seeds == GMM_CORRECTNESS_SEEDS
    assert (
        contract.correctness.numerical_contract_id == default_gmm_numerical_contract().contract_id
    )
    assert contract.correctness.profile_count == 5
    assert tuple(profile.reason for profile in contract.correctness.profiles) == tuple(
        GmmCorrectnessProfileReason
    )
    assert contract.correctness.require_all_expert_shards_covered is True
    assert contract.correctness.compare_complete_active_spans is True
    assert contract.correctness.cpu_oracle_rows_per_expert_shard == 1
    assert contract.correctness.cpu_oracle_down_columns == (0, 2047, 2048, 4095)
    assert contract.correctness.cpu_oracle_seeded_interior_columns_per_row == 4
    assert tuple(row.device_index for row in contract.correctness.cpu_oracle_rows) == tuple(
        range(8)
    )
    assert contract.correctness.tolerances_frozen_before_timing is True


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("production_abi", "m"), 287),
        (("production_abi", "local_experts_per_device"), 31),
        (("production_abi", "kernels", 0, "output_dtype"), "bf16"),
        (("production_abi", "kernels", 2, "zero_initialize"), False),
        (("production_abi", "expert_location"), "dynamic-eplb"),
        (("target_runtime", "device_type"), "TPU v6e"),
        (("target_runtime", "instance_name"), "some-other-tpu"),
        (("target_runtime", "gmm_tensor_axis_size"), 8),
        (("arms", 1, "tile_m"), 128),
        (("corpus", "groups_are_independent_samples"), True),
        (("search", "can_promote"), True),
        (("search", "layer_weight_banks"), 1),
        (("search", "screening_rounds_per_family"), 5),
        (("search", "families"), (GmmSearchFamily.DOWN, GmmSearchFamily.GATE_UP)),
        (("search", "operand_seed"), GMM_OPERAND_SEED + 1),
        (("search", "layer_input_banks"), 2_560),
        (("search", "external_operands_shared_across_arms"), False),
        (("search", "candidate_intermediates_shared_across_arms"), True),
        (("search", "compilation_excluded_from_timing"), False),
        (("confirmation", "paired_rounds"), 30),
        (("confirmation", "allow_retry"), True),
        (("confirmation", "combined_failure_rule"), "confirm-hybrids"),
        (("correctness", "seeds"), GMM_CORRECTNESS_SEEDS[:-1]),
        (("correctness", "absolute_tolerance"), 1.0),
        (("correctness", "cpu_oracle_down_columns"), (0, 4095)),
    ),
)
def test_contract_rejects_relaxed_or_changed_claims(
    path: tuple[object, ...], value: object
) -> None:
    contract, _ = _contract()
    payload = contract.model_dump(mode="json", exclude={"search_id"})
    target = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValidationError):
        InklingGmmTileSearchContract.model_validate(payload)


def test_contract_rejects_an_incomplete_or_reordered_source_manifest() -> None:
    contract, _ = _contract()
    payload = contract.model_dump(mode="json", exclude={"search_id"})
    payload["implementation_source_manifest"] = list(
        reversed(payload["implementation_source_manifest"][:-1])
    )

    with pytest.raises(ValidationError, match="source manifest"):
        InklingGmmTileSearchContract.model_validate(payload)


def test_route_binding_accepts_only_the_exact_complete_corpus() -> None:
    contract, report = _contract()
    raw = (json.dumps(report.model_dump(mode="json"), sort_keys=True) + "\n").encode()

    validate_route_corpus_binding(contract, report, report_bytes=raw)

    with pytest.raises(ValueError, match="report content hash"):
        validate_route_corpus_binding(contract, report, report_bytes=raw + b" ")
    unrelated = b"{}\n"
    unrelated_contract = contract.model_copy(
        update={
            "route_corpus": contract.route_corpus.model_copy(
                update={"report_sha256": hashlib.sha256(unrelated).hexdigest()}
            )
        }
    )
    with pytest.raises(ValueError, match="bytes do not encode"):
        validate_route_corpus_binding(unrelated_contract, report, report_bytes=unrelated)
    with pytest.raises(ValueError, match="report identity"):
        validate_route_corpus_binding(
            contract.model_copy(
                update={
                    "route_corpus": contract.route_corpus.model_copy(update={"report_id": "f" * 64})
                }
            ),
            report,
            report_bytes=raw,
        )
    provisional = report.model_copy(update={"report_id": "0" * 64})
    provisional_raw = (
        json.dumps(provisional.model_dump(mode="json"), sort_keys=True) + "\n"
    ).encode()
    provisional_contract = contract.model_copy(
        update={
            "route_corpus": contract.route_corpus.model_copy(
                update={
                    "report_id": "0" * 64,
                    "report_sha256": hashlib.sha256(provisional_raw).hexdigest(),
                }
            )
        }
    )
    with pytest.raises(ValueError, match="final identity"):
        validate_route_corpus_binding(
            provisional_contract,
            provisional,
            report_bytes=provisional_raw,
        )

    wrong_workload_provisional = report.model_copy(
        update={
            "report_id": "0" * 64,
            "concurrency": 96,
            "num_experts_per_token": 3,
            "request_state_slots": tuple(range(96)),
            "recurrent_state_slots": tuple(range(96, 192)),
        }
    )
    wrong_workload = wrong_workload_provisional.model_copy(
        update={
            "report_id": model_identity_sha256(
                wrong_workload_provisional,
                exclude={"report_id"},
            )
        }
    )
    wrong_workload_raw = (
        json.dumps(wrong_workload.model_dump(mode="json"), sort_keys=True) + "\n"
    ).encode()
    wrong_workload_contract = contract.model_copy(
        update={
            "route_corpus": contract.route_corpus.model_copy(
                update={
                    "report_id": wrong_workload.report_id,
                    "report_sha256": hashlib.sha256(wrong_workload_raw).hexdigest(),
                }
            )
        }
    )
    with pytest.raises(ValueError, match="production workload"):
        validate_route_corpus_binding(
            wrong_workload_contract,
            wrong_workload,
            report_bytes=wrong_workload_raw,
        )


def test_local_active_span_uses_global_contiguous_expert_order() -> None:
    group_sizes = tuple(range(256))

    assert local_active_span(group_sizes, device_index=0) == (0, sum(range(32)))
    assert local_active_span(group_sizes, device_index=3) == (
        sum(range(96)),
        sum(range(128)),
    )
    with pytest.raises(ValueError, match="256"):
        local_active_span(group_sizes[:-1], device_index=0)
    with pytest.raises(ValueError, match="device index"):
        local_active_span(group_sizes, device_index=8)


def test_screening_orders_are_exact_balanced_forward_reverse_latin_squares() -> None:
    contract, _ = _contract()

    for family in GmmSearchFamily:
        orders = screening_orders(contract, family)
        assert len(orders) == 10
        assert orders[0] == tuple(GmmArmName)
        assert orders[5] == tuple(reversed(tuple(GmmArmName)))
        for arm in GmmArmName:
            positions = [order.index(arm) for order in orders]
            assert sorted(positions) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_correctness_profiles_and_cpu_rows_are_reconstructed_from_the_route_corpus() -> None:
    report = InklingGmmRouteCorpusReport.model_validate_json(
        Path("evidence/inkling/gmm-route-corpus-v1.json").read_text()
    )

    profiles = select_correctness_profiles(report)
    rows = select_cpu_oracle_rows(report, profiles)

    assert tuple((profile.completion_step, profile.layer_index) for profile in profiles) == (
        (24, 41),
        (20, 3),
        (58, 14),
        (54, 25),
        (43, 33),
    )
    assert tuple(row.device_index for row in rows) == tuple(range(8))
    assert tuple(row.profile_index for row in rows) == (0,) * 8
    assert tuple(row.row_index for row in rows) == (0, 40, 85, 88, 99, 163, 211, 283)
    assert all(len(row.down_columns) == 8 for row in rows)
    assert all({0, 2047, 2048, 4095}.issubset(row.down_columns) for row in rows)


def test_screening_statistics_select_the_lowest_median_with_declared_ties() -> None:
    contract, _ = _contract()
    durations = {
        GmmArmName.INCUMBENT: 100,
        GmmArmName.SPARSE_M64: 90,
        GmmArmName.SPARSE_M32: 95,
        GmmArmName.SPLIT_N: 105,
        GmmArmName.SPARSE_M64_SPLIT_N: 90,
    }
    observations = tuple(
        GmmScreenObservation(
            family=GmmSearchFamily.GATE_UP,
            round_index=round_index,
            position=position,
            arm=arm,
            duration_ns=durations[arm],
        )
        for round_index, order in enumerate(screening_orders(contract, GmmSearchFamily.GATE_UP))
        for position, arm in enumerate(order)
    )

    result = screening_statistics(contract, GmmSearchFamily.GATE_UP, observations)

    assert result.finalist is GmmArmName.SPARSE_M64
    forged = list(observations)
    forged[0] = forged[0].model_copy(update={"arm": GmmArmName.SPLIT_N})
    with pytest.raises(ValueError, match="execution order"):
        screening_statistics(contract, GmmSearchFamily.GATE_UP, tuple(forged))


def test_confirmation_statistics_apply_the_strict_paired_bootstrap_gate() -> None:
    contract, _ = _contract()
    candidate = GmmPolicyPair(
        gate_up=GmmArmName.SPARSE_M64,
        down=GmmArmName.SPLIT_N,
    )
    observations = []
    for round_index, order in enumerate(confirmation_orders(contract, candidate)):
        for position, policy in enumerate(order):
            sample = 950 if policy == candidate else 1_000
            observations.append(
                GmmConfirmationObservation(
                    round_index=round_index,
                    position=position,
                    policy=policy,
                    samples_ns=(sample,) * 5,
                )
            )

    result = confirmation_statistics(contract, candidate, tuple(observations))

    assert result.median_improvement == pytest.approx(0.05)
    assert result.confidence_interval == pytest.approx((0.05, 0.05))
    assert result.confirmed is True
    forged = list(observations)
    forged[0], forged[1] = forged[1], forged[0]
    with pytest.raises(ValueError, match="(position|execution order)"):
        confirmation_statistics(contract, candidate, tuple(forged))


def test_committed_confirmation_retains_the_incumbent_with_immutable_evidence() -> None:
    contract = InklingGmmTileSearchContract.model_validate_json(
        Path("contracts/inkling-gmm-tile-search-v1.json").read_bytes()
    )
    confirmation = json.loads(
        Path("evidence/inkling/gmm-tile-search-confirmation-v1.json").read_text()
    )
    receipt = json.loads(
        Path("evidence/inkling/gmm-tile-search-confirmation-receipt-v1.json").read_text()
    )

    assert contract.search_id == "d2b8a27f500145cc86f2bc803ab2f5622cf980928c8945afef5e07deeccbdbee"
    assert contract.tpu_cake_git_commit == "82e51d4f2d24cdb24a4b825f9420e0bfafdd6fb6"
    assert contract.confirmation.warmup_full_corpus_blocks_per_arm == 1
    assert confirmation["verification_id"] == (
        "ec9ee340f7ac5342abb54e4e7fd87db8b027a9d0fc3899c17c4f47a32821737e"
    )
    statistics = confirmation["confirmation_statistics"]
    assert statistics["candidate"]["gate_up"] == "sparse-m64-split-n"
    assert statistics["candidate"]["down"] == "sparse-m64-split-n"
    assert statistics["median_improvement"] == pytest.approx(0.00909941263516112)
    assert statistics["confidence_interval"] == pytest.approx(
        (0.009090670385233257, 0.009109655255769122)
    )
    assert statistics["minimum_practical_improvement"] == 0.03
    assert statistics["confirmed"] is False
    assert confirmation["claims"]["promotion_authorized"] is False
    assert receipt["search_id"] == contract.search_id
    assert receipt["confirmation_verification_id"] == confirmation["verification_id"]
    assert receipt["selection"] == "baseline-retained-confirmation-below-practical-threshold"
    assert receipt["promotion_authorized"] is False
    assert receipt["archive_sha256"] == (
        "c711ed418c12f1467832abdf6cdb3bf516f889c327af3055ea71dc1f3e3a6d56"
    )
    assert receipt["receipt_id"] == _json_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_id"}
    )
