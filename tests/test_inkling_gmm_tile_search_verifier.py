import hashlib
import json
from pathlib import Path

import pytest

from tpu_cake.contracts import SourceFileContract
from tpu_cake.identity import model_identity_sha256
from tpu_cake.inkling_gmm_route_corpus import (
    InklingGmmRouteCorpusReport,
    RouteGroupSizes,
)
from tpu_cake.inkling_gmm_tile_search import (
    GMM_IMPLEMENTATION_SOURCE_PATHS,
    GmmArmName,
    GmmSearchFamily,
    default_gmm_tile_search_contract,
)
from tpu_cake.inkling_gmm_tile_search_correctness import (
    CpuOracleMeasurement,
    GmmCorrectnessGateReport,
    OperandSentinelMeasurement,
    OutputMeasurement,
    PolicyCorrectnessMeasurement,
    ProfileCorrectnessMeasurement,
    StageOutputMeasurement,
)
from tpu_cake.inkling_gmm_tile_search_verifier import (
    _estimated_operand_bytes_per_device,
    _expected_custom_call_counts,
    _expected_policies,
    _expected_scopes,
    _screening_orders,
    verify_screening,
    write_verified_report,
)


def _json_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _route_report() -> tuple[InklingGmmRouteCorpusReport, bytes]:
    groups = tuple(
        RouteGroupSizes(
            completion_step=completion_step,
            layer_index=layer_index,
            group_sizes=(33,) + (1,) * 255,
        )
        for completion_step in range(2, 66)
        for layer_index in range(2, 42)
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
        group_sizes=groups,
        corpus_sha256=_json_sha256([group.model_dump(mode="json") for group in groups]),
    )
    report = provisional.model_copy(
        update={"report_id": model_identity_sha256(provisional, exclude={"report_id"})}
    )
    raw = (json.dumps(report.model_dump(mode="json"), sort_keys=True) + "\n").encode()
    return report, raw


def _contract(report: InklingGmmRouteCorpusReport, report_raw: bytes):
    verifier_path = Path(__file__).parents[1] / ("src/tpu_cake/inkling_gmm_tile_search_verifier.py")
    return default_gmm_tile_search_contract(
        accepted_route_report_id=report.report_id,
        accepted_route_report_sha256=hashlib.sha256(report_raw).hexdigest(),
        accepted_route_corpus_sha256=report.corpus_sha256,
        accepted_route_report=report,
        tpu_cake_git_commit="a" * 40,
        tpu_cake_uv_lock_sha256="b" * 64,
        runner_source_sha256="c" * 64,
        verifier_source_sha256=hashlib.sha256(verifier_path.read_bytes()).hexdigest(),
        inkling_git_commit="e" * 40,
        inkling_uv_lock_sha256="f" * 64,
        implementation_source_manifest=tuple(
            SourceFileContract(path=path, sha256=f"{index + 1:064x}")
            for index, path in enumerate(GMM_IMPLEMENTATION_SOURCE_PATHS)
        ),
    )


def _hlo_text(custom_call_counts: dict[str, int]) -> str:
    instructions = "\n".join(
        f"  %{scope}.{index} = f32[] custom-call(), "
        f'custom_call_target="tpu_custom_call", metadata={{op_name="{scope}"}}'
        for scope, count in custom_call_counts.items()
        for index in range(count)
    )
    return (
        "HloModule test_module\n\n"
        "ENTRY %main () -> f32[] {\n"
        f"{instructions}\n"
        "  ROOT %root = f32[] constant(0)\n"
        "}\n"
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _correctness_report(contract, report) -> GmmCorrectnessGateReport:
    digest = "1" * 64
    stage = StageOutputMeasurement(
        active_sha256=digest,
        nonfinite_count=0,
        outside_nonzero_count=0,
    )
    outputs = OutputMeasurement(gate=stage, up=stage, down=stage)
    operands = tuple(
        OperandSentinelMeasurement(
            name=name,
            shape=(1,),
            dtype="bfloat16",
            indices=((0,),),
            descriptor_sha256=digest,
            sentinel_sha256=digest,
        )
        for name in ("inputs", "gate", "up", "down")
    )
    profiles = []
    for profile_index, (profile, seed) in enumerate(
        zip(contract.correctness.profiles, contract.correctness.seeds, strict=True)
    ):
        cpu_count = (
            sum(row.profile_index == profile_index for row in contract.correctness.cpu_oracle_rows)
            * 3
        )
        profiles.append(
            ProfileCorrectnessMeasurement(
                profile_index=profile_index,
                seed=seed,
                completion_step=profile.completion_step,
                layer_index=profile.layer_index,
                group_sizes_sha256=profile.group_sizes_sha256,
                operands=operands,
                policies=tuple(
                    PolicyCorrectnessMeasurement(policy=policy, outputs=outputs)
                    for policy in _expected_policies(contract)
                ),
                cpu_oracle=tuple(
                    CpuOracleMeasurement(
                        stage=f"stage-{index}",
                        expected_sha256=digest,
                        actual_sha256=digest,
                        maximum_absolute_error=0,
                        maximum_relative_error=0,
                    )
                    for index in range(cpu_count)
                ),
            )
        )
    provisional = GmmCorrectnessGateReport(
        report_id="0" * 64,
        search_id=contract.search_id,
        route_report_id=report.report_id,
        numerical_contract_id=contract.correctness.numerical_contract_id,
        profiles=tuple(profiles),
    )
    return provisional.model_copy(
        update={
            "report_id": _json_sha256(provisional.model_dump(mode="json", exclude={"report_id"}))
        }
    )


def _bundle(tmp_path: Path) -> tuple[dict[str, Path], dict[str, object]]:
    report, report_raw = _route_report()
    contract = _contract(report, report_raw)
    contract_path = tmp_path / "contract.json"
    report_path = tmp_path / "route-report.json"
    raw_path = tmp_path / "raw.json"
    contract_path.write_text(
        json.dumps(
            contract.model_dump(mode="json", exclude_computed_fields=True),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    report_path.write_bytes(report_raw)
    correctness_path = tmp_path / "correctness-gate.json"
    correctness = _correctness_report(contract, report)
    correctness_path.write_text(
        json.dumps(
            correctness.model_dump(mode="json", exclude_computed_fields=True),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    hlo_root = tmp_path / "hlo"
    hlo_root.mkdir()
    compiled = []
    for index, policy in enumerate(_expected_policies(contract)):
        stablehlo = "module @jit_chain {\n  func.func public @main() {\n    return\n  }\n}\n"
        custom_call_counts = _expected_custom_call_counts(contract, policy)
        compiler_hlo = _hlo_text(custom_call_counts)
        stablehlo_path = hlo_root / f"{index}.stablehlo.mlir"
        compiler_hlo_path = hlo_root / f"{index}.compiler-hlo.txt"
        stablehlo_path.write_text(stablehlo)
        compiler_hlo_path.write_text(compiler_hlo)
        compiled.append(
            {
                "policy": policy.model_dump(mode="json"),
                "stablehlo_path": stablehlo_path.relative_to(tmp_path).as_posix(),
                "stablehlo_sha256": hashlib.sha256(stablehlo.encode()).hexdigest(),
                "compiler_hlo_path": compiler_hlo_path.relative_to(tmp_path).as_posix(),
                "compiler_hlo_sha256": hashlib.sha256(compiler_hlo.encode()).hexdigest(),
                "gmm_scope_labels": list(_expected_scopes(contract, policy)),
                "gmm_custom_call_counts": custom_call_counts,
                "stablehlo_bytes": len(stablehlo.encode()),
                "compiler_hlo_bytes": len(compiler_hlo.encode()),
            }
        )
    durations = {
        GmmSearchFamily.GATE_UP: {
            GmmArmName.INCUMBENT: 100,
            GmmArmName.SPARSE_M64: 90,
            GmmArmName.SPARSE_M32: 95,
            GmmArmName.SPLIT_N: 105,
            GmmArmName.SPARSE_M64_SPLIT_N: 110,
        },
        GmmSearchFamily.DOWN: {
            GmmArmName.INCUMBENT: 100,
            GmmArmName.SPARSE_M64: 90,
            GmmArmName.SPARSE_M32: 95,
            GmmArmName.SPLIT_N: 80,
            GmmArmName.SPARSE_M64_SPLIT_N: 110,
        },
    }
    observations = [
        {
            "family": family.value,
            "round_index": round_index,
            "position": position,
            "arm": arm.value,
            "duration_ns": durations[family][arm] + round_index,
        }
        for family in contract.search.families
        for round_index, order in enumerate(_screening_orders(contract, family))
        for position, arm in enumerate(order)
    ]
    devices = [
        {
            "id": device_id,
            "process_index": 0,
            "platform": "tpu",
            "device_kind": "TPU7x",
            "coords": [device_id // 4, (device_id // 2) % 2, 0],
            "core_on_chip": device_id % 2,
        }
        for device_id in range(8)
    ]
    raw = {
        "schema_version": "inkling-gmm-tile-search-runner-observations-v1",
        "search_id": contract.search_id,
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "route_report_id": report.report_id,
        "route_report_sha256": hashlib.sha256(report_raw).hexdigest(),
        "source_environment": {
            "tpu_cake_git_commit": contract.tpu_cake_git_commit,
            "tpu_cake_uv_lock_sha256": contract.tpu_cake_uv_lock_sha256,
            "runner_source_sha256": contract.runner_source_sha256,
            "verifier_source_sha256": contract.verifier_source_sha256,
            "inkling_git_commit": contract.inkling_git_commit,
            "inkling_uv_lock_sha256": contract.inkling_uv_lock_sha256,
        },
        "execution_target": {
            "project_id": contract.target_runtime.project_id,
            "zone": contract.target_runtime.zone,
            "instance_name": contract.target_runtime.instance_name,
            "accelerator_type": contract.target_runtime.accelerator_type,
        },
        "correctness_gate": {
            "schema_version": correctness.schema_version,
            "report_id": correctness.report_id,
            "path": correctness_path.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(correctness_path.read_bytes()).hexdigest(),
        },
        "runtime": {
            "jax": "0.11.0",
            "jaxlib": "0.11.0",
            "libtpu": "0.0.44.1",
            "process_count": 1,
            "process_index": 0,
            "devices": devices,
        },
        "residency": {
            "estimated_operand_bytes_per_device": _estimated_operand_bytes_per_device(contract),
            "free_memory_before_allocation": [90 * 1024**3] * 8,
            "free_memory_before_timing": [20 * 1024**3] * 8,
        },
        "compiled_policies": compiled,
        "screening_observations": observations,
        "limitations": [
            "The pre-timing correctness report is bound but not independently replayed here.",
            "This runner does not create an immutable receipt.",
            "This runner does not make or authorize a promotion decision.",
        ],
    }
    _write_json(raw_path, raw)
    return {
        "contract": contract_path,
        "route_report": report_path,
        "raw": raw_path,
    }, raw


def _verify(paths: dict[str, Path]):
    return verify_screening(
        contract_path=paths["contract"],
        route_report_path=paths["route_report"],
        raw_observations_path=paths["raw"],
    )


def test_verifier_recomputes_the_balanced_screens_and_emits_only_screening_claims(
    tmp_path: Path,
) -> None:
    paths, _ = _bundle(tmp_path)

    first = _verify(paths)
    second = _verify(paths)

    assert first == second
    assert first["evidence_scope"] == "screening-only"
    assert first["finalists"] == {"gate-up": "sparse-m64", "down": "split-n"}
    assert first["claims"] == {
        "correctness_gate_bound": True,
        "correctness_independently_replayed": False,
        "confirmation_run": False,
        "immutable_receipt_created": False,
        "promotion_authorized": False,
    }
    assert len(first["compiled_policies"]) == 9
    assert len(first["screening_statistics"]) == 2
    for family in first["screening_statistics"]:
        assert len(family["execution_orders"]) == 10
        assert all(len(order) == 5 for order in family["execution_orders"])
        for arm in family["arms"]:
            assert len(arm["durations_ns"]) == 10
    output = tmp_path / "verified.json"
    write_verified_report(output, first)
    assert json.loads(output.read_text()) == first


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        (lambda raw: raw.update({"unexpected": True}), "RAW_SCHEMA"),
        (
            lambda raw: raw["source_environment"].update({"runner_source_sha256": "0" * 64}),
            "SOURCE_BINDING",
        ),
        (lambda raw: raw["runtime"].update({"process_count": 2}), "RUNTIME_PROCESS"),
        (
            lambda raw: raw["execution_target"].update({"instance_name": "other-tpu"}),
            "EXECUTION_TARGET_BINDING",
        ),
        (
            lambda raw: raw["residency"].update({"free_memory_before_allocation": [1] * 8}),
            "RESIDENCY_ALLOCATION_GATE",
        ),
        (
            lambda raw: raw["screening_observations"].__setitem__(
                slice(0, 2), list(reversed(raw["screening_observations"][:2]))
            ),
            "OBSERVATION_ORDER",
        ),
        (
            lambda raw: raw["limitations"].append("The candidate is correct."),
            "LIMITATIONS",
        ),
    ),
)
def test_verifier_rejects_schema_provenance_runtime_and_order_forgery(
    tmp_path: Path,
    mutation,
    error: str,
) -> None:
    paths, raw = _bundle(tmp_path)
    mutation(raw)
    _write_json(paths["raw"], raw)

    with pytest.raises(ValueError, match=error):
        _verify(paths)


def test_verifier_rejects_a_forged_hlo_even_when_its_hash_is_rebound(tmp_path: Path) -> None:
    paths, raw = _bundle(tmp_path)
    item = raw["compiled_policies"][0]
    hlo_path = tmp_path / item["compiler_hlo_path"]
    forged = hlo_path.read_text().replace("gmm_v2-g_32", "gmm_v2-g_31", 1)
    hlo_path.write_text(forged)
    item["compiler_hlo_sha256"] = hashlib.sha256(forged.encode()).hexdigest()
    item["compiler_hlo_bytes"] = len(forged.encode())
    _write_json(paths["raw"], raw)

    with pytest.raises(ValueError, match="COMPILER_HLO_SCOPES"):
        _verify(paths)


def test_verifier_rejects_scope_metadata_outside_an_hlo_instruction(tmp_path: Path) -> None:
    paths, raw = _bundle(tmp_path)
    item = raw["compiled_policies"][0]
    hlo_path = tmp_path / item["compiler_hlo_path"]
    lines = hlo_path.read_text().splitlines()
    scope_line = next(index for index, line in enumerate(lines) if "gmm_v2-" in line)
    label = next(iter(item["gmm_scope_labels"]))
    lines[scope_line] = f"  // forged metadata {label}"
    forged = "\n".join(lines) + "\n"
    hlo_path.write_text(forged)
    item["compiler_hlo_sha256"] = hashlib.sha256(forged.encode()).hexdigest()
    item["compiler_hlo_bytes"] = len(forged.encode())
    _write_json(paths["raw"], raw)

    with pytest.raises(ValueError, match="COMPILER_HLO_SCOPE_INSTRUCTION"):
        _verify(paths)


def test_verifier_rejects_a_missing_custom_call_with_rebound_artifact_hash(
    tmp_path: Path,
) -> None:
    paths, raw = _bundle(tmp_path)
    item = raw["compiled_policies"][0]
    hlo_path = tmp_path / item["compiler_hlo_path"]
    lines = hlo_path.read_text().splitlines()
    custom_call = next(
        index for index, line in enumerate(lines) if "gmm_v2-" in line and "custom-call(" in line
    )
    del lines[custom_call]
    forged = "\n".join(lines) + "\n"
    hlo_path.write_text(forged)
    item["compiler_hlo_sha256"] = hashlib.sha256(forged.encode()).hexdigest()
    item["compiler_hlo_bytes"] = len(forged.encode())
    _write_json(paths["raw"], raw)

    with pytest.raises(ValueError, match="COMPILER_HLO_CUSTOM_CALL_COUNTS"):
        _verify(paths)


def test_verifier_rejects_artifact_path_escape(tmp_path: Path) -> None:
    paths, raw = _bundle(tmp_path)
    raw["compiled_policies"][0]["compiler_hlo_path"] = "../outside.txt"
    _write_json(paths["raw"], raw)

    with pytest.raises(ValueError, match="COMPILER_HLO_PATH"):
        _verify(paths)


def test_verifier_rejects_a_symlinked_artifact(tmp_path: Path) -> None:
    paths, raw = _bundle(tmp_path)
    item = raw["compiled_policies"][0]
    original = tmp_path / item["compiler_hlo_path"]
    target = tmp_path / "copied-hlo.txt"
    target.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(target)

    with pytest.raises(ValueError, match="COMPILER_HLO_PATH"):
        _verify(paths)
