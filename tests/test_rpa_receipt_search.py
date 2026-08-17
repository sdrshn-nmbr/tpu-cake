from pathlib import Path

import pytest

from tpu_cake.contracts import RpaSearchSelection, RuntimeIdentity
from tpu_cake.rpa_receipt_search import (
    _copy_search_evidence,
    _run_identity,
    _search_provenance,
    _search_selection,
    build_search_bound_fused_rpa_receipt,
)
from tpu_cake.rpa_runner import FusedRpaRunResult
from tpu_cake.rpa_search import (
    RpaSearchContract,
    RpaSearchResult,
    RpaSearchRunEvidence,
)
from tpu_cake.workloads.inkling_rpa import inkling_fused_rpa_experiment


def _contract() -> RpaSearchContract:
    return RpaSearchContract.model_validate_json(
        Path("contracts/inkling-fused-rpa-block-search.json").read_text()
    )


def _run_result(**updates: object) -> FusedRpaRunResult:
    values = {
        "schedule_sha256": "1" * 64,
        "pallas_source_sha256": "2" * 64,
        "stablehlo_sha256": "3" * 64,
        "compiler_hlo_sha256": "4" * 64,
        "input_sha256": ("5" * 64,),
        "output_sha256": ("6" * 64,),
        "oracle_sha256": ("7" * 64,),
        "runtime": RuntimeIdentity(python="3.12.3", jax="0.11.0"),
        "backend": "tpu",
        "device_kind": "TPU7x",
        "device_count": 8,
        "execution_scope": "local-shard-caller-owned-sharding",
        "backend_executor": "module.kernel",
        "backend_executor_sha256": "8" * 64,
        "backend_manifest": (),
    }
    values.update(updates)
    return FusedRpaRunResult.model_construct(**values)


@pytest.mark.parametrize(
    ("provisional", "winner", "selected", "selection"),
    (
        (
            None,
            None,
            "incumbent",
            RpaSearchSelection.BASELINE_RETAINED_NO_PROMOTABLE_CHALLENGER,
        ),
        (
            "kv-64",
            None,
            "incumbent",
            RpaSearchSelection.BASELINE_RETAINED_CONFIRMATION_FAILED,
        ),
        (
            "kv-64",
            "kv-64",
            "kv-64",
            RpaSearchSelection.CHALLENGER_PROMOTED,
        ),
    ),
)
def test_rpa_search_selection_records_why_a_schedule_was_chosen(
    provisional: str | None,
    winner: str | None,
    selected: str,
    selection: RpaSearchSelection,
) -> None:
    result = RpaSearchResult.model_construct(
        baseline="incumbent",
        provisional_winner=provisional,
        winner=winner,
    )

    assert _search_selection(result) == (selected, selection)


def test_rpa_search_provenance_binds_the_selected_schedule(tmp_path: Path) -> None:
    contract = _contract()
    selected_run = tmp_path / "round-00/00-incumbent"
    selected_run.mkdir(parents=True)
    (selected_run / "profiler_config.json").write_text("profiler\n")
    result = RpaSearchResult.model_construct(
        search_id=contract.search_id,
        baseline="incumbent",
        provisional_winner=None,
        winner=None,
        runs=(
            RpaSearchRunEvidence.model_construct(
                path="round-00/00-incumbent",
                candidate="incumbent",
                result_sha256="1" * 64,
            ),
            *tuple(
                RpaSearchRunEvidence.model_construct(
                    path=f"round-{index:02d}/00-kv-64",
                    candidate="kv-64",
                    result_sha256="2" * 64,
                )
                for index in range(49)
            ),
        ),
    )
    (tmp_path / "contract.json").write_text("contract\n")
    (tmp_path / "result.json").write_text("result\n")

    provenance = _search_provenance(tmp_path, contract, result)

    assert provenance.selection is (RpaSearchSelection.BASELINE_RETAINED_NO_PROMOTABLE_CHALLENGER)
    assert provenance.selected_candidate == "incumbent"
    assert provenance.selected_block_sizes == (8, 128, 8, 128)
    assert (
        provenance.selected_schedule_sha256
        == inkling_fused_rpa_experiment((8, 128, 8, 128)).schedule_sha256
    )
    assert provenance.run_count == 50


@pytest.mark.parametrize(
    ("field", "changed"),
    (
        ("pallas_source_sha256", "a" * 64),
        ("stablehlo_sha256", "b" * 64),
        ("compiler_hlo_sha256", "c" * 64),
        ("input_sha256", ("d" * 64,)),
        ("output_sha256", ("e" * 64,)),
        ("oracle_sha256", ("f" * 64,)),
        ("device_count", 4),
        ("backend_executor", "module.other"),
    ),
)
def test_finalist_identity_covers_execution_evidence(
    field: str,
    changed: object,
) -> None:
    assert _run_identity(_run_result()) != _run_identity(_run_result(**{field: changed}))


def test_search_evidence_copy_is_an_independent_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "trace.pb").write_bytes(b"before")
    destination = tmp_path / "destination"

    _copy_search_evidence(source, destination)
    (source / "trace.pb").write_bytes(b"after")

    assert (destination / "trace.pb").read_bytes() == b"before"
    assert (source / "trace.pb").stat().st_ino != (destination / "trace.pb").stat().st_ino


@pytest.mark.parametrize("source_inside_final", (False, True))
def test_search_and_final_roots_must_be_disjoint(
    tmp_path: Path,
    source_inside_final: bool,
) -> None:
    final = tmp_path / "final"
    search = final / "search-source" if source_inside_final else tmp_path / "search"
    search.mkdir(parents=True)
    if not source_inside_final:
        final = search / "nested-final"
    with pytest.raises(ValueError, match="MUST_BE_DISJOINT"):
        build_search_bound_fused_rpa_receipt(
            final,
            search,
            _contract(),
        )
