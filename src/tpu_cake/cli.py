from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin
from xdsl.parser import Parser

from tpu_cake.contracts import (
    KernelExperiment,
    ProfileExpectation,
    RunReceipt,
    experiment_artifact_json,
)
from tpu_cake.dialects.distributed_tensor import DistributedTensor
from tpu_cake.dialects.tpu_schedule import TPUSchedule
from tpu_cake.frontend import canonical_module_text
from tpu_cake.receipt import validate_receipt
from tpu_cake.rpa_bundle import build_fused_rpa_receipt, validate_fused_rpa_receipt
from tpu_cake.rpa_receipt_search import (
    build_search_bound_fused_rpa_receipt,
    validate_search_bound_fused_rpa_receipt,
)
from tpu_cake.rpa_search import RpaSearchContract, validate_rpa_search_result
from tpu_cake.run_bundle import build_distributed_matmul_receipt
from tpu_cake.runner import RunMode, run_distributed_matmul
from tpu_cake.search import MatmulSearchContract, run_matmul_search
from tpu_cake.seqax_bundle import (
    build_seqax_forward_receipt,
    validate_seqax_forward_receipt,
)
from tpu_cake.seqax_pallas_bundle import (
    build_seqax_pallas_receipt,
    validate_seqax_pallas_receipt,
)
from tpu_cake.seqax_pallas_diagnostic import (
    run_seqax_pallas_incumbent_diagnostic,
    validate_seqax_pallas_incumbent_diagnostic,
)
from tpu_cake.seqax_pallas_runner import run_seqax_physical_pallas
from tpu_cake.seqax_pallas_search import SeqaxPallasSearchContract
from tpu_cake.seqax_pallas_search_runner import (
    run_seqax_pallas_search,
    validate_seqax_pallas_search,
)
from tpu_cake.seqax_runner import run_seqax_forward
from tpu_cake.seqax_surface import (
    SeqaxSurfaceReceipt,
    run_seqax_surface,
    validate_seqax_surface_receipt,
)
from tpu_cake.seqax_surface_profile import (
    SeqaxSurfaceProfileReceipt,
    build_seqax_surface_profile_receipt,
    run_seqax_surface_profile_phase,
    validate_seqax_surface_profile_receipt,
)
from tpu_cake.seqax_weight_placement import (
    SeqaxWeightPlacementContract,
    SeqaxWeightPlacementName,
)
from tpu_cake.seqax_weight_placement_runner import (
    probe_weight_placement_memory,
    run_seqax_weight_placement,
    validate_seqax_weight_placement,
)
from tpu_cake.workloads import (
    inkling_fused_rpa_experiment,
    inkling_rpa_experiment,
    inkling_rpa_schedule,
    matmul_experiment,
    matmul_schedule,
)
from tpu_cake.xprof_evidence import assess_capture, capture_metrics

_WORKLOADS = {
    "matmul": (matmul_schedule, matmul_experiment),
    "inkling-rpa": (inkling_rpa_schedule, inkling_rpa_experiment),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tpu-cake")
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify-schedule")
    verify.add_argument("schedule", type=Path)

    inspect = commands.add_parser("inspect-profile")
    inspect.add_argument("capture", type=Path)
    inspect.add_argument("--contract", required=True, type=Path)
    inspect.add_argument("--output", type=Path)

    render = commands.add_parser("render-workload")
    render.add_argument("workload", choices=tuple(_WORKLOADS))
    render.add_argument("--output", type=Path)

    experiment = commands.add_parser("experiment")
    experiment.add_argument("workload", choices=tuple(_WORKLOADS))
    experiment.add_argument("--output", type=Path)

    run = commands.add_parser("run-matmul")
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--mode", choices=tuple(RunMode), default=RunMode.TIMING)
    run.add_argument("--mesh-size", type=int, default=8)
    run.add_argument("--m", type=int, default=128)
    run.add_argument("--k", type=int, default=1024)
    run.add_argument("--n", type=int, default=1024)
    run.add_argument("--warmup-iterations", type=int, default=10)
    run.add_argument("--measured-iterations", type=int, default=100)
    run.add_argument("--tile-m", type=int)
    run.add_argument("--tile-n", type=int)
    run.add_argument("--interpret", action="store_true")

    finalize = commands.add_parser("finalize-matmul-run")
    finalize.add_argument("run_root", type=Path)
    finalize.add_argument("--search-root", type=Path)

    verify_bundle = commands.add_parser("verify-matmul-bundle")
    verify_bundle.add_argument("run_root", type=Path)

    finalize_rpa = commands.add_parser("finalize-rpa-run")
    finalize_rpa.add_argument("run_root", type=Path)
    finalize_rpa.add_argument("--search-root", type=Path)
    finalize_rpa.add_argument("--search-contract", type=Path)

    verify_rpa_bundle = commands.add_parser("verify-rpa-bundle")
    verify_rpa_bundle.add_argument("run_root", type=Path)
    verify_rpa_bundle.add_argument("--search-contract", type=Path)
    verify_rpa_search = commands.add_parser("verify-rpa-search")
    verify_rpa_search.add_argument("run_root", type=Path)
    verify_rpa_search.add_argument("--contract", type=Path, required=True)

    run_seqax = commands.add_parser("run-seqax-forward")
    run_seqax.add_argument("--output-dir", required=True, type=Path)
    run_seqax.add_argument("--mode", choices=tuple(RunMode), default=RunMode.TIMING)

    run_seqax_pallas = commands.add_parser("run-seqax-physical-pallas")
    run_seqax_pallas.add_argument("--output-dir", required=True, type=Path)
    run_seqax_pallas.add_argument(
        "--mode",
        choices=tuple(RunMode),
        default=RunMode.TIMING,
    )

    finalize_seqax_pallas = commands.add_parser("finalize-seqax-physical-pallas")
    finalize_seqax_pallas.add_argument("run_root", type=Path)

    verify_seqax_pallas = commands.add_parser("verify-seqax-physical-pallas")
    verify_seqax_pallas.add_argument("run_root", type=Path)

    search_seqax_pallas = commands.add_parser("search-seqax-physical-pallas")
    search_seqax_pallas.add_argument("--contract", required=True, type=Path)
    search_seqax_pallas.add_argument("--output-dir", required=True, type=Path)

    verify_seqax_pallas_search = commands.add_parser("verify-seqax-physical-pallas-search")
    verify_seqax_pallas_search.add_argument("run_root", type=Path)
    verify_seqax_pallas_search.add_argument("--contract", required=True, type=Path)

    search_seqax_weight_placement = commands.add_parser("search-seqax-weight-placement")
    search_seqax_weight_placement.add_argument("--contract", required=True, type=Path)
    search_seqax_weight_placement.add_argument("--output-dir", required=True, type=Path)

    verify_seqax_weight_placement = commands.add_parser("verify-seqax-weight-placement")
    verify_seqax_weight_placement.add_argument("run_root", type=Path)
    verify_seqax_weight_placement.add_argument("--contract", required=True, type=Path)

    probe_seqax_weight_memory = commands.add_parser("probe-seqax-weight-placement-memory")
    probe_seqax_weight_memory.add_argument(
        "--candidate",
        required=True,
        choices=tuple(SeqaxWeightPlacementName),
    )

    diagnose_seqax_pallas = commands.add_parser("diagnose-seqax-physical-pallas")
    diagnose_seqax_pallas.add_argument("--search-root", required=True, type=Path)
    diagnose_seqax_pallas.add_argument("--contract", required=True, type=Path)
    diagnose_seqax_pallas.add_argument("--output-dir", required=True, type=Path)

    verify_seqax_pallas_diagnostic = commands.add_parser("verify-seqax-physical-pallas-diagnostic")
    verify_seqax_pallas_diagnostic.add_argument("run_root", type=Path)

    finalize_seqax = commands.add_parser("finalize-seqax-forward")
    finalize_seqax.add_argument("run_root", type=Path)

    verify_seqax = commands.add_parser("verify-seqax-forward")
    verify_seqax.add_argument("run_root", type=Path)

    run_seqax_surface = commands.add_parser("run-seqax-surface")
    run_seqax_surface.add_argument("--output-dir", required=True, type=Path)

    verify_seqax_surface = commands.add_parser("verify-seqax-surface")
    verify_seqax_surface.add_argument("run_root", type=Path)

    run_seqax_surface_profile = commands.add_parser("run-seqax-surface-profile")
    run_seqax_surface_profile.add_argument("--output-dir", required=True, type=Path)
    run_seqax_surface_profile.add_argument("--surface-root", required=True, type=Path)
    run_seqax_surface_profile.add_argument(
        "--scenario", choices=("tiny", "wider", "deeper"), required=True
    )
    run_seqax_surface_profile.add_argument(
        "--mode", choices=(RunMode.TRACE, RunMode.COUNTERS), required=True
    )

    finalize_seqax_surface_profile = commands.add_parser("finalize-seqax-surface-profile")
    finalize_seqax_surface_profile.add_argument("run_root", type=Path)
    finalize_seqax_surface_profile.add_argument("--surface-root", required=True, type=Path)

    verify_seqax_surface_profile = commands.add_parser("verify-seqax-surface-profile")
    verify_seqax_surface_profile.add_argument("run_root", type=Path)

    search = commands.add_parser("search-matmul")
    search.add_argument("contract", type=Path)
    search.add_argument("--output-dir", required=True, type=Path)
    search.add_argument("--interpret", action="store_true")
    return parser


def _verify_schedule(path: Path) -> int:
    context = Context()
    context.load_dialect(Builtin)
    context.load_dialect(DistributedTensor)
    context.load_dialect(TPUSchedule)
    module = Parser(context, path.read_text(), name=str(path)).parse_module()
    module.verify()
    print(f"SCHEDULE_ACCEPTED path={path}")
    return 0


def _inspect_profile(capture: Path, contract_path: Path, output: Path | None) -> int:
    contract = ProfileExpectation.model_validate(tomllib.loads(contract_path.read_text()))
    assessment = assess_capture(capture, contract)
    verdict = "ACCEPTED" if assessment.accepted else "REJECTED"
    print(f"PROFILE_{verdict} contract={contract.name}")
    for finding in assessment.findings:
        print(f"{finding.severity.value.upper()} {finding.code}: {finding.message}")
    print(
        "EVIDENCE "
        f"tpu_device_planes={sum(plane.name.startswith('/device:TPU:') and 'SparseCore' not in plane.name for plane in assessment.capture.planes)} "
        f"timed_programs={sorted(assessment.capture.timed_program_ids)} "
        "periodic_counter_series_derivable="
        f"{assessment.capture.counters.periodic_series_derivable}"
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "assessment": assessment.model_dump(mode="json"),
            "metrics": [
                metric.model_dump(mode="json") for metric in capture_metrics(assessment.capture)
            ],
        }
        output.write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if assessment.accepted else 1


def _render_workload(workload: str, output: Path | None) -> int:
    schedule = _WORKLOADS[workload][0]()
    rendered = canonical_module_text(schedule)
    if output is None:
        print(rendered, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
        print(f"WORKLOAD_RENDERED workload={workload} path={output}")
    return 0


def _experiment(workload: str, output: Path | None) -> int:
    experiment = _WORKLOADS[workload][1]()
    rendered = experiment_artifact_json(experiment) + "\n"
    if output is None:
        print(rendered, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
        print(f"EXPERIMENT_WRITTEN workload={workload} id={experiment.experiment_id} path={output}")
    return 0


def _verify_rpa_bundle(root: Path, search_contract_path: Path | None) -> int:
    root = root.resolve()
    receipt = RunReceipt.model_validate_json((root / "receipt.json").read_text())
    if receipt.rpa_search_provenance is None:
        if search_contract_path is not None:
            raise ValueError("RPA_SEARCH_PROVENANCE_REQUIRED")
        validate_fused_rpa_receipt(receipt, inkling_fused_rpa_experiment(), root=root)
    else:
        if search_contract_path is None:
            raise ValueError("RPA_SEARCH_CONTRACT_REQUIRED")
        contract = RpaSearchContract.model_validate_json(search_contract_path.read_text())
        validate_search_bound_fused_rpa_receipt(
            receipt,
            root=root,
            expected_contract=contract,
        )
    verdict = "ACCEPTED" if receipt.status.value == "passed" else "REJECTED"
    print(
        f"RPA_BUNDLE_{verdict} status={receipt.status.value} "
        f"artifacts={len(receipt.artifacts)} metrics={len(receipt.metrics)}"
    )
    return 0 if receipt.status.value == "passed" else 1


def main() -> None:
    args = _parser().parse_args()
    if args.command == "verify-schedule":
        code = _verify_schedule(args.schedule)
    elif args.command == "inspect-profile":
        code = _inspect_profile(args.capture, args.contract, args.output)
    elif args.command == "render-workload":
        code = _render_workload(args.workload, args.output)
    elif args.command == "run-matmul":
        result = run_distributed_matmul(
            args.output_dir,
            mode=RunMode(args.mode),
            mesh_size=args.mesh_size,
            m=args.m,
            k=args.k,
            n=args.n,
            warmup_iterations=args.warmup_iterations,
            measured_iterations=args.measured_iterations,
            tile_m=args.tile_m,
            tile_n=args.tile_n,
            interpret=args.interpret,
        )
        print(result.model_dump_json(indent=2))
        code = 0 if result.passed else 1
    elif args.command == "finalize-matmul-run":
        receipt = build_distributed_matmul_receipt(
            args.run_root,
            search_root=args.search_root,
        )
        print(receipt.model_dump_json(indent=2))
        code = 0 if receipt.status.value == "passed" else 1
    elif args.command == "verify-matmul-bundle":
        root = args.run_root.resolve()
        receipt = RunReceipt.model_validate_json((root / "receipt.json").read_text())
        experiment = KernelExperiment.model_validate_json(
            (root / "timing" / "experiment.json").read_text()
        )
        validate_receipt(receipt, experiment, root=root)
        print(
            f"MATMUL_BUNDLE_ACCEPTED status={receipt.status.value} "
            f"artifacts={len(receipt.artifacts)} metrics={len(receipt.metrics)}"
        )
        code = 0
    elif args.command == "finalize-rpa-run":
        if (args.search_root is None) != (args.search_contract is None):
            raise ValueError("RPA_SEARCH_ROOT_AND_CONTRACT_REQUIRED_TOGETHER")
        if args.search_root is None:
            receipt = build_fused_rpa_receipt(args.run_root)
        else:
            contract = RpaSearchContract.model_validate_json(args.search_contract.read_text())
            receipt = build_search_bound_fused_rpa_receipt(
                args.run_root,
                args.search_root,
                contract,
            )
        print(receipt.model_dump_json(indent=2))
        code = 0 if receipt.status.value == "passed" else 1
    elif args.command == "verify-rpa-bundle":
        code = _verify_rpa_bundle(args.run_root, args.search_contract)
    elif args.command == "verify-rpa-search":
        contract = RpaSearchContract.model_validate_json(args.contract.read_text())
        result = validate_rpa_search_result(args.run_root, contract)
        print(f"RPA_SEARCH_ACCEPTED winner={result.winner or 'none'} runs={len(result.runs)}")
        code = 0
    elif args.command == "run-seqax-forward":
        result = run_seqax_forward(args.output_dir, mode=RunMode(args.mode))
        print(result.model_dump_json(indent=2))
        code = 0 if result.passed else 1
    elif args.command == "run-seqax-physical-pallas":
        result = run_seqax_physical_pallas(
            args.output_dir,
            mode=RunMode(args.mode),
        )
        print(
            "SEQAX_PHYSICAL_PALLAS_PHASE_COMPLETE "
            f"mode={result.mode.value} "
            "acceptance_requires_final_receipt=true"
        )
        print(result.model_dump_json(indent=2))
        code = 0 if result.correctness_passed else 1
    elif args.command == "finalize-seqax-physical-pallas":
        receipt = build_seqax_pallas_receipt(args.run_root)
        print(receipt.model_dump_json(indent=2))
        code = 0 if receipt.status.value == "passed" else 1
    elif args.command == "verify-seqax-physical-pallas":
        root = args.run_root.resolve()
        receipt = RunReceipt.model_validate_json((root / "receipt.json").read_text())
        validate_seqax_pallas_receipt(receipt, root=root)
        print(
            "SEQAX_PHYSICAL_PALLAS_ACCEPTED "
            f"status={receipt.status.value} artifacts={len(receipt.artifacts)} "
            f"metrics={len(receipt.metrics)}"
        )
        code = 0
    elif args.command == "search-seqax-physical-pallas":
        contract = SeqaxPallasSearchContract.model_validate_json(args.contract.read_text())
        result = run_seqax_pallas_search(args.output_dir, contract)
        print(result.model_dump_json(indent=2))
        code = 0
    elif args.command == "verify-seqax-physical-pallas-search":
        contract = SeqaxPallasSearchContract.model_validate_json(args.contract.read_text())
        result = validate_seqax_pallas_search(args.run_root, contract)
        print(
            "SEQAX_PHYSICAL_PALLAS_SEARCH_ACCEPTED "
            f"winner={result.winner or 'none'} candidates={len(result.candidates)} "
            f"primitive_cases={len(result.primitive_observations)}"
        )
        code = 0
    elif args.command == "search-seqax-weight-placement":
        contract = SeqaxWeightPlacementContract.model_validate_json(args.contract.read_text())
        result = run_seqax_weight_placement(args.output_dir, contract)
        print(result.model_dump_json(indent=2))
        code = 0
    elif args.command == "verify-seqax-weight-placement":
        contract = SeqaxWeightPlacementContract.model_validate_json(args.contract.read_text())
        result = validate_seqax_weight_placement(args.run_root, contract)
        print(
            "SEQAX_WEIGHT_PLACEMENT_ACCEPTED "
            f"winner={result.winner or 'none'} candidates={len(result.candidates)} "
            f"scope={result.correctness_scope}"
        )
        code = 0
    elif args.command == "probe-seqax-weight-placement-memory":
        observation = probe_weight_placement_memory(SeqaxWeightPlacementName(args.candidate))
        print("SEQAX_WEIGHT_PLACEMENT_MEMORY_JSON=" + observation.model_dump_json())
        code = 0
    elif args.command == "diagnose-seqax-physical-pallas":
        contract = SeqaxPallasSearchContract.model_validate_json(args.contract.read_text())
        receipt = run_seqax_pallas_incumbent_diagnostic(
            args.output_dir,
            args.search_root,
            contract,
        )
        print(receipt.model_dump_json(indent=2))
        code = 0
    elif args.command == "verify-seqax-physical-pallas-diagnostic":
        receipt = validate_seqax_pallas_incumbent_diagnostic(args.run_root)
        print(
            "SEQAX_PHYSICAL_PALLAS_DIAGNOSTIC_ACCEPTED "
            f"artifacts={len(receipt.artifacts)} search_id={receipt.search_id}"
        )
        code = 0
    elif args.command == "finalize-seqax-forward":
        receipt = build_seqax_forward_receipt(args.run_root)
        print(receipt.model_dump_json(indent=2))
        code = 0 if receipt.status.value == "passed" else 1
    elif args.command == "verify-seqax-forward":
        root = args.run_root.resolve()
        receipt = RunReceipt.model_validate_json((root / "receipt.json").read_text())
        validate_seqax_forward_receipt(receipt, root=root)
        verdict = "ACCEPTED" if receipt.status.value == "passed" else "REJECTED"
        print(
            f"SEQAX_FORWARD_{verdict} status={receipt.status.value} "
            f"artifacts={len(receipt.artifacts)} metrics={len(receipt.metrics)}"
        )
        code = 0 if receipt.status.value == "passed" else 1
    elif args.command == "run-seqax-surface":
        receipt = run_seqax_surface(args.output_dir)
        print(receipt.model_dump_json(indent=2))
        code = 0
    elif args.command == "verify-seqax-surface":
        root = args.run_root.resolve()
        receipt = SeqaxSurfaceReceipt.model_validate_json((root / "receipt.json").read_text())
        validate_seqax_surface_receipt(receipt, root=root)
        verdict = "PROMOTED" if receipt.candidate_promoted else "RETAINED_BASELINE"
        print(
            f"SEQAX_SURFACE_ACCEPTED decision={verdict} "
            f"scenarios={len(receipt.comparison.scenario_improvements)} "
            f"artifacts={len(receipt.artifacts)}"
        )
        code = 0
    elif args.command == "run-seqax-surface-profile":
        result = run_seqax_surface_profile_phase(
            args.output_dir,
            surface_root=args.surface_root,
            scenario_name=args.scenario,
            mode=RunMode(args.mode),
        )
        print(
            "SEQAX_SURFACE_PROFILE_PHASE_COMPLETE "
            f"scenario={result.invocation.scenario} mode={result.invocation.mode.value} "
            "acceptance_requires_final_receipt=true"
        )
        print(result.model_dump_json(indent=2))
        code = 0 if result.passed else 1
    elif args.command == "finalize-seqax-surface-profile":
        receipt = build_seqax_surface_profile_receipt(
            args.run_root,
            surface_root=args.surface_root,
        )
        print(receipt.model_dump_json(indent=2))
        code = 0 if receipt.accepted else 1
    elif args.command == "verify-seqax-surface-profile":
        root = args.run_root.resolve()
        receipt = SeqaxSurfaceProfileReceipt.model_validate_json(
            (root / "receipt.json").read_text()
        )
        validate_seqax_surface_profile_receipt(receipt, root=root)
        print(
            "SEQAX_SURFACE_PROFILE_ACCEPTED "
            f"scenarios={len(receipt.results) // 2} "
            f"captures={len(receipt.results)} artifacts={len(receipt.artifacts)} "
            f"metrics={len(receipt.metrics)}"
        )
        code = 0
    elif args.command == "search-matmul":
        contract = MatmulSearchContract.model_validate_json(args.contract.read_text())
        result = run_matmul_search(args.output_dir, contract, interpret=args.interpret)
        print(result.model_dump_json(indent=2))
        code = 0
    else:
        code = _experiment(args.workload, args.output)
    sys.exit(code)


if __name__ == "__main__":
    main()
