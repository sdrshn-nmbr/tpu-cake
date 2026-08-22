from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

from xdsl.dialects.builtin import ModuleOp

from tpu_cake.canonical import parse_tpu_cake_module
from tpu_cake.contracts import (
    KernelExperiment,
    ProfileExpectation,
    RunReceipt,
    experiment_artifact_json,
)
from tpu_cake.cost_model import tpu7x_tensorcore_rates
from tpu_cake.frontend import canonical_module_text
from tpu_cake.inkling_decode_profile import (
    InklingDecodeProfileContract,
    capture_inkling_decode_profile_request,
    inspect_inkling_decode_profile,
    validate_inkling_decode_profile,
    write_inkling_decode_profile_assessment,
)
from tpu_cake.matmul_collective_confirmation import MatmulCollectiveConfirmationContract
from tpu_cake.matmul_collective_confirmation_runner import (
    finalize_matmul_collective_confirmation,
    run_matmul_collective_confirmation,
    validate_matmul_collective_confirmation,
)
from tpu_cake.matmul_collective_repeat_prediction import (
    MatmulCollectiveRepeatPredictionContract,
    MatmulCollectiveRepeatPredictionReport,
    validate_matmul_collective_repeat_prediction,
    write_matmul_collective_repeat_prediction,
)
from tpu_cake.physical_cost_model import (
    PhysicalCollectiveLatencyCalibration,
    PhysicalKernelLatencyReport,
    PhysicalKernelResourceReport,
    analyze_physical_kernel,
    tpu7x_collective_latency_calibration,
    validate_physical_kernel_latency_report,
    validate_physical_kernel_report,
    write_physical_kernel_latency_report,
    write_physical_kernel_report,
)
from tpu_cake.physical_fusion import (
    SeqaxSiluMultiplyFusionContract,
    SeqaxSiluMultiplyFusionReport,
    validate_seqax_silu_multiply_fusion_report,
    write_seqax_silu_multiply_fusion_report,
)
from tpu_cake.receipt import validate_receipt
from tpu_cake.rpa_bundle import build_fused_rpa_receipt, validate_fused_rpa_receipt
from tpu_cake.rpa_donation_confirmation import InklingRpaDonationConfirmationContract
from tpu_cake.rpa_donation_confirmation_runner import (
    validate_inkling_rpa_donation_confirmation,
)
from tpu_cake.rpa_receipt_search import (
    build_search_bound_fused_rpa_receipt,
    validate_search_bound_fused_rpa_receipt,
)
from tpu_cake.rpa_search import RpaSearchContract, validate_rpa_search_result
from tpu_cake.rpa_surface import InklingShardedRpaSurfaceContract
from tpu_cake.rpa_surface_runner import (
    validate_inkling_sharded_rpa_relocation_attestation,
    validate_inkling_sharded_rpa_surface,
    write_inkling_sharded_rpa_relocation_attestation,
)
from tpu_cake.run_bundle import build_distributed_matmul_receipt
from tpu_cake.runner import MatmulCollectiveStrategy, RunMode, run_distributed_matmul
from tpu_cake.search import MatmulSearchContract, run_matmul_search
from tpu_cake.seqax_bundle import (
    build_seqax_forward_receipt,
    validate_seqax_forward_receipt,
)
from tpu_cake.seqax_cost_calibration import (
    SeqaxCostCalibrationContract,
    SeqaxCostCalibrationReport,
    validate_seqax_cost_calibration,
    write_seqax_cost_calibration,
)
from tpu_cake.seqax_numerical import SeqaxBf16ValidationContract
from tpu_cake.seqax_numerical_runner import (
    run_seqax_bf16_validation,
    validate_seqax_bf16_relocation_attestation,
    validate_seqax_bf16_validation,
    write_seqax_bf16_relocation_attestation,
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
from tpu_cake.seqax_residual_confirmation import SeqaxResidualConfirmationContract
from tpu_cake.seqax_residual_confirmation_runner import (
    run_seqax_residual_confirmation,
    validate_seqax_residual_confirmation,
)
from tpu_cake.seqax_residual_profile import SeqaxResidualProfileContract
from tpu_cake.seqax_residual_profile_runner import (
    capture_seqax_residual_profile_hlo_identities,
    run_seqax_residual_profile,
    validate_seqax_residual_profile,
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
from tpu_cake.seqax_weight_confirmation import SeqaxWeightConfirmationContract
from tpu_cake.seqax_weight_confirmation_runner import (
    run_seqax_weight_confirmation,
    validate_seqax_weight_confirmation,
)
from tpu_cake.seqax_weight_placement import (
    SeqaxWeightPlacementContract,
    SeqaxWeightPlacementName,
)
from tpu_cake.seqax_weight_placement_diagnostic import (
    SeqaxWeightPlacementDiagnosticContract,
)
from tpu_cake.seqax_weight_placement_diagnostic_runner import (
    run_seqax_weight_placement_diagnostic,
    validate_seqax_weight_placement_diagnostic,
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

    estimate_physical_cost = commands.add_parser("estimate-physical-cost")
    estimate_physical_cost.add_argument("schedule", type=Path)
    estimate_physical_cost.add_argument("--output", required=True, type=Path)

    verify_physical_cost = commands.add_parser("verify-physical-cost")
    verify_physical_cost.add_argument("report", type=Path)
    verify_physical_cost.add_argument("--schedule", required=True, type=Path)

    estimate_physical_latency = commands.add_parser("estimate-physical-latency")
    estimate_physical_latency.add_argument("schedule", type=Path)
    estimate_physical_latency.add_argument("--calibration", required=True, type=Path)
    estimate_physical_latency.add_argument("--output", required=True, type=Path)

    verify_physical_latency = commands.add_parser("verify-physical-latency")
    verify_physical_latency.add_argument("report", type=Path)
    verify_physical_latency.add_argument("--schedule", required=True, type=Path)
    verify_physical_latency.add_argument("--calibration", required=True, type=Path)

    compare_physical_fusion = commands.add_parser("compare-seqax-silu-fusion")
    compare_physical_fusion.add_argument("contract", type=Path)
    compare_physical_fusion.add_argument("--output", required=True, type=Path)

    verify_physical_fusion = commands.add_parser("verify-seqax-silu-fusion")
    verify_physical_fusion.add_argument("report", type=Path)
    verify_physical_fusion.add_argument("--contract", required=True, type=Path)

    inspect = commands.add_parser("inspect-profile")
    inspect.add_argument("capture", type=Path)
    inspect.add_argument("--contract", required=True, type=Path)
    inspect.add_argument("--output", type=Path)

    capture_inkling_decode = commands.add_parser("capture-inkling-decode-profile-request")
    capture_inkling_decode.add_argument("--contract", required=True, type=Path)
    capture_inkling_decode.add_argument("--url", default="http://127.0.0.1:30000")
    capture_inkling_decode.add_argument("--output", required=True, type=Path)
    capture_inkling_decode.add_argument("--profile-root", required=True, type=Path)
    capture_inkling_decode.add_argument("--prompt-cases", required=True, type=Path)
    capture_inkling_decode.add_argument("--inkling-repo", required=True, type=Path)

    for name in ("inspect-inkling-decode-profile", "verify-inkling-decode-profile"):
        command = commands.add_parser(name)
        command.add_argument("capture", type=Path)
        command.add_argument("--request", required=True, type=Path)
        command.add_argument("--prompt-cases", required=True, type=Path)
        command.add_argument("--contract", required=True, type=Path)
        command.add_argument("--output", type=Path)

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
    run.add_argument(
        "--collective-strategy",
        choices=tuple(MatmulCollectiveStrategy),
        default=MatmulCollectiveStrategy.XLA_REDUCE_SCATTER,
    )
    run.add_argument("--interpret", action="store_true")

    finalize = commands.add_parser("finalize-matmul-run")
    finalize.add_argument("run_root", type=Path)
    finalize.add_argument("--search-root", type=Path)

    verify_bundle = commands.add_parser("verify-matmul-bundle")
    verify_bundle.add_argument("run_root", type=Path)

    confirm_matmul_collective = commands.add_parser("confirm-matmul-collective")
    confirm_matmul_collective.add_argument("--output-dir", required=True, type=Path)
    confirm_matmul_collective.add_argument("--diagnostic-root", required=True, type=Path)
    confirm_matmul_collective.add_argument(
        "--diagnostic-archive",
        required=True,
        type=Path,
    )
    confirm_matmul_collective.add_argument("--contract", required=True, type=Path)

    verify_matmul_collective = commands.add_parser("verify-matmul-collective-confirmation")
    verify_matmul_collective.add_argument("run_root", type=Path)
    verify_matmul_collective.add_argument("--contract", required=True, type=Path)

    finalize_matmul_collective = commands.add_parser("finalize-matmul-collective-confirmation")
    finalize_matmul_collective.add_argument("run_root", type=Path)
    finalize_matmul_collective.add_argument("--contract", required=True, type=Path)

    predict_matmul_collective = commands.add_parser("evaluate-matmul-collective-repeat-prediction")
    predict_matmul_collective.add_argument("--diagnostic-root", required=True, type=Path)
    predict_matmul_collective.add_argument("--diagnostic-archive", required=True, type=Path)
    predict_matmul_collective.add_argument("--confirmation-root", required=True, type=Path)
    predict_matmul_collective.add_argument("--confirmation-archive", required=True, type=Path)
    predict_matmul_collective.add_argument("--confirmation-contract", required=True, type=Path)
    predict_matmul_collective.add_argument("--contract", required=True, type=Path)
    predict_matmul_collective.add_argument("--output", required=True, type=Path)

    verify_matmul_prediction = commands.add_parser("verify-matmul-collective-repeat-prediction")
    verify_matmul_prediction.add_argument("report", type=Path)
    verify_matmul_prediction.add_argument("--diagnostic-root", required=True, type=Path)
    verify_matmul_prediction.add_argument("--diagnostic-archive", required=True, type=Path)
    verify_matmul_prediction.add_argument("--confirmation-root", required=True, type=Path)
    verify_matmul_prediction.add_argument("--confirmation-archive", required=True, type=Path)
    verify_matmul_prediction.add_argument("--confirmation-contract", required=True, type=Path)
    verify_matmul_prediction.add_argument("--contract", required=True, type=Path)

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

    verify_rpa_surface = commands.add_parser("verify-inkling-sharded-rpa-surface")
    verify_rpa_surface.add_argument("run_root", type=Path)
    verify_rpa_surface.add_argument("--contract", type=Path, required=True)

    attest_rpa_surface = commands.add_parser("attest-inkling-sharded-rpa-surface")
    attest_rpa_surface.add_argument("archive", type=Path)
    attest_rpa_surface.add_argument("--contract", type=Path, required=True)
    attest_rpa_surface.add_argument("--output", type=Path, required=True)

    verify_rpa_attestation = commands.add_parser("verify-inkling-sharded-rpa-attestation")
    verify_rpa_attestation.add_argument("attestation", type=Path)
    verify_rpa_attestation.add_argument("--archive", type=Path, required=True)
    verify_rpa_attestation.add_argument("--contract", type=Path, required=True)

    verify_rpa_donation = commands.add_parser("verify-inkling-rpa-donation-confirmation")
    verify_rpa_donation.add_argument("run_root", type=Path)
    verify_rpa_donation.add_argument("--contract", type=Path, required=True)

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

    confirm_seqax_weight_placement = commands.add_parser("confirm-seqax-weight-placement")
    confirm_seqax_weight_placement.add_argument("--contract", required=True, type=Path)
    confirm_seqax_weight_placement.add_argument("--output-dir", required=True, type=Path)

    verify_seqax_weight_confirmation = commands.add_parser(
        "verify-seqax-weight-placement-confirmation"
    )
    verify_seqax_weight_confirmation.add_argument("run_root", type=Path)
    verify_seqax_weight_confirmation.add_argument("--contract", required=True, type=Path)

    validate_seqax_bf16 = commands.add_parser("validate-seqax-bf16-forward")
    validate_seqax_bf16.add_argument("--contract", required=True, type=Path)
    validate_seqax_bf16.add_argument("--output-dir", required=True, type=Path)

    verify_seqax_bf16 = commands.add_parser("verify-seqax-bf16-forward")
    verify_seqax_bf16.add_argument("run_root", type=Path)
    verify_seqax_bf16.add_argument("--contract", required=True, type=Path)

    attest_seqax_bf16 = commands.add_parser("attest-seqax-bf16-forward-relocation")
    attest_seqax_bf16.add_argument("archive", type=Path)
    attest_seqax_bf16.add_argument("--contract", required=True, type=Path)
    attest_seqax_bf16.add_argument("--output", required=True, type=Path)

    verify_seqax_bf16_attestation = commands.add_parser("verify-seqax-bf16-forward-relocation")
    verify_seqax_bf16_attestation.add_argument("attestation", type=Path)
    verify_seqax_bf16_attestation.add_argument("archive", type=Path)
    verify_seqax_bf16_attestation.add_argument("--contract", required=True, type=Path)

    diagnose_seqax_weight_placement = commands.add_parser("diagnose-seqax-weight-placement")
    diagnose_seqax_weight_placement.add_argument("--search-root", required=True, type=Path)
    diagnose_seqax_weight_placement.add_argument("--search-contract", required=True, type=Path)
    diagnose_seqax_weight_placement.add_argument("--contract", required=True, type=Path)
    diagnose_seqax_weight_placement.add_argument("--output-dir", required=True, type=Path)

    verify_seqax_weight_placement_diagnostic = commands.add_parser(
        "verify-seqax-weight-placement-diagnostic"
    )
    verify_seqax_weight_placement_diagnostic.add_argument("run_root", type=Path)
    verify_seqax_weight_placement_diagnostic.add_argument(
        "--search-contract", required=True, type=Path
    )
    verify_seqax_weight_placement_diagnostic.add_argument("--contract", required=True, type=Path)

    probe_seqax_weight_memory = commands.add_parser("probe-seqax-weight-placement-memory")
    probe_seqax_weight_memory.add_argument(
        "--candidate",
        required=True,
        choices=tuple(SeqaxWeightPlacementName),
    )

    capture_residual_profile_hlo = commands.add_parser("capture-seqax-residual-profile-hlo")
    capture_residual_profile_hlo.add_argument("--contract", required=True, type=Path)

    run_residual_profile = commands.add_parser("run-seqax-residual-profile")
    run_residual_profile.add_argument("--contract", required=True, type=Path)
    run_residual_profile.add_argument("--output-dir", required=True, type=Path)

    verify_residual_profile = commands.add_parser("verify-seqax-residual-profile")
    verify_residual_profile.add_argument("run_root", type=Path)
    verify_residual_profile.add_argument("--contract", required=True, type=Path)

    run_residual_confirmation = commands.add_parser("run-seqax-residual-confirmation")
    run_residual_confirmation.add_argument("--contract", required=True, type=Path)
    run_residual_confirmation.add_argument("--output-dir", required=True, type=Path)

    verify_residual_confirmation = commands.add_parser("verify-seqax-residual-confirmation")
    verify_residual_confirmation.add_argument("run_root", type=Path)
    verify_residual_confirmation.add_argument("--contract", required=True, type=Path)

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

    calibrate_seqax_cost = commands.add_parser("calibrate-seqax-cost")
    calibrate_seqax_cost.add_argument("profile_root", type=Path)
    calibrate_seqax_cost.add_argument("--contract", required=True, type=Path)
    calibrate_seqax_cost.add_argument("--output", required=True, type=Path)

    verify_seqax_cost = commands.add_parser("verify-seqax-cost-calibration")
    verify_seqax_cost.add_argument("report", type=Path)
    verify_seqax_cost.add_argument("--profile-root", required=True, type=Path)
    verify_seqax_cost.add_argument("--contract", required=True, type=Path)

    search = commands.add_parser("search-matmul")
    search.add_argument("contract", type=Path)
    search.add_argument("--output-dir", required=True, type=Path)
    search.add_argument("--interpret", action="store_true")
    return parser


def _parse_schedule(path: Path) -> ModuleOp:
    return parse_tpu_cake_module(path.read_text(), name=str(path))


def _verify_schedule(path: Path) -> int:
    module = _parse_schedule(path)
    module.verify()
    print(f"SCHEDULE_ACCEPTED path={path}")
    return 0


def _estimate_physical_cost(schedule_path: Path, output: Path) -> int:
    module = _parse_schedule(schedule_path)
    report = write_physical_kernel_report(
        output,
        module=module,
        hardware=tpu7x_tensorcore_rates(),
    )
    print(
        "PHYSICAL_COST_DERIVED "
        f"schedule_sha256={report.physical_schedule_sha256} "
        f"mxu_regions={len(report.mxu_regions)} "
        f"limiting_priced_resource={report.predicted_limiting_priced_resource}"
    )
    return 0


def _verify_physical_cost(report_path: Path, schedule_path: Path) -> int:
    module = _parse_schedule(schedule_path)
    report = PhysicalKernelResourceReport.model_validate_json(report_path.read_text())
    validate_physical_kernel_report(
        report,
        module=module,
        hardware=tpu7x_tensorcore_rates(),
    )
    print(
        "PHYSICAL_COST_REPLAYED "
        f"schedule_sha256={report.physical_schedule_sha256} "
        f"mxu_regions={len(report.mxu_regions)} "
        f"limiting_priced_resource={report.predicted_limiting_priced_resource}"
    )
    return 0


def _load_physical_latency_calibration(
    calibration_path: Path,
) -> PhysicalCollectiveLatencyCalibration:
    calibration = PhysicalCollectiveLatencyCalibration.model_validate_json(
        calibration_path.read_text()
    )
    if calibration != tpu7x_collective_latency_calibration():
        raise ValueError("PHYSICAL_COLLECTIVE_LATENCY_EXTERNAL_CONTRACT_MISMATCH")
    return calibration


def _estimate_physical_latency(
    schedule_path: Path,
    calibration_path: Path,
    output: Path,
) -> int:
    module = _parse_schedule(schedule_path)
    resource_report = analyze_physical_kernel(
        module,
        hardware=tpu7x_tensorcore_rates(),
    )
    calibration = _load_physical_latency_calibration(calibration_path)
    report = write_physical_kernel_latency_report(
        output,
        module=module,
        resource_report=resource_report,
        calibration=calibration,
    )
    print(
        "PHYSICAL_LATENCY_DERIVED "
        f"schedule_sha256={report.physical_schedule_sha256} "
        f"calibration_id={report.calibration_id} "
        f"collectives={len(report.operations)} "
        f"collective_serial_ns={report.collective_measured_serial_scenario_ns}"
    )
    return 0


def _verify_physical_latency(
    report_path: Path,
    schedule_path: Path,
    calibration_path: Path,
) -> int:
    module = _parse_schedule(schedule_path)
    resource_report = analyze_physical_kernel(
        module,
        hardware=tpu7x_tensorcore_rates(),
    )
    calibration = _load_physical_latency_calibration(calibration_path)
    report = PhysicalKernelLatencyReport.model_validate_json(report_path.read_text())
    validate_physical_kernel_latency_report(
        report,
        module=module,
        resource_report=resource_report,
        calibration=calibration,
    )
    print(
        "PHYSICAL_LATENCY_REPLAYED "
        f"schedule_sha256={report.physical_schedule_sha256} "
        f"calibration_id={report.calibration_id} "
        f"collectives={len(report.operations)} "
        f"collective_serial_ns={report.collective_measured_serial_scenario_ns}"
    )
    return 0


def _compare_physical_fusion(contract_path: Path, output: Path) -> int:
    contract = SeqaxSiluMultiplyFusionContract.model_validate_json(contract_path.read_text())
    report = write_seqax_silu_multiply_fusion_report(
        output,
        contract=contract,
    )
    print(
        "SEQAX_SILU_FUSION_COMPARED "
        f"contract_id={report.contract_id} "
        f"scenarios={len(report.scenarios)} "
        f"measured_winner={report.measured_performance_winner}"
    )
    return 0


def _verify_physical_fusion(
    report_path: Path,
    contract_path: Path,
) -> int:
    report = SeqaxSiluMultiplyFusionReport.model_validate_json(report_path.read_text())
    contract = SeqaxSiluMultiplyFusionContract.model_validate_json(contract_path.read_text())
    validate_seqax_silu_multiply_fusion_report(
        report,
        contract=contract,
    )
    print(
        "SEQAX_SILU_FUSION_REPLAYED "
        f"contract_id={report.contract_id} "
        f"scenarios={len(report.scenarios)} "
        f"measured_winner={report.measured_performance_winner}"
    )
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


def _capture_inkling_decode_profile_request(args: argparse.Namespace) -> int:
    contract = InklingDecodeProfileContract.model_validate_json(args.contract.read_text())
    record = capture_inkling_decode_profile_request(
        url=args.url,
        output_path=args.output,
        profile_root=args.profile_root,
        prompt_cases_path=args.prompt_cases,
        inkling_repo=args.inkling_repo,
        contract=contract,
    )
    print(
        "INKLING_DECODE_PROFILE_CAPTURED "
        f"contract_id={contract.contract_id} "
        f"requests={len(record.request.rid)} "
        "profile_stop_after_minimum_completion_tokens="
        f"{record.profile_stop_after_minimum_completion_tokens} "
        f"profile_session={record.provenance.profile_directory_name}"
    )
    return 0


def _inspect_inkling_decode_profile(args: argparse.Namespace, *, require_pinned: bool) -> int:
    contract = InklingDecodeProfileContract.model_validate_json(args.contract.read_text())
    if require_pinned:
        assessment = validate_inkling_decode_profile(
            capture_root=args.capture,
            request_path=args.request,
            prompt_cases_path=args.prompt_cases,
            contract=contract,
        )
    else:
        assessment = inspect_inkling_decode_profile(
            capture_root=args.capture,
            request_path=args.request,
            prompt_cases_path=args.prompt_cases,
            contract=contract,
        )
    if args.output is not None:
        write_inkling_decode_profile_assessment(args.output, assessment)
    if assessment.accepted:
        verdict = "VERIFIED"
    elif {finding.code for finding in assessment.findings} == {"HLO_IDENTITIES_PENDING"}:
        verdict = "PENDING"
    else:
        verdict = "REJECTED"
    print(
        f"INKLING_DECODE_PROFILE_{verdict} "
        f"contract_id={contract.contract_id} "
        f"programs={len(assessment.required_programs)}"
    )
    for finding in assessment.capture.findings + assessment.findings:
        print(f"{finding.severity.value.upper()} {finding.code}: {finding.message}")
    if require_pinned:
        return 0 if assessment.accepted else 1
    return 0 if verdict in {"VERIFIED", "PENDING"} else 1


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
    elif args.command == "estimate-physical-cost":
        code = _estimate_physical_cost(args.schedule, args.output)
    elif args.command == "verify-physical-cost":
        code = _verify_physical_cost(args.report, args.schedule)
    elif args.command == "estimate-physical-latency":
        code = _estimate_physical_latency(
            args.schedule,
            args.calibration,
            args.output,
        )
    elif args.command == "verify-physical-latency":
        code = _verify_physical_latency(
            args.report,
            args.schedule,
            args.calibration,
        )
    elif args.command == "compare-seqax-silu-fusion":
        code = _compare_physical_fusion(args.contract, args.output)
    elif args.command == "verify-seqax-silu-fusion":
        code = _verify_physical_fusion(args.report, args.contract)
    elif args.command == "inspect-profile":
        code = _inspect_profile(args.capture, args.contract, args.output)
    elif args.command == "capture-inkling-decode-profile-request":
        code = _capture_inkling_decode_profile_request(args)
    elif args.command == "inspect-inkling-decode-profile":
        code = _inspect_inkling_decode_profile(args, require_pinned=False)
    elif args.command == "verify-inkling-decode-profile":
        code = _inspect_inkling_decode_profile(args, require_pinned=True)
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
            collective_strategy=MatmulCollectiveStrategy(args.collective_strategy),
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
    elif args.command == "confirm-matmul-collective":
        contract = MatmulCollectiveConfirmationContract.model_validate_json(
            args.contract.read_text()
        )
        result = run_matmul_collective_confirmation(
            args.output_dir,
            args.diagnostic_root,
            args.diagnostic_archive,
            contract,
        )
        print(
            "MATMUL_COLLECTIVE_CONFIRMATION_ACCEPTED "
            f"confirmation_id={result.confirmation_id} "
            f"decision={result.statistics.decision} "
            f"selected_strategy={result.statistics.selected_strategy} "
            f"scope={result.claim_scope}"
        )
        code = 0
    elif args.command == "verify-matmul-collective-confirmation":
        contract = MatmulCollectiveConfirmationContract.model_validate_json(
            args.contract.read_text()
        )
        result = validate_matmul_collective_confirmation(args.run_root, contract)
        print(
            "MATMUL_COLLECTIVE_CONFIRMATION_REPLAYED "
            f"confirmation_id={result.confirmation_id} "
            f"decision={result.statistics.decision} "
            f"selected_strategy={result.statistics.selected_strategy} "
            f"scope={result.claim_scope}"
        )
        code = 0
    elif args.command == "finalize-matmul-collective-confirmation":
        contract = MatmulCollectiveConfirmationContract.model_validate_json(
            args.contract.read_text()
        )
        result = finalize_matmul_collective_confirmation(args.run_root, contract)
        print(
            "MATMUL_COLLECTIVE_CONFIRMATION_FINALIZED "
            f"confirmation_id={result.confirmation_id} "
            f"decision={result.statistics.decision} "
            f"selected_strategy={result.statistics.selected_strategy} "
            f"scope={result.claim_scope}"
        )
        code = 0
    elif args.command == "evaluate-matmul-collective-repeat-prediction":
        confirmation_contract = MatmulCollectiveConfirmationContract.model_validate_json(
            args.confirmation_contract.read_text()
        )
        contract = MatmulCollectiveRepeatPredictionContract.model_validate_json(
            args.contract.read_text()
        )
        report = write_matmul_collective_repeat_prediction(
            args.output,
            diagnostic_root=args.diagnostic_root,
            diagnostic_archive=args.diagnostic_archive,
            confirmation_root=args.confirmation_root,
            confirmation_archive=args.confirmation_archive,
            confirmation_contract=confirmation_contract,
            contract=contract,
        )
        print(
            "MATMUL_COLLECTIVE_REPEAT_PREDICTION_EVALUATED "
            f"contract_id={report.contract_id} "
            f"ranking_agrees={report.strategy_ranking_agrees} "
            f"scope={report.status}"
        )
        code = 0
    elif args.command == "verify-matmul-collective-repeat-prediction":
        confirmation_contract = MatmulCollectiveConfirmationContract.model_validate_json(
            args.confirmation_contract.read_text()
        )
        contract = MatmulCollectiveRepeatPredictionContract.model_validate_json(
            args.contract.read_text()
        )
        report = MatmulCollectiveRepeatPredictionReport.model_validate_json(args.report.read_text())
        validate_matmul_collective_repeat_prediction(
            report,
            diagnostic_root=args.diagnostic_root,
            diagnostic_archive=args.diagnostic_archive,
            confirmation_root=args.confirmation_root,
            confirmation_archive=args.confirmation_archive,
            confirmation_contract=confirmation_contract,
            contract=contract,
        )
        print(
            "MATMUL_COLLECTIVE_REPEAT_PREDICTION_REPLAYED "
            f"contract_id={report.contract_id} "
            f"ranking_agrees={report.strategy_ranking_agrees} "
            f"scope={report.status}"
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
    elif args.command == "verify-inkling-sharded-rpa-surface":
        contract = InklingShardedRpaSurfaceContract.model_validate_json(args.contract.read_text())
        result = validate_inkling_sharded_rpa_surface(args.run_root, contract)
        print(
            "INKLING_SHARDED_RPA_SURFACE_PRODUCER_ACCEPTED "
            f"surface_id={result.surface_id} observations={len(result.correctness)} "
            f"rounds={len(result.rounds)} scope={result.claim_scope}"
        )
        code = 0
    elif args.command == "attest-inkling-sharded-rpa-surface":
        contract = InklingShardedRpaSurfaceContract.model_validate_json(args.contract.read_text())
        attestation = write_inkling_sharded_rpa_relocation_attestation(
            args.output,
            archive=args.archive,
            contract=contract,
        )
        print(
            "INKLING_SHARDED_RPA_SURFACE_PORTABLE_ACCEPTED "
            f"surface_id={attestation.surface_id} observations={len(attestation.observations)} "
            f"scope={attestation.claim_scope}"
        )
        code = 0
    elif args.command == "verify-inkling-sharded-rpa-attestation":
        contract = InklingShardedRpaSurfaceContract.model_validate_json(args.contract.read_text())
        attestation = validate_inkling_sharded_rpa_relocation_attestation(
            args.attestation,
            archive=args.archive,
            contract=contract,
        )
        print(
            "INKLING_SHARDED_RPA_ATTESTATION_VERIFIED "
            f"surface_id={attestation.surface_id} observations={len(attestation.observations)} "
            f"scope={attestation.claim_scope}"
        )
        code = 0
    elif args.command == "verify-inkling-rpa-donation-confirmation":
        contract = InklingRpaDonationConfirmationContract.model_validate_json(
            args.contract.read_text()
        )
        result = validate_inkling_rpa_donation_confirmation(args.run_root, contract)
        print(
            "INKLING_RPA_DONATION_CONFIRMATION_VERIFIED "
            f"confirmation_id={result.confirmation_id} "
            f"winner={result.winner or 'none'} accepted={str(result.accepted).lower()} "
            f"scope={result.claim_scope}"
        )
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
    elif args.command == "confirm-seqax-weight-placement":
        contract = SeqaxWeightConfirmationContract.model_validate_json(args.contract.read_text())
        result = run_seqax_weight_confirmation(args.output_dir, contract)
        print(result.model_dump_json(indent=2))
        code = 0
    elif args.command == "verify-seqax-weight-placement-confirmation":
        contract = SeqaxWeightConfirmationContract.model_validate_json(args.contract.read_text())
        result = validate_seqax_weight_confirmation(args.run_root, contract)
        print(
            "SEQAX_WEIGHT_PLACEMENT_CONFIRMATION_ACCEPTED "
            f"winner={result.winner or 'none'} rounds={result.statistics.round_count} "
            f"confidence={result.statistics.confidence_level} "
            f"scope={result.correctness_scope}"
        )
        code = 0
    elif args.command == "validate-seqax-bf16-forward":
        contract = SeqaxBf16ValidationContract.model_validate_json(args.contract.read_text())
        result = run_seqax_bf16_validation(args.output_dir, contract)
        print(result.model_dump_json(indent=2))
        code = 0 if result.producer_passed else 1
    elif args.command == "verify-seqax-bf16-forward":
        contract = SeqaxBf16ValidationContract.model_validate_json(args.contract.read_text())
        result = validate_seqax_bf16_validation(args.run_root, contract)
        print(
            "SEQAX_BF16_FORWARD_PRODUCER_VALIDATED "
            f"scenarios={len(result.plans)} observations={len(result.observations)} "
            f"discriminators={len(result.discriminators)} scope={result.claim_scope}"
        )
        code = 0
    elif args.command == "attest-seqax-bf16-forward-relocation":
        contract = SeqaxBf16ValidationContract.model_validate_json(args.contract.read_text())
        attestation = write_seqax_bf16_relocation_attestation(
            args.output,
            archive=args.archive,
            contract=contract,
        )
        print(
            "SEQAX_BF16_FORWARD_PORTABLE_ACCEPTED "
            f"observations={len(attestation.observations)} "
            f"scope={attestation.claim_scope}"
        )
        code = 0
    elif args.command == "verify-seqax-bf16-forward-relocation":
        contract = SeqaxBf16ValidationContract.model_validate_json(args.contract.read_text())
        attestation = validate_seqax_bf16_relocation_attestation(
            args.attestation,
            archive=args.archive,
            contract=contract,
        )
        print(
            "SEQAX_BF16_FORWARD_RELOCATION_REPLAYED "
            f"observations={len(attestation.observations)} "
            f"scope={attestation.claim_scope}"
        )
        code = 0
    elif args.command == "diagnose-seqax-weight-placement":
        search_contract = SeqaxWeightPlacementContract.model_validate_json(
            args.search_contract.read_text()
        )
        contract = SeqaxWeightPlacementDiagnosticContract.model_validate_json(
            args.contract.read_text()
        )
        result = run_seqax_weight_placement_diagnostic(
            args.output_dir,
            args.search_root,
            search_contract,
            contract,
        )
        print(result.model_dump_json(indent=2))
        code = 0
    elif args.command == "verify-seqax-weight-placement-diagnostic":
        search_contract = SeqaxWeightPlacementContract.model_validate_json(
            args.search_contract.read_text()
        )
        contract = SeqaxWeightPlacementDiagnosticContract.model_validate_json(
            args.contract.read_text()
        )
        result = validate_seqax_weight_placement_diagnostic(
            args.run_root,
            search_contract,
            contract,
        )
        print(
            "SEQAX_WEIGHT_PLACEMENT_DIAGNOSTIC_ACCEPTED "
            f"candidates={len(result.candidates)} scope={result.correctness_scope}"
        )
        code = 0
    elif args.command == "probe-seqax-weight-placement-memory":
        observation = probe_weight_placement_memory(SeqaxWeightPlacementName(args.candidate))
        print("SEQAX_WEIGHT_PLACEMENT_MEMORY_JSON=" + observation.model_dump_json())
        code = 0
    elif args.command == "capture-seqax-residual-profile-hlo":
        contract = SeqaxResidualProfileContract.model_validate_json(args.contract.read_text())
        identities = capture_seqax_residual_profile_hlo_identities(contract)
        print(json.dumps(identities, indent=2, sort_keys=True))
        code = 0
    elif args.command == "run-seqax-residual-profile":
        contract = SeqaxResidualProfileContract.model_validate_json(args.contract.read_text())
        result = run_seqax_residual_profile(args.output_dir, contract)
        print(result.model_dump_json(indent=2))
        code = 0
    elif args.command == "verify-seqax-residual-profile":
        contract = SeqaxResidualProfileContract.model_validate_json(args.contract.read_text())
        result = validate_seqax_residual_profile(args.run_root, contract)
        print(
            "SEQAX_RESIDUAL_PROFILE_DIAGNOSTIC_PASSED "
            f"candidates={len(result.candidates)} accepted=false "
            f"scope={result.correctness_scope}"
        )
        code = 0
    elif args.command == "run-seqax-residual-confirmation":
        contract = SeqaxResidualConfirmationContract.model_validate_json(args.contract.read_text())
        result = run_seqax_residual_confirmation(args.output_dir, contract)
        print(result.model_dump_json(indent=2))
        code = 0
    elif args.command == "verify-seqax-residual-confirmation":
        contract = SeqaxResidualConfirmationContract.model_validate_json(args.contract.read_text())
        result = validate_seqax_residual_confirmation(args.run_root, contract)
        print(
            "SEQAX_RESIDUAL_CONFIRMATION_EVIDENCE_PASSED "
            f"winner={result.winner} scope={result.claim_scope}"
        )
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
    elif args.command == "calibrate-seqax-cost":
        contract = SeqaxCostCalibrationContract.model_validate_json(args.contract.read_text())
        report = write_seqax_cost_calibration(
            args.output,
            profile_root=args.profile_root,
            contract=contract,
        )
        print(
            "SEQAX_COST_CALIBRATION_DERIVED "
            f"status={report.status} points={len(report.points)} "
            f"fit_error={report.maximum_in_surface_relative_error} "
            f"predictive_validation={report.predictive_validation}"
        )
        code = 0
    elif args.command == "verify-seqax-cost-calibration":
        contract = SeqaxCostCalibrationContract.model_validate_json(args.contract.read_text())
        report = SeqaxCostCalibrationReport.model_validate_json(args.report.read_text())
        validate_seqax_cost_calibration(
            report,
            profile_root=args.profile_root,
            contract=contract,
        )
        print(
            "SEQAX_COST_CALIBRATION_REPLAYED "
            f"status={report.status} points={len(report.points)} "
            f"fit_error={report.maximum_in_surface_relative_error} "
            f"predictive_validation={report.predictive_validation}"
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
