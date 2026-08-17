from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin
from xdsl.parser import Parser

from tpu_cake.contracts import ProfileExpectation
from tpu_cake.dialects.tpu_schedule import TPUSchedule
from tpu_cake.frontend import canonical_module_text
from tpu_cake.workloads import (
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
    return parser


def _verify_schedule(path: Path) -> int:
    context = Context()
    context.load_dialect(Builtin)
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
        f"counter_rates_derivable={assessment.capture.counters.rates_derivable}"
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
    rendered = experiment.model_dump_json(indent=2) + "\n"
    if output is None:
        print(rendered, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
        print(f"EXPERIMENT_WRITTEN workload={workload} id={experiment.experiment_id} path={output}")
    return 0


def main() -> None:
    args = _parser().parse_args()
    if args.command == "verify-schedule":
        code = _verify_schedule(args.schedule)
    elif args.command == "inspect-profile":
        code = _inspect_profile(args.capture, args.contract, args.output)
    elif args.command == "render-workload":
        code = _render_workload(args.workload, args.output)
    else:
        code = _experiment(args.workload, args.output)
    sys.exit(code)


if __name__ == "__main__":
    main()
