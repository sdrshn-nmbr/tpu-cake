from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from decimal import Decimal, localcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from tpu_cake.xprof_export import export_xprof_capture

_PRODUCER_PATH = Path(__file__).with_name("inkling_decode_operation_selection.py")
_VERIFIER_PATH = Path(__file__)
_REPOSITORY_ROOT = _VERIFIER_PATH.resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _share(numerator: int, denominator: int) -> Decimal:
    with localcontext() as context:
        context.prec = 50
        return Decimal(numerator) / Decimal(denominator)


def _children(node: dict[str, Any], error: str) -> list[dict[str, Any]]:
    children = node.get("children")
    if not isinstance(children, list) or any(not isinstance(child, dict) for child in children):
        raise ValueError(error)
    return children


def _metrics(node: dict[str, Any], error: str) -> tuple[int, int]:
    metrics = node.get("metrics")
    if not isinstance(metrics, dict):
        raise TypeError(error)
    raw_time = metrics.get("rawTime")
    occurrences = metrics.get("occurrences")
    if (
        isinstance(raw_time, bool)
        or not isinstance(raw_time, int)
        or raw_time <= 0
        or isinstance(occurrences, bool)
        or not isinstance(occurrences, int)
        or occurrences <= 0
    ):
        raise ValueError(error)
    return raw_time, occurrences


def _custom_family(name: str) -> str:
    match = re.fullmatch(
        r"(?P<family>[^.]+)\.\d+(?:\.[^. ]+)*(?: and its duplicate\(s\))?",
        name,
    )
    if match is None:
        raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_CUSTOM_NAME_INVALID")
    return match.group("family")


def _ranking(path: Path, main_name: str) -> tuple[int, int, list[dict[str, object]]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("deviceType") != "TPU":
        raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_OP_PROFILE_INVALID")
    root = payload.get("byProgram")
    if not isinstance(root, dict):
        raise TypeError("INKLING_OPERATION_SELECTION_INDEPENDENT_PROGRAM_ROOT_INVALID")
    matches = [
        child
        for child in _children(root, "INKLING_OPERATION_SELECTION_INDEPENDENT_PROGRAMS_INVALID")
        if child.get("name") == main_name
    ]
    if len(matches) != 1:
        raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_MAIN_INVENTORY_MISMATCH")
    main = matches[0]
    main_metrics = main.get("metrics")
    if not isinstance(main_metrics, dict):
        raise TypeError("INKLING_OPERATION_SELECTION_INDEPENDENT_MAIN_METRICS_INVALID")
    main_time = main_metrics.get("rawTime")
    if isinstance(main_time, bool) or not isinstance(main_time, int) or main_time <= 0:
        raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_MAIN_TIME_INVALID")

    partitions: dict[tuple[str, str], dict[str, object]] = {}
    custom_count = 0
    for child in _children(main, "INKLING_OPERATION_SELECTION_INDEPENDENT_CATEGORIES_INVALID"):
        name = child.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_CATEGORY_NAME_INVALID")
        raw_time, occurrences = _metrics(
            child, "INKLING_OPERATION_SELECTION_INDEPENDENT_CATEGORY_METRICS_INVALID"
        )
        if name != "custom-call":
            key = ("xprof-category", name)
            if key in partitions:
                raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_CATEGORY_DUPLICATE")
            partitions[key] = {
                "raw_time_ps": raw_time,
                "occurrences": occurrences,
                "members": [name],
            }
            continue
        custom_count += 1
        child_time = 0
        child_occurrences = 0
        for operation in _children(
            child, "INKLING_OPERATION_SELECTION_INDEPENDENT_CUSTOM_CHILDREN_INVALID"
        ):
            operation_name = operation.get("name")
            if not isinstance(operation_name, str):
                raise TypeError("INKLING_OPERATION_SELECTION_INDEPENDENT_CUSTOM_NAME_INVALID")
            operation_time, operation_occurrences = _metrics(
                operation,
                "INKLING_OPERATION_SELECTION_INDEPENDENT_CUSTOM_METRICS_INVALID",
            )
            child_time += operation_time
            child_occurrences += operation_occurrences
            key = ("custom-call-family", _custom_family(operation_name))
            partition = partitions.setdefault(
                key, {"raw_time_ps": 0, "occurrences": 0, "members": []}
            )
            partition["raw_time_ps"] = int(partition["raw_time_ps"]) + operation_time
            partition["occurrences"] = int(partition["occurrences"]) + operation_occurrences
            members = partition["members"]
            if not isinstance(members, list):
                raise TypeError("independent partition members must be a list")
            members.append(operation_name)
        if child_time != raw_time or child_occurrences != occurrences:
            raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_CUSTOM_CONTAINER_MISMATCH")
    if custom_count != 1:
        raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_CUSTOM_INVENTORY_MISMATCH")
    ordered = sorted(
        partitions.items(),
        key=lambda item: (-int(item[1]["raw_time_ps"]), f"{item[0][0]}/{item[0][1]}"),
    )
    ranking = [
        {
            "rank": rank,
            "key": f"{source}/{name}",
            "source": source,
            "raw_time_ps": int(partition["raw_time_ps"]),
            "occurrences": int(partition["occurrences"]),
            "device_op_share_of_main_program": str(
                _share(int(partition["raw_time_ps"]), main_time)
            ),
            "member_names": sorted(partition["members"]),
        }
        for rank, ((source, name), partition) in enumerate(ordered, start=1)
    ]
    attributed = sum(int(partition["raw_time_ps"]) for partition in partitions.values())
    if attributed > main_time:
        raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_MAIN_TIME_UNDERFLOW")
    return main_time, main_time - attributed, ranking


def _gviz_rows(path: Path) -> list[dict[str, object]]:
    table = json.loads(path.read_text())
    columns = [column["id"] for column in table["cols"]]
    result = []
    for row in table["rows"]:
        values = [cell.get("v") for cell in row["c"]]
        if len(values) != len(columns):
            raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_HLO_WIDTH_MISMATCH")
        result.append(dict(zip(columns, values, strict=True)))
    return result


def _candidate_claim(
    path: Path,
    *,
    program_id: str,
    prefix: str,
) -> list[dict[str, object]]:
    times: defaultdict[str, int] = defaultdict(int)
    occurrences: defaultdict[str, int] = defaultdict(int)
    rows: defaultdict[str, int] = defaultdict(int)
    intensities: defaultdict[str, list[Decimal]] = defaultdict(list)
    dma_stalls: defaultdict[str, list[Decimal]] = defaultdict(list)
    bounds: defaultdict[str, set[str]] = defaultdict(set)
    for row in _gviz_rows(path):
        name = str(row.get("hlo_op_name") or "")
        if str(row.get("program_id")) != program_id or not name.startswith(prefix):
            continue
        match = re.fullmatch(
            rf"(?P<family>{re.escape(prefix)}[^.]+)\.\d+(?: and its duplicate\(s\))?",
            name,
        )
        if match is None or row.get("category") != "custom-call":
            raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_HLO_FAMILY_INVALID")
        family = match.group("family")
        time_ps = Decimal(str(row.get("total_self_time"))) * Decimal(1_000_000)
        occurrence = Decimal(str(row.get("occurrences")))
        if (
            not time_ps.is_finite()
            or time_ps <= 0
            or time_ps != time_ps.to_integral_value()
            or occurrence <= 0
            or occurrence != occurrence.to_integral_value()
        ):
            raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_HLO_METRICS_INVALID")
        times[family] += int(time_ps)
        occurrences[family] += int(occurrence)
        rows[family] += 1
        intensity = Decimal(str(row.get("operational_intensity")))
        if not intensity.is_finite() or intensity < 0:
            raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_INTENSITY_INVALID")
        dma_stall = Decimal(str(row.get("dma_stall_percent")))
        if not dma_stall.is_finite() or dma_stall < 0:
            raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_DMA_STALL_INVALID")
        intensities[family].append(intensity)
        dma_stalls[family].append(dma_stall)
        bound = row.get("bound_by")
        if not isinstance(bound, str) or not bound:
            raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_BOUND_INVALID")
        bounds[family].add(bound)
    if not times:
        raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_CANDIDATE_MISSING")
    return [
        {
            "name": family,
            "raw_time_ps": times[family],
            "occurrences": occurrences[family],
            "hlo_rows": rows[family],
            "operational_intensity_min": str(min(intensities[family])),
            "operational_intensity_max": str(max(intensities[family])),
            "dma_stall_percent_min": str(min(dma_stalls[family])),
            "dma_stall_percent_max": str(max(dma_stalls[family])),
            "bound_by": sorted(bounds[family]),
        }
        for family in sorted(times)
    ]


def verify_report_independently(
    *,
    report_path: Path,
    capture_root: Path,
    profile_contract: object,
    selection_contract: object,
) -> None:
    report = json.loads(report_path.read_text())
    contract = selection_contract.model_dump(mode="json", exclude_computed_fields=True)
    profile = profile_contract.model_dump(mode="json", exclude_computed_fields=True)
    expected_sources = {
        "producer_source_sha256": _sha256(_PRODUCER_PATH),
        "verifier_source_sha256": _sha256(_VERIFIER_PATH),
        "uv_lock_sha256": _sha256(_REPOSITORY_ROOT / "uv.lock"),
    }
    for key, observed in expected_sources.items():
        if contract.get(key) != observed or report.get(key) != observed:
            raise ValueError(f"INKLING_OPERATION_SELECTION_INDEPENDENT_{key.upper()}_MISMATCH")
    identity_payload = dict(report)
    report_id = identity_payload.pop("report_id", None)
    expected_id = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if report_id != expected_id:
        raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_REPORT_ID_MISMATCH")

    main_prefix = profile["main_program_prefix"]
    programs = [item for item in profile["programs"] if item["name_prefix"] == main_prefix]
    if len(programs) != 1:
        raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_PROFILE_MAIN_MISMATCH")
    if (
        not str(report.get("main_program_name", "")).startswith(f"{main_prefix}(")
        or report.get("main_program_semantic_hlo_sha256") != programs[0]["semantic_hlo_sha256"]
    ):
        raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_MAIN_IDENTITY_MISMATCH")

    with TemporaryDirectory(prefix="tpu-cake-inkling-operation-verifier-") as temporary:
        manifest = export_xprof_capture(
            capture_root, Path(temporary), tools=("op_profile", "hlo_stats")
        )
        exports = {item.tool: item.output for item in manifest.exports}
        if set(exports) != {"op_profile", "hlo_stats"}:
            raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_EXPORT_INVENTORY_MISMATCH")
        if (
            _sha256(exports["op_profile"]) != contract["op_profile_sha256"]
            or _sha256(exports["hlo_stats"]) != contract["hlo_stats_sha256"]
        ):
            raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_EXPORT_HASH_MISMATCH")
        main_time, unattributed, ranking = _ranking(
            exports["op_profile"], str(report["main_program_name"])
        )
        candidate_families = _candidate_claim(
            exports["hlo_stats"],
            program_id=str(report["main_program_id"]),
            prefix=str(contract["candidate_hlo_prefix"]),
        )

    candidate_time = sum(int(item["raw_time_ps"]) for item in candidate_families)
    candidate_occurrences = sum(int(item["occurrences"]) for item in candidate_families)
    candidate_rows = sum(int(item["hlo_rows"]) for item in candidate_families)
    for family in candidate_families:
        family["device_op_share_of_main_program"] = str(
            _share(int(family["raw_time_ps"]), main_time)
        )
    candidate_families = [
        {
            "name": item["name"],
            "raw_time_ps": item["raw_time_ps"],
            "occurrences": item["occurrences"],
            "hlo_rows": item["hlo_rows"],
            "device_op_share_of_main_program": item["device_op_share_of_main_program"],
            "operational_intensity_min": item["operational_intensity_min"],
            "operational_intensity_max": item["operational_intensity_max"],
            "dma_stall_percent_min": item["dma_stall_percent_min"],
            "dma_stall_percent_max": item["dma_stall_percent_max"],
            "bound_by": item["bound_by"],
        }
        for item in candidate_families
    ]

    candidate_partitions = [
        item
        for item in ranking
        if any(
            str(member).startswith(str(contract["candidate_hlo_prefix"]))
            for member in item["member_names"]
        )
    ]
    if (
        sum(int(item["raw_time_ps"]) for item in candidate_partitions) != candidate_time
        or sum(int(item["occurrences"]) for item in candidate_partitions) != candidate_occurrences
    ):
        raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_CANDIDATE_VIEW_MISMATCH")

    public_claim = {
        "main_program_raw_time_ps": main_time,
        "attributed_raw_time_ps": main_time - unattributed,
        "unattributed_raw_time_ps": unattributed,
        "unattributed_device_op_share_of_main_program": str(_share(unattributed, main_time)),
        "operation_ranking": ranking,
        "winner_partition_key": ranking[0]["key"],
        "winner_raw_time_ps": ranking[0]["raw_time_ps"],
        "winner_device_op_share_of_main_program": ranking[0]["device_op_share_of_main_program"],
        "candidate_raw_time_ps": candidate_time,
        "candidate_occurrences": candidate_occurrences,
        "candidate_hlo_rows": candidate_rows,
        "candidate_device_op_share_of_main_program": str(_share(candidate_time, main_time)),
        "candidate_kernel_families": candidate_families,
    }
    for key, expected in public_claim.items():
        if report.get(key) != expected:
            raise ValueError(f"INKLING_OPERATION_SELECTION_INDEPENDENT_{key.upper()}_MISMATCH")
    expected_claim = contract["expected_claim"]
    observed_intensities = [
        Decimal(str(item[key]))
        for item in candidate_families
        for key in ("operational_intensity_min", "operational_intensity_max")
    ]
    observed_bounds = sorted({bound for item in candidate_families for bound in item["bound_by"]})
    if (
        min(observed_intensities)
        != Decimal(str(expected_claim["candidate_operational_intensity_min"]))
        or max(observed_intensities)
        != Decimal(str(expected_claim["candidate_operational_intensity_max"]))
        or observed_bounds != contract["expected_bound_by"]
    ):
        raise ValueError("INKLING_OPERATION_SELECTION_INDEPENDENT_HLO_CLAIM_MISMATCH")
