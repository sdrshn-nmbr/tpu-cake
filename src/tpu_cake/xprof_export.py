from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from xprof.convert import raw_to_tool_data


class XProfExport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str = Field(min_length=1)
    mime_type: str = Field(min_length=1)
    output: Path
    size_bytes: int = Field(ge=0)


class XProfExportManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    xplane: Path
    available_tools: tuple[str, ...]
    exports: tuple[XProfExport, ...]


DEFAULT_TOOLS = (
    "overview_page",
    "framework_op_stats",
    "op_profile",
    "hlo_stats",
    "roofline_model",
    "perf_counters",
)


def export_xprof_capture(
    capture_root: Path,
    output_root: Path,
    *,
    tools: tuple[str, ...] = DEFAULT_TOOLS,
) -> XProfExportManifest:
    xplanes = sorted(capture_root.rglob("*.xplane.pb"))
    if len(xplanes) != 1:
        raise ValueError(f"XPROF_EXPORT_REQUIRES_ONE_XPLANE observed={xplanes}")
    xplane = xplanes[0].resolve()
    paths = [str(xplane)]
    available = tuple(sorted(raw_to_tool_data.xspace_to_tool_names(paths)))
    available_set = set(available)
    output_root.mkdir(parents=True, exist_ok=True)
    exports: list[XProfExport] = []
    for tool in tools:
        if tool not in available_set:
            continue
        payload, mime_type = raw_to_tool_data.xspace_to_tool_data(paths, tool, {})
        suffix = ".json" if mime_type == "application/json" else ".bin"
        output = output_root / f"{tool}{suffix}"
        encoded = payload if isinstance(payload, bytes) else str(payload).encode()
        output.write_bytes(encoded)
        exports.append(
            XProfExport(
                tool=tool,
                mime_type=mime_type,
                output=output.resolve(),
                size_bytes=len(encoded),
            )
        )
    manifest = XProfExportManifest(
        xplane=xplane,
        available_tools=available,
        exports=tuple(exports),
    )
    (output_root / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    return manifest
