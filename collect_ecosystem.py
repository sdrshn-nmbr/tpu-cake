from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path


ROOT = Path(__file__).parent / "ecosystem"

PAPERS = {
    "cake-2608.12629": "https://arxiv.org/pdf/2608.12629",
    "ragged-paged-attention-2604.15464": "https://arxiv.org/pdf/2604.15464",
}

CODE_ARCHIVES = {
    "accelerator-agents-c4f3b17": "https://github.com/AI-Hypercomputer/accelerator-agents/archive/c4f3b17ba3e034d7f148f47c3e6cbe3905f5e386.zip",
    "tokamax-5802a3d": "https://github.com/openxla/tokamax/archive/5802a3d225c661be603cefbea9843c7e928f6dd0.zip",
}

PINNED_SOURCES = {
    "jax/ragged_paged_attention.py": "https://raw.githubusercontent.com/jax-ml/jax/a66f606f34ea176a1b75d0428bbbb2930303665c/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py",
    "jax/ragged_paged_attention_tuned_block_sizes.py": "https://raw.githubusercontent.com/jax-ml/jax/a66f606f34ea176a1b75d0428bbbb2930303665c/jax/experimental/pallas/ops/tpu/ragged_paged_attention/tuned_block_sizes.py",
    "jax/splash_attention_kernel.py": "https://raw.githubusercontent.com/jax-ml/jax/a66f606f34ea176a1b75d0428bbbb2930303665c/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py",
    "jax/interpret_params.py": "https://raw.githubusercontent.com/jax-ml/jax/a66f606f34ea176a1b75d0428bbbb2930303665c/jax/_src/pallas/mosaic/interpret/params.py",
    "jax/race_detection_state.py": "https://raw.githubusercontent.com/jax-ml/jax/a66f606f34ea176a1b75d0428bbbb2930303665c/jax/_src/pallas/mosaic/interpret/race_detection_state.py",
    "tokamax/ragged_dot_tpu_v2.py": "https://raw.githubusercontent.com/openxla/tokamax/5802a3d225c661be603cefbea9843c7e928f6dd0/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py",
    "tokamax/ragged_gather_reduce_tpu.py": "https://raw.githubusercontent.com/openxla/tokamax/5802a3d225c661be603cefbea9843c7e928f6dd0/tokamax/_src/ops/ragged_gather_reduce/pallas_mosaic_tpu.py",
    "tokamax/autotuner.py": "https://raw.githubusercontent.com/openxla/tokamax/5802a3d225c661be603cefbea9843c7e928f6dd0/tokamax/_src/autotuning/autotuner.py",
    "tokamax/autotuning.md": "https://raw.githubusercontent.com/openxla/tokamax/5802a3d225c661be603cefbea9843c7e928f6dd0/docs/autotuning.md",
    "tokamax/tpu7x_ragged_dot.json": "https://raw.githubusercontent.com/openxla/tokamax/5802a3d225c661be603cefbea9843c7e928f6dd0/tokamax/data/autotuning/tpu7x/pallas_mosaic_tpu_ragged_dot.json",
}

WEB_REFERENCES = {
    "jax-pallas-design": "https://docs.jax.dev/en/latest/pallas/design/design.html",
    "jax-pallas-tpu-details": "https://docs.jax.dev/en/latest/pallas/tpu/details.html",
    "jax-pallas-interpret": "https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.tpu.InterpretParams.html",
    "jax-profiling": "https://docs.jax.dev/en/latest/profiling.html",
    "xprof-kernel-profiling": "https://openxla.org/xprof/kernel-profiling",
    "xprof-custom-call-profiling": "https://openxla.org/xprof/custom_call_profiling",
    "accelerator-agents": "https://github.com/AI-Hypercomputer/accelerator-agents",
    "tokamax": "https://github.com/openxla/tokamax",
    "accelerator-microbenchmarks": "https://github.com/AI-Hypercomputer/accelerator-microbenchmarks",
    "cloud-diagnostics-xprof": "https://github.com/AI-Hypercomputer/cloud-diagnostics-xprof",
}


def fetch(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "inkle-research-collector/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    records: list[dict[str, str | int]] = []
    for name, url in PAPERS.items():
        path = ROOT / "papers" / f"{name}.pdf"
        fetch(url, path)
        records.append({"kind": "paper", "source": url, "path": str(path.relative_to(ROOT))})
        text_path = ROOT / "text" / "papers" / f"{name}.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pdftotext", "-layout", str(path), str(text_path)], check=True)
        records.append({"kind": "paper-text", "source": url, "path": str(text_path.relative_to(ROOT))})
    for name, url in CODE_ARCHIVES.items():
        path = ROOT / "code" / f"{name}.zip"
        fetch(url, path)
        records.append({"kind": "code-archive", "source": url, "path": str(path.relative_to(ROOT))})
    for relative_path, url in PINNED_SOURCES.items():
        path = ROOT / "source" / relative_path
        fetch(url, path)
        records.append({"kind": "pinned-source", "source": url, "path": str(path.relative_to(ROOT))})
    for name, url in WEB_REFERENCES.items():
        records.append({"kind": "web-reference", "source": url, "path": ""})
    for record in records:
        if not record["path"]:
            continue
        path = ROOT / str(record["path"])
        record["bytes"] = path.stat().st_size
        record["sha256"] = sha256(path)
    manifest = {"schema_version": 1, "records": sorted(records, key=lambda item: (str(item["kind"]), str(item["source"])))}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"collected {len(records)} ecosystem artifacts under {ROOT}")


if __name__ == "__main__":
    main()
