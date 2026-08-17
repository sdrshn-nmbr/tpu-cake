from __future__ import annotations

import argparse
import hashlib
import inspect
from pathlib import Path

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    ragged_paged_attention,
)
from sgl_jax.srt.kernels.ragged_paged_attention.tuned_block_sizes import (
    get_simplified_key,
)
from sgl_jax.srt.kernels.ragged_paged_attention.tuned_block_sizes_v3 import (
    get_tuned_block_sizes_v3,
)
from sgl_jax.srt.kernels.ragged_paged_attention.util import get_dtype_packing

from tpu_cake.rpa_runner import run_fused_rpa
from tpu_cake.runner import RunMode


def _source_sha256(value: object) -> str:
    source = inspect.getsourcefile(inspect.unwrap(value))
    if source is None:
        raise RuntimeError("cannot locate a fused RPA backend source")
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def _backend_manifest() -> tuple[tuple[str, str], ...]:
    return (
        ("ragged_paged_attention_v3.py", _source_sha256(ragged_paged_attention)),
        ("tuned_block_sizes.py", _source_sha256(get_simplified_key)),
        ("tuned_block_sizes_v3.py", _source_sha256(get_tuned_block_sizes_v3)),
        ("util.py", _source_sha256(get_dtype_packing)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m tpu_cake.rpa_device_main")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--mode", type=RunMode, choices=tuple(RunMode), required=True)
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--measured-iterations", type=int, default=50)
    args = parser.parse_args()
    result = run_fused_rpa(
        args.output_dir,
        mode=args.mode,
        kernel=ragged_paged_attention,
        backend_manifest=_backend_manifest(),
        seed=args.seed,
        warmup_iterations=args.warmup_iterations,
        measured_iterations=args.measured_iterations,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
