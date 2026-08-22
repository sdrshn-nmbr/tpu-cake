from __future__ import annotations

import argparse
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

from tpu_cake.artifacts import file_sha256
from tpu_cake.rpa_donation_confirmation import InklingRpaDonationConfirmationContract
from tpu_cake.rpa_donation_confirmation_runner import (
    capture_inkling_rpa_donation_hlo_identities,
    run_inkling_rpa_donation_confirmation,
)
from tpu_cake.rpa_runner import run_fused_rpa
from tpu_cake.rpa_search import RpaSearchContract, run_rpa_search
from tpu_cake.rpa_surface import InklingShardedRpaSurfaceContract
from tpu_cake.rpa_surface_runner import run_inkling_sharded_rpa_surface
from tpu_cake.runner import RunMode


def _source_sha256(value: object) -> str:
    source = inspect.getsourcefile(inspect.unwrap(value))
    if source is None:
        raise RuntimeError("cannot locate a fused RPA backend source")
    return file_sha256(Path(source))


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
    parser.add_argument("--mode", type=RunMode, choices=tuple(RunMode))
    parser.add_argument("--search-contract", type=Path)
    parser.add_argument("--surface-contract", type=Path)
    parser.add_argument("--donation-confirmation-contract", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--warmup-iterations", type=int)
    parser.add_argument("--measured-iterations", type=int)
    parser.add_argument(
        "--decode-block-sizes",
        type=int,
        nargs=4,
        metavar=("BQ", "BKV", "CQ", "CKV"),
    )
    args = parser.parse_args()
    if args.donation_confirmation_contract is not None:
        if (
            args.surface_contract is not None
            or args.search_contract is not None
            or args.mode is not None
        ):
            parser.error(
                "--donation-confirmation-contract cannot be combined with another run mode"
            )
        manual_parameters = (
            args.seed,
            args.warmup_iterations,
            args.measured_iterations,
            args.decode_block_sizes,
        )
        if any(value is not None for value in manual_parameters):
            parser.error(
                "--donation-confirmation-contract cannot be combined with manual run parameters"
            )
        contract = InklingRpaDonationConfirmationContract.model_validate_json(
            args.donation_confirmation_contract.read_text()
        )
        if contract.hlo_identity_status == "pending":
            result = capture_inkling_rpa_donation_hlo_identities(
                args.output_dir,
                contract,
                ragged_paged_attention,
            )
        else:
            result = run_inkling_rpa_donation_confirmation(
                args.output_dir,
                contract,
                ragged_paged_attention,
            )
        print(result.model_dump_json(indent=2))
        return
    if args.surface_contract is not None:
        manual_parameters = (
            args.seed,
            args.warmup_iterations,
            args.measured_iterations,
            args.decode_block_sizes,
        )
        if (
            args.mode is not None
            or args.search_contract is not None
            or any(value is not None for value in manual_parameters)
        ):
            parser.error("--surface-contract cannot be combined with manual run parameters")
        contract = InklingShardedRpaSurfaceContract.model_validate_json(
            args.surface_contract.read_text()
        )
        result = run_inkling_sharded_rpa_surface(
            args.output_dir,
            contract,
            ragged_paged_attention,
        )
        print(result.model_dump_json(indent=2))
        return
    if args.search_contract is not None:
        manual_parameters = (
            args.seed,
            args.warmup_iterations,
            args.measured_iterations,
            args.decode_block_sizes,
        )
        if args.mode is not None or any(value is not None for value in manual_parameters):
            parser.error("--search-contract cannot be combined with manual run parameters")
        contract = RpaSearchContract.model_validate_json(args.search_contract.read_text())
        result = run_rpa_search(
            args.output_dir,
            contract,
            kernel=ragged_paged_attention,
            backend_manifest=_backend_manifest(),
        )
        print(result.model_dump_json(indent=2))
        return
    if args.mode is None:
        parser.error("--mode is required unless a contract is supplied")
    result = run_fused_rpa(
        args.output_dir,
        mode=args.mode,
        kernel=ragged_paged_attention,
        backend_manifest=_backend_manifest(),
        seed=args.seed if args.seed is not None else 97,
        warmup_iterations=args.warmup_iterations if args.warmup_iterations is not None else 5,
        measured_iterations=(
            args.measured_iterations if args.measured_iterations is not None else 50
        ),
        decode_block_sizes=(
            tuple(args.decode_block_sizes)
            if args.decode_block_sizes is not None
            else (8, 128, 8, 128)
        ),
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
