from pathlib import Path

from tpu_cake.cli import _verify_schedule


def test_example_schedule_round_trips_through_parser() -> None:
    example = Path(__file__).parents[1] / "examples" / "matmul.mlir"
    assert _verify_schedule(example) == 0
