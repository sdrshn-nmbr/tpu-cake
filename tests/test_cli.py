from pathlib import Path

from tpu_cake.cli import _render_workload, _verify_schedule


def test_frontend_schedule_round_trips_through_parser(tmp_path: Path) -> None:
    schedule = tmp_path / "matmul.mlir"
    assert _render_workload("matmul", schedule) == 0
    assert _verify_schedule(schedule) == 0
