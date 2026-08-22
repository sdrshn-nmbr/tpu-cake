from pathlib import Path
from types import SimpleNamespace

import pytest

from tpu_cake import xprof_evidence
from tpu_cake.xprof_evidence import XPlaneIndex


def test_xplane_index_materializes_one_typed_snapshot_and_closes_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    closed = []
    source_event = SimpleNamespace(name="step", start_ns=7, duration_ns=11)

    class FakeProfile:
        planes = (
            SimpleNamespace(
                name="/device:TPU:0",
                stats={"device_type_string": "TPU7x"},
                lines=(SimpleNamespace(name="XLA Modules", events=(source_event,)),),
            ),
        )

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(
        xprof_evidence.profile_data.ProfileData,
        "from_file",
        lambda _path: FakeProfile(),
    )

    index = XPlaneIndex.from_file(tmp_path / "capture.xplane.pb")
    source_event.name = "mutated-after-close"

    assert closed == [True]
    assert index.planes[0].stat_map() == {"device_type_string": "TPU7x"}
    assert index.planes[0].lines[0].events[0].name == "step"
    assert index.event_count("step") == 1
    assert index.event_count("missing") == 0
