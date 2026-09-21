from io import StringIO
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from parking_vision.config import Config, Spot
from parking_vision.models import Detection, Frame
from parking_vision.pipeline import run
from parking_vision.video import open_video


class FakeDetector:
    def detect(self, frame: Frame) -> list[Detection]:
        return [Detection((0, 0, 16, 16), 0.9, 2)]


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    path = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (32, 32))
    assert writer.isOpened(), "MJPG codec is required for integration test"
    try:
        for _ in range(3):
            writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    finally:
        writer.release()
    return path


@pytest.mark.parametrize("limit,expected", [(None, 3), (1, 1)])
def test_real_video_pipeline(video_path: Path, limit: int | None, expected: int) -> None:
    config = Config(source=str(video_path), spots=[
        Spot(id="A", polygon=[(0, 0), (16, 0), (16, 16), (0, 16)]),
        Spot(id="B", polygon=[(20, 20), (30, 20), (30, 30), (20, 30)]),
    ])
    output = StringIO()
    assert run(config, FakeDetector(), output, limit) == expected
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [r["frame_index"] for r in records] == list(range(expected))
    assert records[0]["free"] == 1
    assert records[0]["spots"][0]["occupied"] is True


def test_video_released_on_exception(video_path: Path) -> None:
    with pytest.raises(RuntimeError):
        with open_video(str(video_path)) as reader:
            raise RuntimeError("failure")
    assert not reader.capture.isOpened()


def test_missing_video(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="Unable to open"):
        with open_video(str(tmp_path / "missing.avi")):
            pass


def test_outside_frame(video_path: Path) -> None:
    config = Config(source=str(video_path), spots=[
        Spot(id="A", polygon=[(0, 0), (100, 0), (100, 100)])
    ])
    with pytest.raises(ValueError, match="outside"):
        run(config, FakeDetector(), StringIO())
