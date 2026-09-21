"""Streaming processing without retaining frames in memory."""

from dataclasses import asdict
import json
import logging
from typing import Protocol, TextIO

from parking_vision.config import Config
from parking_vision.models import Detection, Frame
from parking_vision.occupancy import classify
from parking_vision.video import open_video

logger = logging.getLogger(__name__)


class Detector(Protocol):
    def detect(self, frame: Frame) -> list[Detection]: ...


def run(config: Config, detector: Detector, output: TextIO, max_frames: int | None = None) -> int:
    count = 0
    with open_video(config.source) as video:
        for index, frame in video.frames():
            height, width = frame.shape[:2]
            if any(x > width or y > height for spot in config.spots for x, y in spot.polygon):
                raise ValueError("Spot polygon is outside the frame; recalibrate YAML coordinates")
            states = classify(config.spots, detector.detect(frame), config.occupancy_threshold)
            output.write(json.dumps({
                "frame_index": index,
                "free": sum(not state.occupied for state in states),
                "total": len(states),
                "spots": [asdict(state) for state in states],
            }) + "\n")
            output.flush()
            count += 1
            logger.debug("Processed frame %d", index)
            if max_frames is not None and count >= max_frames:
                break
    logger.info("Processed %d frames", count)
    return count
