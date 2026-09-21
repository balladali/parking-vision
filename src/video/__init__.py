"""OpenCV capture with deterministic resource cleanup."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import cv2

from parking_vision.models import Frame


class VideoReader:
    def __init__(self, capture: Any, source: str | int) -> None:
        self.capture = capture
        self.source = source

    def frames(self) -> Iterator[tuple[int, Frame]]:
        index = 0
        while True:
            ok, frame = self.capture.read()
            if not ok:
                if index == 0 or isinstance(self.source, int):
                    raise OSError("Unable to read a frame from the video source")
                return
            yield index, frame
            index += 1


@contextmanager
def open_video(source: str | int) -> Iterator[VideoReader]:
    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened():
            raise OSError(f"Unable to open video source: {source}")
        yield VideoReader(capture, source)
    finally:
        capture.release()
