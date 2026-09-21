from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from parking_vision.config import DetectorConfig
from parking_vision.detection.yolo import YoloDetector


def test_yolo_adapter() -> None:
    detector = YoloDetector.__new__(YoloDetector)
    detector.config = DetectorConfig()
    boxes = MagicMock()
    boxes.cpu.return_value = boxes
    boxes.xyxy.tolist.return_value = [[1, 2, 10, 20]]
    boxes.conf.tolist.return_value = [0.8]
    boxes.cls.tolist.return_value = [2]
    detector.model = MagicMock()
    detector.model.predict.return_value = [SimpleNamespace(boxes=boxes)]
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    assert detections[0].xyxy == (1, 2, 10, 20)
    assert detections[0].class_id == 2
    assert detector.model.predict.call_args.kwargs["classes"] == [2, 3, 5, 7]
    assert detector.model.predict.call_args.kwargs["source"] is frame
