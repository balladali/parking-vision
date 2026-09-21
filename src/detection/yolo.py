"""Ultralytics adapter, imported only when inference is requested."""

from parking_vision.config import DetectorConfig
from parking_vision.models import Detection, Frame


class YoloDetector:
    def __init__(self, config: DetectorConfig) -> None:
        from ultralytics import YOLO

        self.config = config
        self.model = YOLO(config.weights)

    def detect(self, frame: Frame) -> list[Detection]:
        result = self.model.predict(
            source=frame, conf=self.config.confidence,
            classes=self.config.classes, device=self.config.device, verbose=False,
        )[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.cpu()
        return [
            Detection(tuple(map(float, xyxy)), float(conf), int(cls))
            for xyxy, conf, cls in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=True
            )
        ]
