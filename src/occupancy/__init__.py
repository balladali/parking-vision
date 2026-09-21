"""Per-frame occupancy from vehicle boxes and calibrated spot polygons."""

import cv2
import numpy as np

from parking_vision.config import Spot
from parking_vision.models import Detection, Occupancy


def classify(spots: list[Spot], detections: list[Detection], threshold: float) -> list[Occupancy]:
    """Use maximum box/spot intersection divided by spot area (not box IoU)."""
    states: list[Occupancy] = []
    for spot in spots:
        polygon = np.array(spot.polygon, dtype=np.float32)
        area = cv2.contourArea(polygon)
        coverage = 0.0
        for detection in detections:
            x1, y1, x2, y2 = detection.xyxy
            if x2 <= x1 or y2 <= y1:
                continue
            rectangle = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], dtype=np.float32)
            intersection, _ = cv2.intersectConvexConvex(polygon, rectangle)
            coverage = max(coverage, min(1.0, max(0.0, intersection / area)))
        states.append(Occupancy(spot.id, coverage >= threshold, coverage))
    return states
