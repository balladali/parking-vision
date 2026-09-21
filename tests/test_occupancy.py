import pytest

from parking_vision.config import Spot
from parking_vision.models import Detection
from parking_vision.occupancy import classify


@pytest.mark.parametrize("box,expected,coverage", [
    ((20, 20, 30, 30), False, 0),
    ((0, 0, 10, 10), True, 1),
    ((0, 0, 5, 10), True, 0.5),
    ((0, 0, 4, 10), False, 0.4),
    ((10, 0, 20, 10), False, 0),
    ((10, 10, 0, 0), False, 0),
])
def test_overlap_boundary(box: tuple, expected: bool, coverage: float) -> None:
    spot = Spot(id="A", polygon=[(0, 0), (10, 0), (10, 10), (0, 10)])
    result = classify([spot], [Detection(box, 0.9, 2)], 0.5)[0]
    assert result.occupied is expected
    assert result.coverage == pytest.approx(coverage)


def test_empty_detection_and_duplicate_boxes() -> None:
    spot = Spot(id="A", polygon=[(0, 0), (10, 0), (10, 10), (0, 10)])
    assert not classify([spot], [], 0.5)[0].occupied
    box = Detection((0, 0, 3, 10), 0.9, 2)
    assert not classify([spot], [box, box], 0.5)[0].occupied
