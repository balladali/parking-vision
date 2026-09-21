from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from parking_vision.config import Config, Spot, load_config


def test_example_config() -> None:
    config = load_config(Path(__file__).parents[1] / "configs/example.yaml")
    assert len(config.spots) == 2
    assert Path(config.source).is_absolute()
    assert config.detector.classes == [2, 3, 5, 7]


@pytest.mark.parametrize("changes", [
    {"occupancy_threshold": 0}, {"occupancy_threshold": 1.1},
    {"occupancy_threshold": float("nan")}, {"spots": []},
    {"source": -1}, {"source": True}, {"source": ""},
    {"unknown": 1}, {"detector": {"confidence": 2}},
])
def test_invalid_config(changes: dict) -> None:
    data = {"source": 0, "spots": [{"id": "A", "polygon": [(0, 0), (10, 0), (10, 10)]}]}
    data.update(changes)
    with pytest.raises(ValidationError):
        Config.model_validate(data)


@pytest.mark.parametrize("polygon", [
    [(0, 0), (1, 1), (2, 2)],
    [(0, 0), (10, 10), (0, 10), (10, 0)],
    [(0, 0), (10, 0), (4, 4), (10, 10), (0, 10)],
])
def test_bad_polygon(polygon: list) -> None:
    with pytest.raises(ValidationError):
        Spot(id="A", polygon=polygon)


def test_duplicate_ids() -> None:
    spot = Spot(id="A", polygon=[(0, 0), (10, 0), (10, 10)])
    with pytest.raises(ValidationError, match="unique"):
        Config(source=0, spots=[spot, spot])


def test_paths_relative_to_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"source": "video.avi", "spots": [
        {"id": "A", "polygon": [[0, 0], [10, 0], [10, 10]]}
    ]}), encoding="utf-8")
    assert load_config(path).source == str(tmp_path / "video.avi")
