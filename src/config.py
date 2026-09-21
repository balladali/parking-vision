"""Strict YAML configuration; relative paths resolve against the YAML directory."""

from pathlib import Path
from typing import Annotated, Literal, Self

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

type Coordinate = Annotated[float, Field(ge=0, le=100000, allow_inf_nan=False)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Spot(StrictModel):
    id: str = Field(min_length=1)
    polygon: list[tuple[Coordinate, Coordinate]] = Field(min_length=3)

    @model_validator(mode="after")
    def validate_polygon(self) -> Self:
        contour = np.array(self.polygon, dtype=np.float32)
        if not cv2.isContourConvex(contour) or cv2.contourArea(contour) <= 0:
            raise ValueError("polygon must be convex, non-degenerate and ordered around its perimeter")
        return self


class DetectorConfig(StrictModel):
    weights: str = "yolo11n.pt"
    confidence: float = Field(default=0.25, gt=0, le=1, allow_inf_nan=False)
    classes: list[Annotated[int, Field(ge=0)]] = Field(default_factory=lambda: [2, 3, 5, 7], min_length=1)
    device: str = "cpu"


class Config(StrictModel):
    source: str | Annotated[int, Field(ge=0)]
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    occupancy_threshold: float = Field(default=0.3, gt=0, le=1, allow_inf_nan=False)
    spots: list[Spot] = Field(min_length=1)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        if len({spot.id for spot in self.spots}) != len(self.spots):
            raise ValueError("spot IDs must be unique")
        if isinstance(self.source, str) and not self.source.strip():
            raise ValueError("source must not be empty")
        return self


def load_config(path: Path) -> Config:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    # YAML sequences are lists; convert only point pairs for strict tuple validation.
    if isinstance(data, dict) and isinstance(data.get("spots"), list):
        for spot in data["spots"]:
            if isinstance(spot, dict) and isinstance(spot.get("polygon"), list):
                spot["polygon"] = [tuple(p) if isinstance(p, list) else p for p in spot["polygon"]]
    config = Config.model_validate(data)
    if isinstance(config.source, str):
        config.source = str((path.parent / config.source).resolve())
    weights = Path(config.detector.weights)
    # Bare model names permit Ultralytics' first-run download; explicit paths are local.
    if weights.is_absolute() or len(weights.parts) > 1:
        config.detector.weights = str((path.parent / weights).resolve())
        if not Path(config.detector.weights).is_file():
            raise ValueError(f"Weights file does not exist: {config.detector.weights}")
    return config
