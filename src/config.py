from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DetectorConfig:
    model_path: str
    confidence_threshold: float
    iou_threshold: float
    target_class_ids: list[int]
    device: str


@dataclass
class TrackerConfig:
    algorithm: str
    max_age: int
    min_hits: int
    iou_threshold: float
    max_cosine_distance: float
    n_init: int


@dataclass
class BehaviorConfig:
    standing_speed_threshold: float
    moving_speed_threshold: float
    loitering_seconds: int
    suspicious_idle_seconds: int
    irregular_motion_window: int
    irregular_motion_threshold: float


@dataclass
class DisplayConfig:
    show_trails: bool
    trail_size: int
    dashboard_width: int
    font_scale: float


@dataclass
class RuntimeConfig:
    source: str
    resize_width: int
    output_video: bool
    output_fps: int


@dataclass
class AppConfig:
    detector: DetectorConfig
    tracker: TrackerConfig
    behavior: BehaviorConfig
    display: DisplayConfig
    runtime: RuntimeConfig


def _as_list_of_int(values: list[Any]) -> list[int]:
    return [int(v) for v in values]


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    detector = DetectorConfig(
        model_path=str(raw["detector"]["model_path"]),
        confidence_threshold=float(raw["detector"]["confidence_threshold"]),
        iou_threshold=float(raw["detector"]["iou_threshold"]),
        target_class_ids=_as_list_of_int(raw["detector"]["target_class_ids"]),
        device=str(raw["detector"]["device"]),
    )

    tracker = TrackerConfig(
        algorithm=str(raw["tracker"]["algorithm"]).lower(),
        max_age=int(raw["tracker"]["max_age"]),
        min_hits=int(raw["tracker"]["min_hits"]),
        iou_threshold=float(raw["tracker"]["iou_threshold"]),
        max_cosine_distance=float(raw["tracker"]["max_cosine_distance"]),
        n_init=int(raw["tracker"]["n_init"]),
    )

    behavior = BehaviorConfig(
        standing_speed_threshold=float(raw["behavior"]["standing_speed_threshold"]),
        moving_speed_threshold=float(raw["behavior"]["moving_speed_threshold"]),
        loitering_seconds=int(raw["behavior"]["loitering_seconds"]),
        suspicious_idle_seconds=int(raw["behavior"]["suspicious_idle_seconds"]),
        irregular_motion_window=int(raw["behavior"]["irregular_motion_window"]),
        irregular_motion_threshold=float(raw["behavior"]["irregular_motion_threshold"]),
    )

    display = DisplayConfig(
        show_trails=bool(raw["display"]["show_trails"]),
        trail_size=int(raw["display"]["trail_size"]),
        dashboard_width=int(raw["display"]["dashboard_width"]),
        font_scale=float(raw["display"]["font_scale"]),
    )

    runtime = RuntimeConfig(
        source=str(raw["runtime"]["source"]),
        resize_width=int(raw["runtime"]["resize_width"]),
        output_video=bool(raw["runtime"]["output_video"]),
        output_fps=int(raw["runtime"]["output_fps"]),
    )

    return AppConfig(
        detector=detector,
        tracker=tracker,
        behavior=behavior,
        display=display,
        runtime=runtime,
    )
