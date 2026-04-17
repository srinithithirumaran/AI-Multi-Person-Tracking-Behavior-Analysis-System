from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2


def parse_source(source: str) -> int | str:
    source = source.strip()
    if source.isdigit():
        return int(source)
    return source


def ensure_output_dirs() -> tuple[Path, Path]:
    logs_dir = Path("outputs/logs")
    videos_dir = Path("outputs/videos")
    logs_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir, videos_dir


def build_video_writer(frame_width: int, frame_height: int, fps: int) -> tuple[cv2.VideoWriter, Path]:
    _, videos_dir = ensure_output_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = videos_dir / f"session_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (frame_width, frame_height))
    return writer, path
