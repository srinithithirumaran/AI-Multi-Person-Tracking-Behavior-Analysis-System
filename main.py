from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from src.analytics.behavior import BehaviorAnalyzer
from src.config import AppConfig, load_config
from src.core.detector import PersonDetector
from src.core.tracker import MultiPersonTracker
from src.ui.dashboard import DashboardRenderer
from src.utils.events import EventLogger
from src.utils.io import build_video_writer, ensure_output_dirs, parse_source


def _resize_keep_aspect(frame, width: int):
    if width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w == width:
        return frame
    ratio = width / float(w)
    new_h = int(h * ratio)
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)


def run(config: AppConfig) -> None:
    source = parse_source(config.runtime.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {config.runtime.source}")

    detector = PersonDetector(config.detector)
    tracker = MultiPersonTracker(config.tracker)
    behavior = BehaviorAnalyzer(config.behavior)
    dashboard = DashboardRenderer(config.display)

    logs_dir, _ = ensure_output_dirs()
    event_log_path = logs_dir / f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    logger = EventLogger(event_log_path)

    writer = None
    output_video_path = None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = float(config.runtime.output_fps)

    print("Starting stream. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = _resize_keep_aspect(frame, config.runtime.resize_width)
        timestamp = time.time()

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

        track_behaviors: list[dict[str, object]] = []
        active_track_ids: set[int] = set()

        for trk in tracks:
            tid = int(trk["track_id"])
            active_track_ids.add(tid)

            state = behavior.update_track(
                track_id=tid,
                bbox=trk["bbox"],
                timestamp=timestamp,
                fps=fps,
            )
            merged = {
                **trk,
                **state,
            }
            track_behaviors.append(merged)
            logger.log(state)

        behavior.prune_missing(
            active_track_ids=active_track_ids,
            max_absent_seconds=5.0,
            timestamp=timestamp,
        )

        rendered = dashboard.draw(
            frame=frame,
            tracks_with_behavior=track_behaviors,
            total_count=len(track_behaviors),
        )

        cv2.imshow("AI Multi-Person Tracking & Behavior Analysis", rendered)

        if config.runtime.output_video:
            if writer is None:
                h, w = rendered.shape[:2]
                writer, output_video_path = build_video_writer(
                    frame_width=w,
                    frame_height=h,
                    fps=max(1, int(fps)),
                )
            writer.write(rendered)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print(f"Event log saved at: {event_log_path}")
    if output_video_path is not None:
        print(f"Output video saved at: {output_video_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Multi-Person Tracking & Behavior Analysis System"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        fallback = Path(__file__).resolve().parent / args.config
        if fallback.exists():
            config_path = fallback
        else:
            fallback = Path(__file__).resolve().parent / "configs" / config_path.name
            if fallback.exists():
                config_path = fallback
            else:
                raise FileNotFoundError(f"Config not found: {config_path}")
    config = load_config(config_path)
    run(config)


if __name__ == "__main__":
    main()
