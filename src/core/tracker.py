from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from src.config import TrackerConfig
from src.core.simple_sort import SimpleSort


class MultiPersonTracker:
    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self.algorithm = config.algorithm

        if self.algorithm == "deepsort":
            self.tracker = DeepSort(
                max_age=config.max_age,
                n_init=config.n_init,
                max_cosine_distance=config.max_cosine_distance,
            )
        elif self.algorithm == "sort":
            self.tracker = SimpleSort(
                max_age=config.max_age,
                min_hits=config.min_hits,
                iou_threshold=config.iou_threshold,
            )
        else:
            raise ValueError(
                "tracker.algorithm must be either 'deepsort' or 'sort'"
            )

    def update(self, detections: Iterable[dict[str, Any]], frame: np.ndarray) -> list[dict[str, Any]]:
        if self.algorithm == "deepsort":
            return self._update_deepsort(detections, frame)
        return self._update_sort(detections)

    def _update_deepsort(
        self, detections: Iterable[dict[str, Any]], frame: np.ndarray
    ) -> list[dict[str, Any]]:
        ds_inputs: list[tuple[list[float], float, str]] = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            ds_inputs.append(([x1, y1, w, h], det["confidence"], "person"))

        tracks = self.tracker.update_tracks(ds_inputs, frame=frame)

        output: list[dict[str, Any]] = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            left, top, right, bottom = track.to_ltrb()
            output.append(
                {
                    "track_id": int(track.track_id),
                    "bbox": [
                        float(left),
                        float(top),
                        float(right),
                        float(bottom),
                    ],
                    "confidence": 1.0,
                }
            )
        return output

    def _update_sort(self, detections: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        sort_inputs = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            sort_inputs.append([x1, y1, x2, y2, det["confidence"]])

        if sort_inputs:
            np_inputs = np.array(sort_inputs, dtype=float)
        else:
            np_inputs = np.empty((0, 5), dtype=float)

        tracks = self.tracker.update(np_inputs)

        output: list[dict[str, Any]] = []
        for trk in tracks:
            x1, y1, x2, y2, track_id = trk.tolist()
            output.append(
                {
                    "track_id": int(track_id),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": 1.0,
                }
            )
        return output
