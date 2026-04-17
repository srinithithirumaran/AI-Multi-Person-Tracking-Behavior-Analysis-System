from __future__ import annotations

from typing import Any

import numpy as np
from ultralytics import YOLO

from src.config import DetectorConfig


class PersonDetector:
    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.model = YOLO(config.model_path)

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        results = self.model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=self.config.target_class_ids,
            device=self.config.device,
            verbose=False,
        )

        detections: list[dict[str, Any]] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            detections.append(
                {
                    "bbox": [float(v) for v in xyxy],
                    "confidence": conf,
                    "class_id": cls,
                }
            )

        return detections
