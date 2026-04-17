from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


class EventLogger:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_file()

    def _init_file(self) -> None:
        if self.output_path.exists():
            return
        with self.output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "timestamp",
                    "track_id",
                    "risk_level",
                    "posture",
                    "dwell_seconds",
                    "avg_speed",
                    "flags",
                ]
            )

    def log(self, behavior: dict[str, object]) -> None:
        with self.output_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    behavior["track_id"],
                    behavior["risk_level"],
                    behavior["posture"],
                    f"{float(behavior['dwell_time']):.2f}",
                    f"{float(behavior['avg_speed']):.2f}",
                    "|".join(behavior["flags"]),
                ]
            )
