from __future__ import annotations

from datetime import timedelta

import cv2
import numpy as np

from src.config import DisplayConfig


def _format_seconds(seconds: float) -> str:
    return str(timedelta(seconds=int(max(0, seconds))))


class DashboardRenderer:
    def __init__(self, config: DisplayConfig) -> None:
        self.config = config
        self.track_trails: dict[int, list[tuple[int, int]]] = {}

    def draw(
        self,
        frame: np.ndarray,
        tracks_with_behavior: list[dict[str, object]],
        total_count: int,
    ) -> np.ndarray:
        output = frame.copy()

        for item in tracks_with_behavior:
            track_id = int(item["track_id"])
            bbox = item["bbox"]
            risk = str(item["risk_level"])
            posture = str(item["posture"])
            dwell_time = float(item["dwell_time"])
            stationary_time = float(item.get("stationary_time", dwell_time))
            flags = item["flags"]

            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = self._pick_color(risk)

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            label = f"[ID: {track_id} | {risk} ⚠]"
            label2 = f"{posture} | in frame: {_format_seconds(dwell_time)} | still: {_format_seconds(stationary_time)}"
            if flags:
                label2 += f" | {', '.join(flags)}"

            cv2.putText(
                output,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale + 0.06,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                label2,
                (x1, min(output.shape[0] - 10, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                color,
                1,
                cv2.LINE_AA,
            )

            if self.config.show_trails:
                self._draw_trail(output, track_id, (x1 + x2) // 2, (y1 + y2) // 2, color)

        output = self._draw_side_panel(output, tracks_with_behavior, total_count)
        return output

    def _draw_side_panel(
        self,
        frame: np.ndarray,
        tracks_with_behavior: list[dict[str, object]],
        total_count: int,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        panel_w = min(self.config.dashboard_width, max(220, w // 2))

        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:] = (25, 25, 25)

        header = "LIVE BEHAVIOR DASHBOARD"
        cv2.putText(panel, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        cv2.putText(panel, f"People in frame: {total_count}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        suspicious_count = sum(1 for item in tracks_with_behavior if str(item["risk_level"]) == "Suspicious")
        cv2.putText(panel, f"Suspicious: {suspicious_count}", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        y = 108
        line_height = 24
        for item in tracks_with_behavior[: (h - 100) // line_height]:
            track_id = int(item["track_id"])
            risk = str(item["risk_level"])
            dwell = _format_seconds(float(item["dwell_time"]))
            flags = item["flags"]

            status = risk
            if flags:
                status += f" ({', '.join(flags)})"

            text = f"P{track_id}: {status} | {dwell}"
            color = self._pick_color(risk)
            cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
            y += line_height

        return np.hstack([frame, panel])

    @staticmethod
    def _pick_color(risk: str) -> tuple[int, int, int]:
        if risk == "Suspicious":
            return (0, 0, 255)
        if risk == "Watch":
            return (0, 180, 255)
        if risk == "Loitering":
            return (0, 215, 255)
        if risk == "Moving":
            return (0, 220, 120)
        return (0, 220, 120)

    def _draw_trail(self, frame: np.ndarray, track_id: int, x: int, y: int, color: tuple[int, int, int]) -> None:
        pts = self.track_trails.setdefault(track_id, [])
        pts.append((x, y))
        if len(pts) > self.config.trail_size:
            del pts[0]

        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], color, 2)
