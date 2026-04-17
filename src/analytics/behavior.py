from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot

from src.config import BehaviorConfig


@dataclass
class TrackBehaviorState:
    first_seen_ts: float
    last_seen_ts: float
    last_center: tuple[float, float]
    stationary_seconds: float = 0.0
    speed_history: deque[float] = field(default_factory=lambda: deque(maxlen=30))
    trajectory: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=60))


class BehaviorAnalyzer:
    def __init__(self, config: BehaviorConfig) -> None:
        self.config = config
        self.states: dict[int, TrackBehaviorState] = {}

    def update_track(
        self,
        track_id: int,
        bbox: list[float],
        timestamp: float,
        fps: float,
    ) -> dict[str, object]:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        center = (cx, cy)

        if track_id not in self.states:
            self.states[track_id] = TrackBehaviorState(
                first_seen_ts=timestamp,
                last_seen_ts=timestamp,
                last_center=center,
                speed_history=deque(maxlen=max(5, self.config.irregular_motion_window)),
                trajectory=deque(maxlen=120),
            )

        state = self.states[track_id]
        dt = max(1.0 / max(fps, 1.0), timestamp - state.last_seen_ts)

        dx = center[0] - state.last_center[0]
        dy = center[1] - state.last_center[1]
        speed = hypot(dx, dy) / dt

        state.speed_history.append(speed)
        state.trajectory.append(center)
        if speed <= self.config.standing_speed_threshold:
            state.stationary_seconds += dt
        else:
            state.stationary_seconds = 0.0
        state.last_center = center
        state.last_seen_ts = timestamp

        dwell_time = timestamp - state.first_seen_ts
        avg_speed = sum(state.speed_history) / max(len(state.speed_history), 1)

        posture = "Standing"
        if avg_speed >= self.config.moving_speed_threshold:
            posture = "Moving"

        flags: list[str] = []
        status = "Normal"

        if state.stationary_seconds >= self.config.suspicious_idle_seconds:
            status = "Suspicious"
            flags.append("Idle too long")
        elif state.stationary_seconds >= self.config.loitering_seconds:
            status = "Loitering"
            flags.append("Stayed too long")

        if self._is_irregular_motion(state) or avg_speed >= self.config.moving_speed_threshold * 1.5:
            status = "Suspicious"
            flags.append("Suspicious movement")

        if status == "Normal" and posture == "Moving":
            status = "Moving"

        return {
            "track_id": track_id,
            "dwell_time": dwell_time,
            "stationary_time": state.stationary_seconds,
            "avg_speed": avg_speed,
            "posture": posture,
            "flags": flags,
            "risk_level": status,
            "status_text": status,
        }

    def prune_missing(self, active_track_ids: set[int], max_absent_seconds: float, timestamp: float) -> None:
        to_delete: list[int] = []
        for track_id, state in self.states.items():
            if track_id in active_track_ids:
                continue
            if timestamp - state.last_seen_ts > max_absent_seconds:
                to_delete.append(track_id)

        for track_id in to_delete:
            del self.states[track_id]

    def _is_irregular_motion(self, state: TrackBehaviorState) -> bool:
        if len(state.speed_history) < max(5, self.config.irregular_motion_window // 2):
            return False

        values = list(state.speed_history)
        mean_speed = sum(values) / len(values)
        variance = sum((v - mean_speed) ** 2 for v in values) / len(values)

        return variance > self.config.irregular_motion_threshold
