from __future__ import annotations

from dataclasses import dataclass, field


def iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


@dataclass
class Track:
    track_id: int
    bbox: list[float]
    age: int = 0
    hits: int = 1
    time_since_update: int = 0


@dataclass
class SimpleSort:
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    next_id: int = 1
    tracks: list[Track] = field(default_factory=list)

    def update(self, detections: list[list[float]]) -> list[list[float]]:
        for trk in self.tracks:
            trk.age += 1
            trk.time_since_update += 1

        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(range(len(self.tracks)))

        matches: list[tuple[int, int]] = []
        if self.tracks and detections:
            candidates: list[tuple[float, int, int]] = []
            for t_idx, trk in enumerate(self.tracks):
                for d_idx, det in enumerate(detections):
                    score = iou(trk.bbox, det[:4])
                    if score >= self.iou_threshold:
                        candidates.append((score, t_idx, d_idx))

            candidates.sort(reverse=True, key=lambda item: item[0])
            used_tracks: set[int] = set()
            used_dets: set[int] = set()
            for _, t_idx, d_idx in candidates:
                if t_idx in used_tracks or d_idx in used_dets:
                    continue
                used_tracks.add(t_idx)
                used_dets.add(d_idx)
                matches.append((t_idx, d_idx))

            unmatched_tracks -= used_tracks
            unmatched_dets -= used_dets

        for t_idx, d_idx in matches:
            det = detections[d_idx]
            trk = self.tracks[t_idx]
            trk.bbox = det[:4]
            trk.hits += 1
            trk.time_since_update = 0

        for d_idx in unmatched_dets:
            det = detections[d_idx]
            self.tracks.append(
                Track(track_id=self.next_id, bbox=det[:4])
            )
            self.next_id += 1

        self.tracks = [
            trk for trk in self.tracks if trk.time_since_update <= self.max_age
        ]

        output: list[list[float]] = []
        for trk in self.tracks:
            if trk.hits >= self.min_hits or trk.age < self.min_hits:
                x1, y1, x2, y2 = trk.bbox
                output.append([x1, y1, x2, y2, float(trk.track_id)])

        return output
