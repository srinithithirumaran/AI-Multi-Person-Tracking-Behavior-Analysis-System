# AI Multi-Person Tracking & Behavior Analysis System

Industry-style real-time surveillance AI pipeline using detection + tracking + behavior intelligence.

## Features

- Real-time multi-person detection with YOLOv8
- Persistent multi-target tracking with DeepSORT or SORT-style tracker
- Behavior analysis for:
  - Sitting/Standing (low movement)
  - Moving
  - Loitering (idle too long)
  - Suspicious idle (very long idle)
  - Irregular movement (high speed variance)
- Real-time dashboard overlay with per-person status
- CSV event logs and optional output recording

## Project Structure

- `main.py`: end-to-end runner
- `configs/default.yaml`: system configuration
- `src/core/detector.py`: YOLO detector
- `src/core/tracker.py`: DeepSORT/SORT adapter
- `src/analytics/behavior.py`: behavior engine
- `src/ui/dashboard.py`: drawing + side dashboard
- `src/utils/events.py`: CSV event logging

## Setup

1. Create environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run with webcam:

```powershell
python main.py --config configs/default.yaml
```

3. Run with video file by updating `runtime.source` in `configs/default.yaml`.

## Behavior Logic

- Speed is estimated from center displacement between frames.
- If speed is below thresholds for long durations:
  - `loitering_seconds` -> Loitering
  - `suspicious_idle_seconds` -> Suspicious idle
- High variance in speed history triggers irregular movement.

## Output

- Event CSV: `outputs/logs/events_*.csv`
- Recorded video: `outputs/videos/session_*.mp4` (if enabled)

## Notes

- First run may download YOLO model weights (`yolov8n.pt`).
- For better accuracy and GPU use, set `detector.device: "0"` (CUDA device id) when available.
- Press `q` to stop stream.
