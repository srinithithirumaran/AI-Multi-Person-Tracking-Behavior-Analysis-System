# 🚀 AI Multi-Person Tracking & Behavior Analysis System

> 🎯 Real-time AI surveillance system that detects, tracks, and analyzes human behavior using Deep Learning.

---


## 🔥 Key Features

✅ Real-time multi-person detection using YOLOv8
✅ Persistent tracking with DeepSORT / SORT
✅ Intelligent behavior analysis:

* 🟢 Normal movement
* 🟡 Loitering (idle too long)
* 🔴 Suspicious behavior (irregular motion)

✅ Live dashboard with:

* Person count
* Individual tracking IDs
* Behavior status

✅ Output generation:

* 📹 Recorded video
* 📊 CSV event logs

---

## 🧠 Tech Stack

* Python
* OpenCV
* YOLOv8 (Object Detection)
* DeepSORT (Tracking)
* NumPy / Pandas

---

## ⚙️ How It Works

1. Detect people using YOLOv8
2. Track each person with unique ID
3. Analyze movement patterns
4. Classify behavior in real-time
5. Display results in dashboard

---

## 📂 Project Structure

```
AI-Multi-Person-Tracking/
│── configs/
│── src/
│── outputs/
│── main.py
│── requirements.txt
```

---

## ▶️ Run the Project

```bash
pip install -r requirements.txt
python main.py --config configs/default.yaml
```

---

## 📊 Output

* 📁 Event Logs → `outputs/logs/`
* 🎥 Video Output → `outputs/videos/`

---

## ⚠️ Notes

* First run will download YOLO model
* Press `q` to stop the camera
* For GPU: set `device: "0"`

---

## 🌟 Why This Project Stands Out

This is not just detection — it is a **complete AI system** combining:

* Detection
* Tracking
* Behavior Intelligence

👉 Similar to real-world surveillance AI systems.

---
## 🎥 Demo Preview

👉 *(![Uploading Screenshot 2026-04-18 084617.png…]()
)*

---

## 👩‍💻 Author

Built with focus on real-world AI applications and system design.

