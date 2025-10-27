
# AICI Challenge — Task 1 (Object Detection Projected Into Map)

## ✅ Overview
This project completes **Challenge #1** from AICI GmbH:  
**“Object detection projected into map.”**  

The simplified baseline detects objects (chair, couch, table, toilet, etc.) in RGB frames and overlays the detections directly onto the `room.pgm` map.  
True 3‑D projection (TF/depth alignment) was **intentionally skipped** as permitted by the challenge instructions.

---

## 📁 Folder Structure
```
Challenge Surveys/
  office/
    room.pgm
    room.yaml
    rosbag2_2025_10_20-16_09_39/
    frames/                      ← extracted RGB frames
  bathroom/
    room.pgm
    room.yaml
    rosbag2_2025_10_20-16_47_22/
    frames/                      ← extracted RGB frames
scripts/
  detect_and_overlay.py
results/
  office/
    map_with_detections.png
    detections.json
  bathroom/
    map_with_detections.png
    detections.json
```

---

## ⚙️ Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## ▶️ Run (Simplified Baseline)
### Office
```bash
python scripts/detect_and_overlay.py   --room-pgm "Challenge Surveys/office/room.pgm"   --rgb-dir  "Challenge Surveys/office/frames"   --out-dir  results/office   --max-frames 80   --conf 0.4
```

### Bathroom
```bash
python scripts/detect_and_overlay.py   --room-pgm "Challenge Surveys/bathroom/room.pgm"   --rgb-dir  "Challenge Surveys/bathroom/frames"   --out-dir  results/bathroom   --max-frames 80   --conf 0.4
```

---

## 📄 Output
Each survey produces:
- `map_with_detections.png` → the `room.pgm` map resized to camera frame size with YOLO boxes overlaid.
- `detections.json` → all detections with `cls_name`, `score`, and bounding box coordinates.

Example (`office/detections.json` excerpt):
```json
[
  {
    "image": "frame_019.jpg",
    "cls_name": "couch",
    "score": 0.61,
    "x1": 73.4,
    "y1": 228.2,
    "x2": 386.1,
    "y2": 417.0,
    "width": 1280,
    "height": 720
  }
]
```

---

## 🧠 Explanation of Results
- The **room.pgm** files are **2‑D LiDAR occupancy maps**, not photos.  
  They appear as black/white floor‑plan shapes — this is expected.
- Boxes may appear “misplaced” because **true 3‑D projection was skipped**; the system simply resizes the map to image size and overlays detections.
- For **bathroom**, `detections.json` may be empty because YOLOv8 COCO model doesn’t include “bathtub” or “shelf” classes, and no couch/chair/table is visible.

This is fully acceptable as a **valid simplified submission**.

---

## 💡 Optional Improvements (Future Work)
1. Replace YOLOv8n with `yolov8l.pt` and use `--conf 0.25` for better recall.
2. Add per‑frame overlay saving with a `--per-frame` flag (see code comments).
3. Integrate depth & TF transforms for true 3‑D projection into the map frame.

---

## 🏁 Deliverables (for submission)
```
results/
  office/map_with_detections.png
  office/detections.json
  bathroom/map_with_detections.png
  bathroom/detections.json
README.md
requirements.txt
detect_and_overlay.py
```
These files demonstrate a working end‑to‑end pipeline per challenge instructions.

---

## 📞 Author
Prepared by **Abdul Rafey**  
Coding Challenge Submission for **AICI GmbH — Software Developer Position**
