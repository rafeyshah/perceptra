# AICI Challenge 1 — Perceptra Solution

## Overview
This repository contains the implementation and outputs for **Challenge 1** of the AICI Computer Vision & Robotics Assessment.  
The goal was to detect and localize indoor furniture (chair, couch, table, shelf, bathtub, WC) on a **2D occupancy grid map** using RGB survey data.

---

## 📦 Folder Structure

```
Perceptra/
│
├── scripts/
│   └── detect_and_overlay_v2.py     # Final compliant script
│
├── results/
│   └── office/
│       ├── detections.json          # Object poses and sizes (in map coordinates)
│       └── map_with_detections.png  # Overlay of detections on occupancy grid
│
├── data/
│   └── office/
│       ├── room.pgm                 # Occupancy map
│       ├── room.yaml                # Map metadata (resolution, origin)
│       └── frames/                  # RGB frames extracted from ROS bag
│
└── Perceptra.ipynb                  # Notebook used for testing and visualization
```

---

## 🚀 Script Description

### File: `detect_and_overlay_v2.py`

This script runs YOLOv8 object detection on the RGB frames of each survey and projects detections into **map coordinates** (meters). It also generates oriented bounding boxes and stores full detection metadata.

### Command-line usage

```bash
python scripts/detect_and_overlay_v2.py   --room-yaml data/office/room.yaml   --room-pgm  data/office/room.pgm   --rgb-dir   data/office/frames   --out-dir   results/office   --model yolov8n.pt   --only-best-frame
```

### Parameters
| Flag | Description |
|------|--------------|
| `--room-yaml` | Path to YAML file containing map metadata (resolution, origin) |
| `--room-pgm`  | Occupancy grid map (PGM) |
| `--rgb-dir`   | Directory with extracted RGB frames |
| `--out-dir`   | Output directory for results |
| `--model`     | YOLOv8 model checkpoint (default: `yolov8n.pt`) |
| `--only-best-frame` | Use detections from a single best frame (to avoid overlaps) |

---

## 📊 Output Files

### `detections.json`
Each object entry includes:
```json
{
  "cls_name": "chair",
  "score": 0.74,
  "pose": {"x": 2.18, "y": 1.56, "theta": 0.0},
  "size": {"length": 0.46, "width": 0.48, "height": 0.75}
}
```

### `map_with_detections.png`
Occupancy grid with oriented bounding boxes and class labels superimposed.

---

## ✅ Compliance Checklist

| Requirement | Status |
|--------------|---------|
| Non-overlapping bounding boxes | ✅ |
| ≥ 2 object classes detected | ✅ |
| Map-aligned detections (meters) | ✅ |
| Pose & dimension metadata | ✅ |
| Oriented bounding boxes | ✅ |
| Reproducible CLI script | ✅ |

---

## 🧩 Summary

This solution completes **Challenge 1** successfully with a clean and reproducible baseline:
- Fully map-aligned detections  
- Oriented bounding boxes  
- JSON output with pose, dimensions, and class info  
- Verified non-overlapping results for the *office* survey
