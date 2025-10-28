# AICI Challenge 1 & 2 — Perceptra Solution

## Overview
This repository contains the implementation and outputs for **Challenge 1** and **Challenge 2** of the AICI Computer Vision & Robotics Assessment.  

- **Challenge 1:** Detect and localize indoor furniture (chair, couch, table, shelf, bathtub, WC) on a **2D occupancy grid map** using RGB survey data.  
- **Challenge 2:** Concatenate and **colorize LiDAR point clouds** using synchronized RGB camera data to produce a **single colored 3D point cloud (PLY)** representing the full surveyed environment.

---

## 📦 Folder Structure

```
Perceptra/
│
├── scripts/
│   ├── detect_and_overlay_v2.py       # Challenge 1
│   ├── list_bag_topics.py             # Utility to inspect ROS bag topics
│   └── colorize_and_merge_fixed.py    # Challenge 2 (final robust version)
│
├── results/
│   └── office/
│       ├── detections.json            # Challenge 1: object poses & sizes
│       ├── map_with_detections.png    # Challenge 1: overlay on occupancy grid
│       └── office_colored.ply         # Challenge 2: merged colored point cloud
│
├── data/
│   └── office/
│       ├── room.pgm                   # Occupancy map
│       ├── room.yaml                  # Map metadata (resolution, origin)
│       └── frames/                    # RGB frames extracted from ROS bag
│
└── Perceptra.ipynb                    # Notebook used for testing in Google Colab
```

---

## 🚀 Challenge 1 — Object Detection & Projection

### Script
`detect_and_overlay_v2.py`

Runs YOLOv8 object detection on RGB frames and projects detections into **map coordinates** (meters).  
Generates oriented bounding boxes and stores full detection metadata.

### Usage
```bash
python scripts/detect_and_overlay_v2.py   --room-yaml data/office/room.yaml   --room-pgm  data/office/room.pgm   --rgb-dir   data/office/frames   --out-dir   results/office   --model yolov8n.pt   --only-best-frame
```

### Output
- `detections.json` → Object metadata (class, pose, size)  
- `map_with_detections.png` → Visual overlay of detections on occupancy map  

---

## 🚀 Challenge 2 — Point Cloud Concatenation & Colorization

### Script
`colorize_and_merge_fixed.py`

This script merges LiDAR and RGB camera data into a single **colored point cloud** by:
1. Reading LiDAR and camera messages from ROS2 bags (`rosbags` library).  
2. Using TF transforms for spatial alignment.  
3. Projecting camera colors onto LiDAR points.  
4. Downsampling to manage memory (optimized for Colab).  

### Example (Colab command used)
```bash
python scripts/colorize_and_merge_fixed.py   --bags "/content/drive/MyDrive/Perceptra/bags/office_survey_1"   --cloud-topic "/livox/lidar"   --image-topic "/zed/zed_node/rgb/image_rect_color/compressed"   --caminfo-topic "/zed/zed_node/rgb/camera_info"   --tf-topics /tf /tf_static   --world-frame livox_frame   --sync-tol 0.5   --stride 6   --max-clouds 600   --percloud-voxel 0.08   --voxel 0.15   --out results/office/office_colored.ply
```

### Output
- `office_colored.ply` → Merged, colorized 3D point cloud of the environment  
  (fully viewable in **Open3D**, **CloudCompare**, or **MeshLab**)  

---

## 🧰 Dependencies
Install via pip:
```bash
pip install rosbags open3d==0.18.0 opencv-python-headless numpy pyyaml tqdm ultralytics
```

---

## ✅ Compliance Checklist

| Requirement | Challenge 1 | Challenge 2 |
|--------------|-------------|-------------|
| Non-overlapping bounding boxes | ✅ | — |
| ≥ 2 object classes detected | ✅ | — |
| Map-aligned detections (meters) | ✅ | — |
| Pose & dimension metadata | ✅ | — |
| Oriented bounding boxes | ✅ | — |
| Merged point cloud (PLY) | — | ✅ |
| Colorized using RGB images | — | ✅ |
| TF synchronization | — | ✅ |
| Reproducible CLI script | ✅ | ✅ |

---

## 🧩 Summary
This solution fully completes **Challenge 1** and **Challenge 2** of the AICI Robotics assessment:  
- **Challenge 1:** map-aligned detections, JSON + visual overlays  
- **Challenge 2:** robust colored point-cloud generation with per-cloud voxel downsampling (Colab-optimized)  

Both stages are reproducible via command-line or Colab Notebook and ready for final submission.
