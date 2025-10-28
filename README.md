# 🧠 Perceptra – AICI Robotics Challenge Submission

This repository contains the full solution for the **AICI GmbH Computer Vision & Robotics Challenge**, covering **Challenge 1**, **Challenge 2**, and a placeholder for **Challenge 3 (Alignment)**.

---

## 🧩 Challenge 1 – Object Detection & Map Overlay

### Objective
Detect at least two of six required object classes and overlay oriented bounding boxes on the occupancy grid map.

### Implementation Summary
- Used **YOLOv8** for object detection (COCO pretrained).
- Mapped detections to the AICI challenge classes (bathtub, chair, couch, shelf, table, WC).
- Computed poses (x, y, θ) and dimensions in map coordinates.
- Generated a JSON output and overlay PNG on the occupancy grid.

### Key Outputs
| File | Description |
|------|--------------|
| `detections.json` | All detections with class, dimensions, and pose. |
| `map_with_detections.png` | Occupancy grid map with oriented bounding boxes. |

**Result:** ✅ Detections correctly projected and aligned.  
**Verified Classes:** *Chair, Couch*.

---

## 🪶 Challenge 2 – LiDAR + Camera Colorization & Fusion

### Objective
Merge point clouds and colorize them using synchronized camera frames and TF transforms.

### Implementation Summary
- Used **colorize_and_merge** (final version).  
- Reads ROS bag (`/tf`, `/tf_static`, `/livox/lidar`, `/zed/...` topics).  
- Synchronizes LiDAR–Camera data with dynamic & static TF lookups.  
- Projects image colors onto LiDAR points.  
- Downsamples clouds (voxel size 0.15 m).  
- Verified using `verify_ch2_v2.py`.

### Final Command Used
```bash
python colorize_and_merge.py   --bags "{BAG}"   --cloud-topic "/livox/lidar"   --image-topic "/zed/zed_node/rgb/image_rect_color/compressed"   --caminfo-topic "/zed/zed_node/rgb/camera_info"   --tf-topics /tf /tf_static   --world-frame base_link   --sync-tol 0.5   --stride 6   --max-clouds 600   --voxel 0.15   --out results/<survey>_colored.ply
```

### Verification Summary (from `verify_ch2_v2.py`)
| Survey | Points | Extent (m) | Volume (m³) | Density (pts/m³) | % Colored | Verdict |
|:--|--:|--:|--:|--:|--:|:--|
| **office_colored.ply** | 72,369 | 21.5 × 22.3 × 4.47 | 2,149.9 | 33.7 | **24.3 %** | ✅ Pass |
| **bathroom_colored.ply** | 13,312 | 6.4 × 9.1 × 2.75 | 160.4 | 83.1 | **21.2 %** | ✅ Pass |

### Visual Projections
| File | Description |
|------|--------------|
| `office_colored_proj.png` | Top, side, and front projections of the colored point cloud. |
| `bathroom_colored_proj.png` | Projections of the bathroom scan. |

### Validation Notes
- All 100 / 100 LiDAR clouds successfully transformed via TF.  
- Camera intrinsics and TF alignment verified (`TF_img=ok`).  
- Average colorization coverage ≈ 22 % (expected for forward-facing RGB + 360° LiDAR).  
- Geometry, density, and color alignment confirmed visually.

**Result:** ✅ Challenge 2 successfully completed.

---

## 🧭 Challenge 3 – Alignment (Placeholder)

### Objective (future extension)
Align multiple colorized point clouds (office, bathroom, etc.) into a unified world coordinate frame.

### Planned Approach
- Use **Open3D ICP** or **RANSAC-based registration**.  
- Optionally refine via **feature-based alignment (FPFH descriptors)**.  
- Output: merged, globally aligned `.ply` map.

**Status:** ⏳ Pending (not required for current submission).

---

## 📊 Verification & Evaluation

All metrics computed using:
```bash
python verify_ch2_v2.py   --ply office_colored.ply bathroom_colored.ply   --out ch2_verify_results
```

- Checks bounding box, volume, density, and percent of points colorized.  
- Saves report as `report.json`.  
- Generates preview PNGs for visual confirmation.

**All validation criteria passed for Challenge 1 & 2.**

---

## 🧾 Summary of Deliverables

| Challenge | Output Files | Status |
|------------|---------------|---------|
| 1 | `detections.json`, `map_with_detections.png` | ✅ Completed |
| 2 | `office_colored.ply`, `bathroom_colored.ply`, `report.json`, `*_proj.png` | ✅ Completed |
| 3 | `align_maps.py` (planned) | ⏳ Optional |

---

**Author:** Abdul Rafey  
**Project:** Perceptra  
**Institution:** AICI GmbH – Computer Vision & Robotics Challenge 2025  
**Tools:** ROS 2 / Open3D / YOLOv8 / TF2 / Python 3.10 / Colab GPU
