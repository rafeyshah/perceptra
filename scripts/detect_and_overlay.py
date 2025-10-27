#!/usr/bin/env python3
"""
Challenge 1 (simplified baseline): Detect objects in RGB frames and overlay the boxes directly onto the occupancy map.
- No 3D projection, no TF usage. We simply resize the map to the RGB frame size and draw detections on top.
- Outputs: a PNG per survey with boxes overlaid on the map + a JSON with raw detections.
Author: You (candidate)
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

# --- Target classes for AICI challenge (map them to COCO classes used by YOLOv8) ---
AICI_TO_COCO = {
    "bathtub": "bathtub",      # not in COCO; YOLOv8n default won't detect this (kept for completeness)
    "chair": "chair",
    "couch": "couch",
    "shelf": "book",           # weak proxy; COCO lacks "shelf"
    "table": "dining table",
    "wc": "toilet",
}

# Fallback list we will filter by from YOLO model names.
COCO_FILTER = {v for v in AICI_TO_COCO.values()}

# Try to import ultralytics lazily so the file can still be inspected without env installed.
def _load_yolo(model_name:str="yolov8n.pt"):
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "Ultralytics not installed. Run: pip install ultralytics opencv-python pyyaml\n"
            "If you're offline, install from a wheel cache."
        ) from e
    return YOLO(model_name)

def load_map(pgm_path: Path):
    # Load occupancy grid as grayscale, then convert to 3-channel
    m = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read map image: {pgm_path}")
    m_rgb = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    return m_rgb

def pick_frames(rgb_dir: Path, max_frames: int):
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    frames = sorted([p for p in rgb_dir.rglob("*") if p.suffix.lower() in exts])
    if not frames:
        raise FileNotFoundError(f"No RGB frames found under: {rgb_dir}")
    return frames[:max_frames]

def overlay_on_map(map_img, dets, out_size=None):
    """
    Resize map to out_size (w,h) if provided, then draw rectangles for each det.
    det: {cls_name, score, x1,y1,x2,y2}
    """
    canvas = map_img.copy()
    if out_size is not None:
        canvas = cv2.resize(canvas, out_size, interpolation=cv2.INTER_LINEAR)
    for d in dets:
        x1,y1,x2,y2 = map(int, [d["x1"], d["y1"], d["x2"], d["y2"]])
        cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,255,0), 2)
        label = f'{d["cls_name"]} {d["score"]:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # draw label background
        cv2.rectangle(canvas, (x1, y1-18), (x1+tw+6, y1-2), (0,255,0), -1)
        cv2.putText(canvas, label, (x1+3, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return canvas

def run(args):
    map_img = load_map(Path(args.room_pgm))
    frames = pick_frames(Path(args.rgb_dir), args.max_frames)

    model = _load_yolo(args.model)
    all_dets = []
    H, W = None, None

    for img_path in frames:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue
        H, W = img.shape[:2]
        r = model.predict(source=img, conf=args.conf, verbose=False)[0]

        for b in r.boxes:
            cls_id = int(b.cls[0])
            score = float(b.conf[0])
            x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
            cls_name = r.names[cls_id]
            if cls_name not in COCO_FILTER:
                continue
            all_dets.append({
                "image": str(img_path.name),
                "cls_name": cls_name,
                "score": score,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": W, "height": H
            })

    os.makedirs(args.out_dir, exist_ok=True)
    # Save detections JSON
    det_path = os.path.join(args.out_dir, "detections.json")
    with open(det_path, "w") as f:
        json.dump(all_dets, f, indent=2)
    print(f"[OK] Wrote detections: {det_path} ({len(all_dets)} boxes)")

    # Generate a single overlay image by drawing all boxes on a copy of the map resized to last frame size
    if H is None or W is None:
        raise RuntimeError("No valid frames processed; cannot create overlay.")
    overlay = overlay_on_map(map_img, all_dets, out_size=(W, H))
    out_png = os.path.join(args.out_dir, "map_with_detections.png")
    cv2.imwrite(out_png, overlay)
    print(f"[OK] Wrote overlay: {out_png}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--room-pgm", required=True, help="Path to room.pgm")
    ap.add_argument("--rgb-dir", required=True, help="Directory with RGB frames (.jpg/.png)")
    ap.add_argument("--out-dir", default="results", help="Output directory")
    ap.add_argument("--max-frames", type=int, default=50, help="Process up to N frames")
    ap.add_argument("--conf", type=float, default=0.4, help="YOLO confidence threshold")
    ap.add_argument("--model", default="yolov8n.pt", help="Ultralytics model checkpoint (local path or name)")
    args = ap.parse_args()
    run(args)
