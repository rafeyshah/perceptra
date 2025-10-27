#!/usr/bin/env python3
"""
AICI Challenge 1 — v2 compliant baseline
Adds map-coordinate conversion, pose+dimension fields, and oriented boxes.
"""

import argparse, os, json, cv2, yaml, numpy as np
from pathlib import Path
from ultralytics import YOLO

AICI_TO_COCO = {
    "bathtub": "bathtub",
    "chair": "chair",
    "couch": "couch",
    "shelf": "book",
    "table": "dining table",
    "wc": "toilet",
}
COCO_FILTER = set(AICI_TO_COCO.values())


# ---------- helpers ----------
def load_map_and_meta(pgm_path, yaml_path):
    img = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(pgm_path)
    meta = yaml.safe_load(open(yaml_path))
    res = float(meta["resolution"])
    origin = np.array(meta["origin"][:2], float)  # [ox, oy]
    theta0 = float(meta["origin"][2]) if len(meta["origin"]) == 3 else 0.0
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), res, origin, theta0


def pick_frames(rgb_dir, max_frames):
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    return sorted([p for p in Path(rgb_dir).rglob("*") if p.suffix.lower() in exts])[:max_frames]


def pixel_to_map(x_px, y_px, res, origin, H):
    """Convert image pixel (x_px,y_px) to map (x,y) in meters."""
    # map.pgm has (0,0) bottom-left at origin[0:2]
    x_m = origin[0] + x_px * res
    y_m = origin[1] + (H - y_px) * res
    return x_m, y_m


def draw_oriented_box(canvas, center, size, angle_deg, color=(0,255,0)):
    rect = (center, size, angle_deg)
    box = cv2.boxPoints(rect).astype(int)
    cv2.polylines(canvas, [box], True, color, 2)
    return canvas


# ---------- main ----------
def run(args):
    map_img, res, origin, theta0 = load_map_and_meta(args.room_pgm, args.room_yaml)
    H_map, W_map = map_img.shape[:2]
    frames = pick_frames(args.rgb_dir, args.max_frames)
    model = YOLO(args.model)

    best_frame, best_sum, detections = None, -1, {}

    for p in frames:
        img = cv2.imread(str(p))
        if img is None: continue
        H, W = img.shape[:2]
        r = model.predict(img, conf=args.conf, verbose=False)[0]
        dets, ssum = [], 0.0
        for b in r.boxes:
            cls_name = r.names[int(b.cls)]
            if cls_name not in COCO_FILTER: continue
            score = float(b.conf)
            x1,y1,x2,y2 = map(float, b.xyxy[0])
            cx, cy = (x1+x2)/2, (y1+y2)/2
            w_px, h_px = x2-x1, y2-y1

            # rough orientation from aspect ratio
            angle = 0.0 if w_px >= h_px else 90.0

            # convert to map coords
            x_m, y_m = pixel_to_map(cx, cy, res, origin, H_map)
            L, Wm = w_px*res, h_px*res

            dets.append({
                "image": p.name,
                "cls_name": cls_name,
                "score": score,
                "pose": {"x": x_m, "y": y_m, "theta": np.deg2rad(angle)},
                "size": {"length": L, "width": Wm, "height": 0.75},  # crude height
            })
            ssum += score

        if dets:
            detections[p.name] = dets
            if ssum > best_sum:
                best_sum, best_frame = ssum, p.name

    if args.only_best_frame and best_frame:
        all_dets = detections[best_frame]
        print(f"[INFO] Using best frame {best_frame}")
    else:
        all_dets = [d for dl in detections.values() for d in dl]

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, "detections.json")
    json.dump(all_dets, open(out_json, "w"), indent=2)
    print(f"[OK] Saved {len(all_dets)} detections → {out_json}")

    # overlay oriented boxes on the map
    canvas = map_img.copy()
    for d in all_dets:
        cx = int((d["pose"]["x"] - origin[0]) / res)
        cy = int(H_map - (d["pose"]["y"] - origin[1]) / res)
        Lp = d["size"]["length"] / res
        Wp = d["size"]["width"] / res
        angle_deg = np.rad2deg(d["pose"]["theta"])
        draw_oriented_box(canvas, (cx, cy), (Lp, Wp), angle_deg)
        label = f'{d["cls_name"]} {d["score"]:.2f}'
        cv2.putText(canvas, label, (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    out_png = os.path.join(args.out_dir, "map_with_detections.png")
    cv2.imwrite(out_png, canvas)
    print(f"[OK] Saved overlay → {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--room-yaml", required=True)
    ap.add_argument("--room-pgm", required=True)
    ap.add_argument("--rgb-dir", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-frames", type=int, default=50)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--only-best-frame", action="store_true")
    args = ap.parse_args()
    run(args)
