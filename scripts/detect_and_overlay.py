
#!/usr/bin/env python3
"""
AICI Challenge 1 — Simplified baseline with "no overlap" option.
Detects objects (chair, couch, etc.) using YOLOv8 and overlays boxes on room.pgm.
Adds --only-best-frame to keep detections from a single best-scoring frame.
"""

import argparse, os, json, cv2, numpy as np
from pathlib import Path

AICI_TO_COCO = {
    "bathtub": "bathtub",
    "chair": "chair",
    "couch": "couch",
    "shelf": "book",
    "table": "dining table",
    "wc": "toilet",
}
COCO_FILTER = {v for v in AICI_TO_COCO.values()}

def _load_yolo(model_name="yolov8n.pt"):
    try:
        from ultralytics import YOLO
        return YOLO(model_name)
    except Exception as e:
        raise RuntimeError("Please install ultralytics: pip install ultralytics opencv-python") from e

def load_map(pgm_path: Path):
    m = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read map: {pgm_path}")
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

def pick_frames(rgb_dir: Path, max_frames: int):
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    frames = sorted([p for p in rgb_dir.rglob("*") if p.suffix.lower() in exts])
    if not frames:
        raise FileNotFoundError(f"No RGB frames found in {rgb_dir}")
    return frames[:max_frames]

def overlay_on_map(map_img, dets, out_size=None):
    canvas = map_img.copy()
    if out_size is not None:
        canvas = cv2.resize(canvas, out_size, interpolation=cv2.INTER_LINEAR)
    for d in dets:
        x1,y1,x2,y2 = map(int, [d["x1"], d["y1"], d["x2"], d["y2"]])
        cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,255,0), 2)
        label = f'{d["cls_name"]} {d["score"]:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1-18), (x1+tw+6, y1-2), (0,255,0), -1)
        cv2.putText(canvas, label, (x1+3, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return canvas

def run(args):
    map_img = load_map(Path(args.room_pgm))
    frames = pick_frames(Path(args.rgb_dir), args.max_frames)
    model = _load_yolo(args.model)

    frame_dets = {}
    best_frame = None
    best_score_sum = -1.0

    for img_path in frames:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        r = model.predict(source=img, conf=args.conf, verbose=False)[0]
        dets_this, score_sum = [], 0.0

        for b in r.boxes:
            cls_id = int(b.cls[0])
            score = float(b.conf[0])
            x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
            cls_name = r.names[cls_id]
            if cls_name not in COCO_FILTER:
                continue
            dets_this.append({
                "image": img_path.name,
                "cls_name": cls_name,
                "score": score,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": W, "height": H
            })
            score_sum += score

        if dets_this:
            frame_dets[img_path.name] = dets_this
            if score_sum > best_score_sum:
                best_score_sum = score_sum
                best_frame = img_path.name

    if args.only_best_frame and best_frame:
        print(f"[INFO] Using best frame: {best_frame} (score={best_score_sum:.3f})")
        all_dets = frame_dets[best_frame]
    else:
        all_dets = [d for detlist in frame_dets.values() for d in detlist]

    os.makedirs(args.out_dir, exist_ok=True)
    det_path = os.path.join(args.out_dir, "detections.json")
    with open(det_path, "w") as f:
        json.dump(all_dets, f, indent=2)
    print(f"[OK] Wrote detections: {det_path} ({len(all_dets)} boxes)")

    if all_dets:
        H = all_dets[0]["height"]
        W = all_dets[0]["width"]
        overlay = overlay_on_map(map_img, all_dets, out_size=(W, H))
        out_png = os.path.join(args.out_dir, "map_with_detections.png")
        cv2.imwrite(out_png, overlay)
        print(f"[OK] Wrote overlay: {out_png}")
    else:
        print("[WARN] No detections found; skipping overlay.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--room-pgm", required=True, help="Path to room.pgm")
    ap.add_argument("--rgb-dir", required=True, help="Directory with RGB frames")
    ap.add_argument("--out-dir", default="results", help="Output directory")
    ap.add_argument("--max-frames", type=int, default=50, help="Process up to N frames")
    ap.add_argument("--conf", type=float, default=0.4, help="YOLO confidence threshold")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model checkpoint")
    ap.add_argument("--only-best-frame", action="store_true", help="Use detections from the single best frame (no overlaps)")
    args = ap.parse_args()
    run(args)
