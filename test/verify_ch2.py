#!/usr/bin/env python3
"""
verify_ch2_v2.py — QA for Challenge 2 point clouds (with headless-safe previews)
Requires: open3d, matplotlib, numpy

Improvements over v1:
- If Open3D offscreen rendering fails (e.g., headless Colab without EGL/OSMesa),
  it will ALWAYS create previews using Matplotlib projections (top, front, side).
- Saves: report.json + PNG previews for each input PLY.

Usage:
  python verify_ch2_v2.py --ply office_colored.ply bathroom_colored.ply --out ch2_verify_results
"""
import argparse, os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

def percent_colored(cols, tol=3/255.0):
    if cols.size == 0:
        return 0.0
    default = np.array([128/255.0, 128/255.0, 128/255.0], dtype=np.float64)
    dist = np.linalg.norm(cols - default, axis=1)
    colored = np.count_nonzero(dist > tol)
    return 100.0 * colored / cols.shape[0]

def bbox_stats(pts):
    if pts.size == 0:
        return (np.zeros(3), np.zeros(3), np.zeros(3), 0.0)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    ext = mx - mn
    vol = float(ext[0]*ext[1]*ext[2])
    return mn, mx, ext, vol

def plot_projection(pts, cols, out_png, title=""):
    if pts.size == 0:
        # create empty fig
        plt.figure(figsize=(10, 10))
        plt.title(title + " (empty)")
        plt.axis('off')
        plt.savefig(out_png, bbox_inches='tight', dpi=150)
        plt.close()
        return

    # normalize colors to 0..1
    if cols.size == 0:
        cols = np.full((pts.shape[0],3), 0.5, dtype=float)
    else:
        cols = np.clip(cols, 0.0, 1.0)

    # Three views: XY (top), XZ (front), YZ (side)
    views = [
        ("Top (X-Y)", 0, 1),
        ("Front (X-Z)", 0, 2),
        ("Side (Y-Z)", 1, 2),
    ]

    plt.figure(figsize=(15, 5))
    for i,(label,a,b) in enumerate(views, start=1):
        ax = plt.subplot(1, 3, i)
        ax.scatter(pts[:,a], pts[:,b], s=0.2, c=cols)
        ax.set_title(label)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle=':', linewidth=0.5)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def load_pcd_open3d(path):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)  # [0,1]
    return pcd, pts, cols

def save_preview_open3d(pcd, out_png):
    import open3d as o3d
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1200, height=900)
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    b = pcd.get_axis_aligned_bounding_box()
    ctr.set_lookat(b.get_center())
    ctr.set_zoom(0.7)
    vis.get_render_option().point_size = 1.5
    vis.update_renderer()
    vis.capture_screen_image(out_png)
    vis.destroy_window()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", nargs="+", required=True)
    ap.add_argument("--out", default="ch2_verify_results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    report = []
    for p in args.ply:
        name = os.path.splitext(os.path.basename(p))[0]
        out_png1 = os.path.join(args.out, f"{name}_o3d.png")
        out_png2 = os.path.join(args.out, f"{name}_proj.png")

        # Try Open3D load
        try:
            pcd, pts, cols = load_pcd_open3d(p)
        except Exception as e:
            # Fallback: lightweight PLY reader (very basic, assumes ASCII 'vertex' list)
            # If this fails, instruct user to install open3d.
            try:
                import open3d as o3d  # re-raise if not installed
                raise
            except Exception:
                raise SystemExit(
                    "Failed to read PLY without open3d. Please install it:\n  pip install open3d"
                ) from e

        mn, mx, ext, vol = bbox_stats(pts)
        density = (pts.shape[0]/vol) if vol > 0 else 0.0
        pct = percent_colored(cols)

        # Try Open3D preview; if it fails, use Matplotlib projection
        o3d_ok = True
        try:
            save_preview_open3d(pcd, out_png1)
        except Exception:
            o3d_ok = False
            out_png1 = None

        # Always generate the projection preview
        try:
            plot_projection(pts, cols, out_png2, title=name)
        except Exception:
            out_png2 = None

        info = {
            "file": p,
            "points": int(pts.shape[0]),
            "bbox_min": mn.tolist(),
            "bbox_max": mx.tolist(),
            "extent_m": ext.tolist(),
            "volume_m3": vol,
            "density_pts_per_m3": density,
            "percent_colored": pct,
            "preview_open3d": out_png1,
            "preview_projection": out_png2,
        }
        report.append(info)

    out_json = os.path.join(args.out, "report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Wrote report to {out_json}")
    for r in report:
        print(f"- {r['file']}: {r['points']} pts | colored: {r['percent_colored']:.1f}% | extent (m): {np.round(r['extent_m'],3)}")
        if r['preview_open3d']:
            print(f"  o3d preview: {r['preview_open3d']}")
        if r['preview_projection']:
            print(f"  proj preview: {r['preview_projection']}")

if __name__ == "__main__":
    main()
