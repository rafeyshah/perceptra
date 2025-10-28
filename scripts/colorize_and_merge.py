#!/usr/bin/env python3
"""
Challenge 2 — Point cloud concatenation & colorization (robust, CompressedImage ok)
Fixed:
- Avoids using `or` with NumPy arrays in TF lookups (None checks instead).
- Handles cases where no image is found near a cloud timestamp (no img_frame reference error).
- Adds optional per-cloud voxel downsampling to reduce RAM.
"""

import argparse, sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np, cv2
from tqdm import tqdm

import open3d as o3d
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# ------------- utils -------------
def ensure_dir(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def to_ns(stamp) -> int:
    sec = getattr(stamp, 'sec', None); nsec = getattr(stamp, 'nanosec', None)
    if sec is not None: return int(sec)*1_000_000_000 + int(nsec or 0)
    secs = getattr(stamp, 'secs', None); nsecs = getattr(stamp, 'nsecs', None)
    if secs is not None: return int(secs)*1_000_000_000 + int(nsecs or 0)
    return int(stamp)

def quat_to_mat(x,y,z,w):
    xx,yy,zz = x*x,y*y,z*z; xy,xz,yz=x*y,x*z,y*z; wx,wy,wz=w*x,w*y,w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)  ],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)  ],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=np.float64)

def SE3(Rt):
    T = np.eye(4); T[:3,:3]=Rt[:3,:3]; T[:3,3]=Rt[:3,3]; return T

def transform_points(T, xyz):
    if xyz.size == 0: return xyz
    pts = np.c_[xyz, np.ones((xyz.shape[0],1))]
    return (T @ pts.T).T[:,:3]

# ------------- decoders -------------
def decode_image(msg) -> Tuple[np.ndarray, str]:
    h,w = int(msg.height), int(msg.width)
    enc = msg.encoding.decode() if isinstance(msg.encoding,(bytes,bytearray)) else str(msg.encoding)
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc in ('bgr8','rgb8'):
        img = buf.reshape(h, msg.step)[:, :w*3].reshape(h,w,3).copy()
        if enc == 'rgb8': img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img, 'bgr8'
    if enc in ('mono8',):
        img = buf.reshape(h, msg.step)[:, :w].reshape(h,w,1).copy()
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 'bgr8'
    if enc in ('rgba8','bgra8'):
        img = buf.reshape(h, msg.step)[:, :w*4].reshape(h,w,4).copy()
        if enc == 'rgba8': img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img, 'bgr8'
    raise ValueError(f"Unsupported Image encoding: {enc}")

def decode_compressed_image(msg) -> Tuple[np.ndarray, str]:
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Failed to decode CompressedImage")
    return img, 'bgr8'

_DT = {1: np.int8,2: np.uint8,3: np.int16,4: np.uint16,5: np.int32,6: np.uint32,7: np.float32,8: np.float64}

def decode_pointcloud2(msg):
    names = [f.name.decode() if isinstance(f.name,(bytes,bytearray)) else str(f.name) for f in msg.fields]
    formats = [_DT[int(f.datatype)] for f in msg.fields]
    offsets = [int(f.offset) for f in msg.fields]
    itemsize = int(msg.point_step)
    dtype = np.dtype({'names': names, 'formats': formats, 'offsets': offsets, 'itemsize': itemsize})
    raw = np.frombuffer(bytes(msg.data), dtype=dtype)
    if int(msg.height) > 1:
        raw = raw.reshape(int(msg.height), int(msg.row_step)//itemsize).reshape(-1)
    if not set(('x','y','z')).issubset(raw.dtype.names): raise ValueError("PointCloud2 missing x/y/z")
    xyz = np.vstack((raw['x'],raw['y'],raw['z'])).T.astype(np.float64)
    mask = np.isfinite(xyz).all(axis=1); xyz = xyz[mask]
    rgb=None
    if 'rgb' in raw.dtype.names:
        f = raw['rgb'][mask].view(np.uint32); r=(f>>16)&255; g=(f>>8)&255; b=f&255
        rgb=np.vstack((r,g,b)).T.astype(np.uint8)
    elif 'rgba' in raw.dtype.names:
        f = raw['rgba'][mask].view(np.uint32); r=(f>>24)&255; g=(f>>16)&255; b=(f>>8)&255
        rgb=np.vstack((r,g,b)).T.astype(np.uint8)
    return xyz, rgb

# ------------- TF buffer -------------
from collections import defaultdict, deque
class TFBuffer:
    def __init__(self): self.store=defaultdict(deque)
    def add(self, parent, child, t_ns, trans, rot):
        T=np.eye(4); T[:3,:3]=quat_to_mat(float(rot.x),float(rot.y),float(rot.z),float(rot.w))
        T[:3,3]=[float(trans.x),float(trans.y),float(trans.z)]
        dq=self.store[(parent,child)]; dq.append((t_ns,T)); 
        if len(dq)>800: dq.popleft()
    def _lookup_direct(self, parent, child, t_ns, tol_ns):
        dq=self.store.get((parent,child)); 
        if not dq: return None
        best=None; best_dt=None
        for (tn,T) in dq:
            dt=abs(tn-t_ns)
            if best is None or dt<best_dt: best,best_dt=(T,dt),dt
        return best[0] if (best and best_dt<=tol_ns) else None
    def lookup(self, target, source, t_ns, tol_ns=100_000_000):
        if target==source: return np.eye(4)
        from collections import deque as Q
        q=Q([(source,np.eye(4))]); seen={source}
        while q:
            frm,Ttc=q.popleft()
            T=self._lookup_direct(target, frm, t_ns, tol_ns)
            if T is not None: return T @ Ttc
            for (p,c),dq in self.store.items():
                if p==frm and c not in seen:
                    Tpc=self._lookup_direct(p,c,t_ns,tol_ns)
                    if Tpc is None: continue
                    seen.add(c); q.append((c, Ttc @ np.linalg.inv(SE3(Tpc))))
                elif c==frm and p not in seen:
                    Tpf=self._lookup_direct(p,c,t_ns,tol_ns)
                    if Tpf is None: continue
                    seen.add(p); q.append((p, Ttc @ SE3(Tpf)))
        return None

# ------------- projection -------------
def colorize_points_from_image(xyz_cam, img_bgr, K):
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    X,Y,Z = xyz_cam[:,0],xyz_cam[:,1],xyz_cam[:,2]
    eps=1e-6; valid=Z>eps
    u=(fx*(X/(Z+eps))+cx).astype(np.int32)
    v=(fy*(Y/(Z+eps))+cy).astype(np.int32)
    H,W=img_bgr.shape[:2]
    m=valid & (u>=0) & (u<W) & (v>=0) & (v<H)
    colors=np.full((xyz_cam.shape[0],3),128,dtype=np.uint8)
    colors[m]=img_bgr[v[m],u[m]]
    return colors

# ------------- bag plumbing -------------
def expand_bags(paths: List[str]) -> List[Path]:
    out=[]
    for p in paths:
        P=Path(p)
        if P.is_dir():
            db=list(sorted(P.glob("*.db3"))); out.extend(db if db else [P])
        else:
            out.append(P)
    return out

def open_reader_with_fallback(paths: List[Path]):
    order=[Stores.ROS2_IRON, Stores.ROS2_HUMBLE, Stores.ROS2_FOXY, Stores.ROS1_NOETIC]
    last=None
    for st in order:
        try:
            rdr=AnyReader(paths, default_typestore=get_typestore(st)); rdr.open(); return rdr
        except Exception as e:
            last=e; continue
    raise last if last else RuntimeError("Unable to open bag with any typestore")

def read_bags(args):
    from dataclasses import dataclass
    @dataclass
    class FB: images:list; caminfos:list; clouds:list
    fb=FB(images=[], caminfos=[], clouds=[])
    tf=TFBuffer()

    bag_paths=expand_bags(args.bags)
    rdr=open_reader_with_fallback(bag_paths)
    try:
        conns={}
        for c in rdr.connections: conns.setdefault(c.topic, []).append(c)
        def msgs(topic):
            for c in conns.get(topic, []):
                for _,_,raw in rdr.messages(connections=[c]): yield c, raw

        # TF
        for tft in (args.tf_topics or []):
            for c,raw in msgs(tft):
                m=rdr.deserialize(raw, c.msgtype)
                for ts in getattr(m,'transforms',[]):
                    parent = ts.header.frame_id.decode() if isinstance(ts.header.frame_id,(bytes,bytearray)) else str(ts.header.frame_id)
                    child  = ts.child_frame_id.decode() if isinstance(ts.child_frame_id,(bytes,bytearray)) else str(ts.child_frame_id)
                    tf.add(parent, child, to_ns(ts.header.stamp), ts.transform.translation, ts.transform.rotation)

        # CameraInfo
        for c,raw in msgs(args.caminfo_topic):
            m=rdr.deserialize(raw, c.msgtype)
            K=np.array(m.k if hasattr(m,'k') else m.K, dtype=np.float64).reshape(3,3)
            W,H=int(m.width),int(m.height)
            fid = m.header.frame_id.decode() if isinstance(m.header.frame_id,(bytes,bytearray)) else str(m.header.frame_id)
            fb.caminfos.append((to_ns(m.header.stamp), K, (W,H), fid))

        # Images (handles CompressedImage)
        for c,raw in msgs(args.image_topic):
            m=rdr.deserialize(raw, c.msgtype)
            if 'CompressedImage' in c.msgtype:
                img,_=decode_compressed_image(m)
            else:
                img,_=decode_image(m)
            fid = m.header.frame_id.decode() if isinstance(m.header.frame_id,(bytes,bytearray)) else str(m.header.frame_id)
            fb.images.append((to_ns(m.header.stamp), img, fid))

        # Clouds
        i=0
        for c,raw in msgs(args.cloud_topic):
            if args.max_clouds and i>=args.max_clouds: break
            if args.stride>1 and (i%args.stride)!=0: i+=1; continue
            m=rdr.deserialize(raw, c.msgtype)
            xyz,_=decode_pointcloud2(m)
            fid = m.header.frame_id.decode() if isinstance(m.header.frame_id,(bytes,bytearray)) else str(m.header.frame_id)
            fb.clouds.append((to_ns(m.header.stamp), xyz, fid)); i+=1
    finally:
        try: rdr.close()
        except Exception: pass

    fb.images.sort(key=lambda x:x[0]); fb.caminfos.sort(key=lambda x:x[0]); fb.clouds.sort(key=lambda x:x[0])
    return fb, tf

def nearest_by_time(items, t_ns, tol_ns):
    if not items: return None
    lo,hi=0,len(items)-1
    while lo<hi:
        mid=(lo+hi)//2
        if items[mid][0]<t_ns: lo=mid+1
        else: hi=mid
    cand=[lo, max(0,lo-1)]
    best=None; bestdt=None
    for i in cand:
        dt=abs(items[i][0]-t_ns)
        if best is None or dt<bestdt: best,bestdt=i,dt
    return best if bestdt <= tol_ns else None

def run(args):
    fb, tf = read_bags(args)
    if not fb.clouds: print("[ERROR] No point clouds found.", file=sys.stderr); return 2
    if not fb.caminfos: print("[WARN] No CameraInfo; cannot colorize.", file=sys.stderr)
    if not fb.images: print("[WARN] No images; output will be gray.", file=sys.stderr)
    K, cam_frame = (fb.caminfos[-1][1], fb.caminfos[-1][3]) if fb.caminfos else (None,None)
    tol_ns = int(args.sync_tol*1e9)

    ptsA=[]; rgbA=[]
    for (t_ns, xyz, cloud_frame) in tqdm(fb.clouds, desc="Colorizing & merging clouds"):
        # Transform cloud into world frame
        T_wc = tf.lookup(args.world_frame, cloud_frame, t_ns, tol_ns)
        if T_wc is None and fb.clouds:
            T_wc = tf.lookup(args.world_frame, cloud_frame, fb.clouds[0][0], tol_ns)
        if T_wc is None:
            # If still None and world==cloud frame, just identity
            if args.world_frame == cloud_frame:
                T_wc = np.eye(4)
            else:
                continue  # skip this cloud
        xyz_world = transform_points(SE3(T_wc), xyz)

        # Default color = gray
        colors = np.full((xyz_world.shape[0],3),128,np.uint8)

        # Try to colorize if we have images & K
        if fb.images and K is not None:
            idx = nearest_by_time(fb.images, t_ns, tol_ns)
            if idx is not None:
                t_img, img_bgr, img_frame = fb.images[idx]
                T_wc2 = tf.lookup(args.world_frame, img_frame, t_img, tol_ns)
                if T_wc2 is None:
                    T_wc2 = tf.lookup(args.world_frame, img_frame, t_ns, tol_ns)
                if T_wc2 is not None:
                    xyz_cam = transform_points(np.linalg.inv(SE3(T_wc2)), xyz_world)
                    colors = colorize_points_from_image(xyz_cam, img_bgr, K)

        # Per-cloud downsample before merging (to keep RAM low)
        if args.percloud_voxel and args.percloud_voxel > 0:
            v = float(args.percloud_voxel)
            q = np.floor(xyz_world / v).astype(np.int64)
            _, keep_idx = np.unique(q, axis=0, return_index=True)
            keep_idx.sort()
            xyz_world = xyz_world[keep_idx]
            colors = colors[keep_idx]

        ptsA.append(xyz_world.astype(np.float32)); rgbA.append(colors.astype(np.uint8))

    if not ptsA: print("[ERROR] After TF/sync, zero points remained.", file=sys.stderr); return 3
    pts=np.concatenate(ptsA,0); rgb=np.concatenate(rgbA,0)
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors=o3d.utility.Vector3dVector((rgb/255.0).astype(np.float64))
    if args.voxel and args.voxel>0: pcd=pcd.voxel_down_sample(float(args.voxel))
    ensure_dir(args.out); o3d.io.write_point_cloud(args.out, pcd, write_ascii=False, compressed=False)
    print(f"[OK] Wrote colored point cloud → {args.out}")
    print(f"[INFO] Points: {np.asarray(pcd.points).shape[0]}  (voxel={args.voxel}, percloud_voxel={args.percloud_voxel})")
    return 0

def build_argparser():
    ap = argparse.ArgumentParser(description="Concatenate & colorize point clouds from ROS bag(s).")
    ap.add_argument("--bags", nargs="+", required=True, help="Bag folders (rosbag2) or files (*.db3/*.bag).")
    ap.add_argument("--cloud-topic", required=True, help="PointCloud2 topic (e.g., /livox/lidar).")
    ap.add_argument("--image-topic", required=True, help="Image/CompressedImage topic (e.g., /zed/.../image_rect_color/compressed).")
    ap.add_argument("--caminfo-topic", required=True, help="CameraInfo topic (e.g., /zed/.../camera_info).")
    ap.add_argument("--tf-topics", nargs="*", default=["/tf","/tf_static"])
    ap.add_argument("--world-frame", default="map")
    ap.add_argument("--sync-tol", type=float, default=0.05)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-clouds", type=int, default=0)
    ap.add_argument("--percloud-voxel", dest="percloud_voxel", type=float, default=0.0,
                    help="Downsample each cloud BEFORE merging (meters).")
    ap.add_argument("--voxel", type=float, default=0.03, help="Final downsample after merge (meters).")
    ap.add_argument("--out", required=True)
    return ap

if __name__ == "__main__":
    parser=build_argparser(); sys.exit(run(parser.parse_args()))
