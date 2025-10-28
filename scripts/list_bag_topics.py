#!/usr/bin/env python3
"""
list_bag_topics.py — list topics & message types from bags (ROS1/ROS2).
- Accepts rosbag2 folders or *.db3/.bag files
- Works even when metadata.yaml is broken by opening *.db3 shards directly
- Provides default typestore when bag lacks embedded types
"""
import argparse
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

def expand_bags(paths):
    out = []
    for p in paths:
        P = Path(p)
        if P.is_dir():
            dbs = sorted(P.glob("*.db3"))
            out.extend(dbs if dbs else [P])
        else:
            out.append(P)
    return out

def open_reader_with_fallback(paths):
    order = [Stores.ROS2_IRON, Stores.ROS2_HUMBLE, Stores.ROS2_FOXY, Stores.ROS1_NOETIC]
    last = None
    for st in order:
        try:
            rdr = AnyReader(paths, default_typestore=get_typestore(st))
            rdr.open()
            return rdr
        except Exception as e:
            last = e
            continue
    raise last if last else RuntimeError("Failed to open with any typestore.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bags', nargs='+', required=True, help='Bag folders or files')
    args = ap.parse_args()

    paths = expand_bags(args.bags)
    reader = open_reader_with_fallback(paths)
    try:
        print("== Available topics ==")
        for c in sorted(reader.connections, key=lambda c: c.topic):
            print(f"{c.topic:45s}  {c.msgtype}")
        print("\nPick:")
        print("  --cloud-topic  <topic with sensor_msgs/msg/PointCloud2>")
        print("  --image-topic  <topic with sensor_msgs/msg/Image>")
        print("  --caminfo-topic <topic with sensor_msgs/msg/CameraInfo>")
    finally:
        try: reader.close()
        except Exception: pass

if __name__ == "__main__":
    main()
