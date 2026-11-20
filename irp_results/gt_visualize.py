#!/usr/bin/env python3
"""
gt_visualize.py — Draw GT 3D boxes (projected) on 5 KITTI val frames.

Usage:
    python gt_visualize.py

Output:
    /home/cim/irp_results/gt_check/frame_XXXXXX.png  (5 images)
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
import cv2

# ── Paths ────────────────────────────────────────────────────────────────────
KITTI_INFO_VAL  = "/home/cim/OpenPCDet/data/kitti/kitti_infos_val.pkl"
IMAGE_DIR       = "/home/cim/LIdar_cam/dataset/data_object_image_2/training/image_2"
CALIB_DIR       = "/home/cim/LIdar_cam/dataset/data_object_calib/training/calib"
OUT_DIR         = "/home/cim/irp_results/gt_check"
NUM_FRAMES      = 5

os.makedirs(OUT_DIR, exist_ok=True)

# ── Calib reader ─────────────────────────────────────────────────────────────
def read_calib(calib_path):
    data = {}
    with open(calib_path) as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, val = line.split(':', 1)
            data[key.strip()] = np.array([float(x) for x in val.split()])
    P2 = data['P2'].reshape(3, 4)
    R0 = data['R0_rect'].reshape(3, 3)
    Tr = data['Tr_velo_to_cam'].reshape(3, 4)
    return P2, R0, Tr

# ── 3-D box corners in LiDAR frame ───────────────────────────────────────────
def boxes_to_corners_3d(boxes):
    """
    boxes: (N, 7) [x, y, z, dx, dy, dz, heading]  — LiDAR coords
    returns: (N, 8, 3) corners
    """
    x, y, z = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    dx, dy, dz = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    heading = boxes[:, 6]
    N = boxes.shape[0]

    # unit box corners (8, 3)
    corners_norm = np.array([
        [-0.5, -0.5, 0], [-0.5,  0.5, 0],
        [ 0.5,  0.5, 0], [ 0.5, -0.5, 0],
        [-0.5, -0.5, 1], [-0.5,  0.5, 1],
        [ 0.5,  0.5, 1], [ 0.5, -0.5, 1],
    ], dtype=np.float32)  # (8, 3)

    # scale
    corners = corners_norm[None] * np.stack([dx, dy, dz], axis=-1)[:, None, :]  # (N,8,3)

    # rotate around z-axis by heading
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    rot = np.zeros((N, 3, 3), dtype=np.float32)
    rot[:, 0, 0] =  cos_h
    rot[:, 0, 1] = -sin_h
    rot[:, 1, 0] =  sin_h
    rot[:, 1, 1] =  cos_h
    rot[:, 2, 2] = 1.0

    corners = np.einsum('nij,nkj->nki', rot, corners)  # (N,8,3)
    corners += np.stack([x, y, z], axis=-1)[:, None, :]  # translate
    return corners  # (N, 8, 3) in LiDAR

# ── Project corners to image ──────────────────────────────────────────────────
def project_to_image(corners_lidar, P2, R0, Tr):
    """
    corners_lidar: (N, 8, 3)
    Returns: (N, 8, 2) pixel coords
    """
    N = corners_lidar.shape[0]
    pts = corners_lidar.reshape(-1, 3)  # (N*8, 3)

    # LiDAR → cam0 (rectified)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])          # (N*8, 4)
    pts_cam = (Tr @ pts_h.T).T              # (N*8, 3)  Tr is 3x4
    pts_rect = (R0 @ pts_cam.T).T          # (N*8, 3)

    # cam0 → image (P2 is 3x4; append 1 for homogeneous)
    ones2 = np.ones((pts_rect.shape[0], 1), dtype=np.float32)
    pts_rect_h = np.hstack([pts_rect, ones2])  # (N*8, 4)
    pts_img_h = (P2 @ pts_rect_h.T).T          # (N*8, 3)

    pts_img = pts_img_h[:, :2] / pts_img_h[:, 2:3]  # (N*8, 2)

    # Depth filter: keep only points with z > 0 (in front of camera)
    depth = pts_img_h[:, 2].reshape(N, 8)
    pts_img = pts_img.reshape(N, 8, 2)
    return pts_img, depth  # (N,8,2), (N,8)

# ── Draw 3-D box on image ─────────────────────────────────────────────────────
# Edges: bottom face (0-3), top face (4-7), verticals (0-4,1-5,2-6,3-7)
EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]
CLASS_COLORS = {
    'Car':        (0,   255, 0),
    'Pedestrian': (255, 128, 0),
    'Cyclist':    (0,   128, 255),
}

def draw_boxes(img, corners_px, depths, names):
    """corners_px: (N,8,2) float; depths: (N,8) float"""
    for i in range(corners_px.shape[0]):
        # skip if any corner behind camera
        if np.any(depths[i] <= 0):
            continue
        color = CLASS_COLORS.get(names[i] if names is not None else '', (200, 200, 200))
        pts = corners_px[i].astype(int)  # (8,2)
        for s, e in EDGES:
            p1 = tuple(pts[s])
            p2 = tuple(pts[e])
            cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)
        # label at corner 4 (top-front-left)
        if names is not None:
            cv2.putText(img, names[i], tuple(pts[4]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(KITTI_INFO_VAL):
        print(f"ERROR: {KITTI_INFO_VAL} not found. Run info generation first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {KITTI_INFO_VAL} ...")
    with open(KITTI_INFO_VAL, 'rb') as f:
        infos = pickle.load(f)

    print(f"Val set has {len(infos)} frames. Visualising first {NUM_FRAMES}.")

    # Pick frames that have at least 1 annotation
    selected = []
    for info in infos:
        if 'annos' in info and len(info['annos']['name']) > 0:
            selected.append(info)
        if len(selected) >= NUM_FRAMES:
            break

    for idx, info in enumerate(selected):
        frame_id = info['point_cloud']['lidar_idx']
        img_path   = os.path.join(IMAGE_DIR, f"{frame_id}.png")
        calib_path = os.path.join(CALIB_DIR, f"{frame_id}.txt")

        if not os.path.exists(img_path):
            print(f"  WARN: image not found: {img_path}")
            continue
        if not os.path.exists(calib_path):
            print(f"  WARN: calib not found: {calib_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARN: could not read {img_path}")
            continue

        P2, R0, Tr = read_calib(calib_path)
        annos = info['annos']

        # GT boxes in LiDAR format (from info)
        # OpenPCDet stores gt_boxes_lidar in infos
        if 'gt_boxes_lidar' not in annos:
            print(f"  WARN: no gt_boxes_lidar in frame {frame_id}, skipping")
            continue

        gt_boxes = annos['gt_boxes_lidar']   # (N, 7)
        names    = annos['name']             # (N,)
        difficulty = annos.get('difficulty', ['?']*len(names))

        if gt_boxes.shape[0] == 0:
            print(f"  Frame {frame_id}: no GT boxes, skipping.")
            continue

        corners = boxes_to_corners_3d(gt_boxes)       # (N,8,3)
        corners_px, depths = project_to_image(corners, P2, R0, Tr)

        img = draw_boxes(img, corners_px, depths, names)

        # Overlay frame info
        cv2.putText(img, f"Frame {frame_id} | {len(gt_boxes)} GT boxes",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

        legend_y = 60
        for cls, col in CLASS_COLORS.items():
            cv2.putText(img, f"  {cls}", (10, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
            legend_y += 22

        out_path = os.path.join(OUT_DIR, f"frame_{frame_id}.png")
        cv2.imwrite(out_path, img)
        print(f"  Saved: {out_path}  (classes: {list(names)})")

    print(f"\nDone. Check {OUT_DIR}/")

if __name__ == "__main__":
    main()
