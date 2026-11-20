"""
Evaluate the pretrained PointPillars checkpoint on KITTI val with LiDAR thinned to 64/32/16 beams.
No retraining. Run from OpenPCDet/tools:

  python ~/irp_results/eval_beams.py --beams 32 --ckpt ~/LIdar_cam/pointpillar_7728.pth
  python ~/irp_results/eval_beams.py --beams 16 --ckpt ~/LIdar_cam/pointpillar_7728.pth

Ring assignment approximates the HDL-64E layout (the exact per-laser angles are not in the KITTI files):
  upper block: 32 lasers from +2.0 deg down in 1/3 deg steps; lower block: 32 lasers from -8.83 deg in 0.5 deg steps.
Each point -> nearest nominal laser; keep every (64/beams)-th laser. This is an approximation of a
real 32/16-beam sensor (a real VLP-16 also has a different vertical FOV, +-15 deg, and different mounting height).
"""
import os, sys, runpy, argparse
import numpy as np

import torch as _torch
_orig_load = _torch.load
def _load(*a, **k):
    k.setdefault("weights_only", False)   # our own checkpoints; PyTorch>=2.6 defaults to True
    return _orig_load(*a, **k)
_torch.load = _load

UPPER = 2.0 - np.arange(32) / 3.0
LOWER = -8.83 - np.arange(32) * 0.5
ANGLES = np.concatenate([UPPER, LOWER])            # 64 nominal laser elevations (deg)


def ring_index(points):
    elev = np.degrees(np.arctan2(points[:, 2], np.hypot(points[:, 0], points[:, 1])))
    return np.abs(elev[:, None] - ANGLES[None, :]).argmin(axis=1)


def subsample_beams(points, beams):
    if beams >= 64:
        return points
    k = 64 // beams
    return points[ring_index(points) % k == 0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--beams", type=int, required=True, choices=[64, 32, 16, 8])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", default="cfgs/kitti_models/pointpillar.yaml")
    ap.add_argument("--batch_size", type=int, default=4)
    a, extra = ap.parse_known_args()

    from pcdet.datasets.kitti.kitti_dataset import KittiDataset
    orig = KittiDataset.get_lidar
    n_in = n_out = 0

    def get_lidar(self, idx):
        global n_in, n_out
        pts = orig(self, idx)
        out = subsample_beams(pts, a.beams)
        n_in += len(pts); n_out += len(out)
        if idx == "000000" or (hasattr(idx, "__len__") and str(idx) == "000000"):
            print(f"[beams={a.beams}] frame 000000: {len(pts)} -> {len(out)} points")
        return out
    KittiDataset.get_lidar = get_lidar

    sys.argv = ["test.py", "--cfg_file", a.cfg, "--batch_size", str(a.batch_size),
                "--ckpt", a.ckpt, "--workers", "0", "--extra_tag", f"beams{a.beams}" + (("_" + os.environ["MODALITY"]) if os.environ.get("MODALITY") else "")]
    sys.argv += extra          # --set must come last (it swallows the rest)
    try:
        runpy.run_path("test.py", run_name="__main__")
    finally:
        if n_in:
            print(f"[beams={a.beams}] kept {n_out/n_in:.1%} of points overall")
