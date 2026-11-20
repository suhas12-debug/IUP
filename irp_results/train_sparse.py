"""
Fine-tune OpenPCDet PointPillars (KITTI) with sparse-LiDAR augmentation. Run from OpenPCDet/tools:

  python ~/irp_results/train_sparse.py --cfg_file cfgs/kitti_models/pointpillar_sparse.yaml \
     --pretrained_model ~/LIdar_cam/pointpillar_7728.pth --extra_tag finetune_sparse \
     --batch_size 8 --epochs 20 --workers 4 --set OPTIMIZATION.LR 0.0005

Everything after the script name is passed to tools/train.py unchanged. Extra env vars:
  BEAM_PROBS="0.25,0.25,0.5"   probabilities for 64 / 32 / 16 beams per training sample
  ZSHIFT_MAX=0.8  ZSHIFT_PROB=0.5   simulate a lower sensor mount (metres)
Augmentation is applied AFTER OpenPCDet's own augmentations (GT-sampling, flip, rotate, scale), only in
training. Elevation angles are unchanged by flip/rotate/scale, so ring thinning stays consistent.
Validation inside train.py uses full 64-beam scans (augmentor is not called in eval).
"""
import os, sys, runpy
import numpy as np

import torch as _torch
_orig_load = _torch.load
def _load(*a, **k):
    k.setdefault("weights_only", False)   # our own checkpoints; PyTorch>=2.6 defaults to True
    return _orig_load(*a, **k)
_torch.load = _load

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_beams import subsample_beams

PROBS = np.array([float(x) for x in os.environ.get("BEAM_PROBS", "0.25,0.25,0.5").split(",")])
PROBS = PROBS / PROBS.sum()
ZMAX = float(os.environ.get("ZSHIFT_MAX", "0.8"))
ZPROB = float(os.environ.get("ZSHIFT_PROB", "0.5"))
BEAMS = [64, 32, 16]


def sparse_aug(data_dict):
    pts = data_dict["points"]
    dz = 0.0
    if ZMAX > 0 and np.random.rand() < ZPROB:
        dz = np.random.uniform(0.0, ZMAX)       # sensor lower => scene appears higher
        pts = pts.copy()
        pts[:, 2] += dz
        if "gt_boxes" in data_dict and len(data_dict["gt_boxes"]):
            data_dict["gt_boxes"] = data_dict["gt_boxes"].copy()
            data_dict["gt_boxes"][:, 2] += dz
    # Store dz so kitti_dataset.__getitem__ can compose it into lidar_aug_to_cam
    data_dict['z_shift_aug'] = dz
    b = int(np.random.choice(BEAMS, p=PROBS))
    data_dict["points"] = subsample_beams(pts, b)
    return data_dict


if __name__ == "__main__":
    from pcdet.datasets.augmentor.data_augmentor import DataAugmentor
    _orig = DataAugmentor.forward

    def forward(self, data_dict):
        return sparse_aug(_orig(self, data_dict))
    DataAugmentor.forward = forward
    print(f"[train_sparse] beams {BEAMS} probs {PROBS.round(3)}  zshift max {ZMAX} m prob {ZPROB}")
    sys.argv = ["train.py"] + sys.argv[1:]
    runpy.run_path("train.py", run_name="__main__")
