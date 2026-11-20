"""
Diagnose why camera presence changes nothing. Run from OpenPCDet/tools:
  python ~/irp_results/diag_camera.py [checkpoint]
Prints: camera BEV statistics (is it empty / misplaced?), fused-feature difference between
MODALITY=both and MODALITY=lidar, and how the camera BEV overlaps the LiDAR BEV.
"""
import os, sys, torch
sys.path.insert(0, os.path.expanduser("~/OpenPCDet"))
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

CKPT = sys.argv[1] if len(sys.argv) > 1 else "../output/kitti_models/pointpillar_fusion/joint/ckpt/checkpoint_epoch_20.pth"
cfg_from_yaml_file("cfgs/kitti_models/pointpillar_fusion.yaml", cfg)
logger = common_utils.create_logger()
ds, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=2,
                                 dist=False, workers=0, logger=logger, training=False)
model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model_state"]
print("load_state_dict:", model.load_state_dict(sd, strict=False))
model.cuda().eval()

cap = {}
model.camera_branch.register_forward_hook(lambda m, i, o: cap.__setitem__("cam", o[0].detach().float()))
model.fusion.register_forward_hook(lambda m, i, o: cap.__setitem__("fused", o.detach().float()))
model.fusion.register_forward_hook(lambda m, i, o: cap.__setitem__("lidar_in", i[0].detach().float()))

def run(mode):
    os.environ["MODALITY"] = mode
    cap.clear()
    batch = next(iter(loader))            # same first batch each time (no shuffle)
    load_data_to_gpu(batch)
    with torch.no_grad():
        model(batch)
    return dict(cap), batch

both, batch = run("both")
lid, _ = run("lidar")
print("\nbatch keys with image:", [k for k in ("img_rgb", "K", "lidar_aug_to_cam", "depth_target") if k in batch])
print("img_rgb mean/std:", float(batch["img_rgb"].mean()), float(batch["img_rgb"].std()))
print("lidar_aug_to_cam[0]:\n", batch["lidar_aug_to_cam"][0].cpu().numpy().round(3))

cam = both.get("cam")
if cam is None:
    print("\n!! camera branch was NOT executed in 'both' mode (cam_present is False) -> bug in modality logic")
else:
    a = cam.abs()
    nz = (a.sum(1) > 1e-6).float().mean(dim=(1, 2))
    print(f"\ncamera BEV {tuple(cam.shape)}: mean|x|={float(a.mean()):.5f} max={float(a.max()):.3f} "
          f"| non-empty cells per sample: {[round(float(x),3) for x in nz]}")
    L = both["lidar_in"].abs().mean(1)    # (B,H,W) LiDAR activity
    C = a.mean(1)                          # (B,H,W) camera activity
    print(f"LiDAR BEV mean|x|={float(both['lidar_in'].abs().mean()):.4f}")
    # where is the camera BEV active, in metres (x forward = width axis, 0.32 m/cell)?
    cols = (C > 1e-6).float().sum(1).sum(0)           # (W,)
    xs = (cols > 0).nonzero().flatten()
    if len(xs):
        print(f"camera BEV active x-range: {float(xs.min())*0.32:.1f} - {float(xs.max())*0.32:.1f} m")
    # correlation between camera and lidar activity maps
    c = torch.corrcoef(torch.stack([L.flatten(), C.flatten()]))[0, 1]
    print(f"correlation(LiDAR activity, camera activity) = {float(c):.3f}")

d = (both["fused"] - lid["fused"]).abs()
print(f"\nfused output: mean|both - lidar| = {float(d.mean()):.6f}, max = {float(d.max()):.4f}, "
      f"mean|fused| = {float(both['fused'].abs().mean()):.4f}  (relative change {float(d.mean()/both['fused'].abs().mean())*100:.3f}%)")
