"""
Kaggle runner for the camera+LiDAR fusion project (OpenPCDet PointPillars + camera branch + sensor-agnostic fusion).
Pushed with:  kaggle kernels push -p <this folder>      (GPU + internet on, see kernel-metadata.json)

EDIT ONLY THE RUN SETTINGS BELOW between runs.
  RUN = "joint3"        -> fusion training, modality dropout 0.4/0.3/0.3, distillation on
  RUN = "control_lidar" -> same recipe, LiDAR only (probs 0,1,0)  = fair control
Training resumes automatically if checkpoints exist in output/.../<RUN>/ckpt (see RESUME_DATASET below).
"""
import os, sys, glob, shutil, subprocess, time, json

# ------------------------- RUN SETTINGS -------------------------
RUN = "joint3"
EPOCHS = 20
BATCH = 2
TIME_BUDGET_H = 8.3          # stop training before Kaggle's session limit; checkpoints are saved every epoch
RESUME_DATASET = "irp-ckpt"  # optional Kaggle dataset (slug) holding a previous <RUN>/ckpt folder; ignored if absent
DO_EVAL = True               # after finishing all epochs: eval both/lidar/camera x 64/32/16 beams
# ----------------------------------------------------------------
RUNS = {
    "joint3":        dict(probs="0.4,0.3,0.3", env=dict(LAMBDA_DISTILL="1", LAMBDA_DEPTH="1")),
    "control_lidar": dict(probs="0,1,0",       env=dict(LAMBDA_DISTILL="0", LAMBDA_DEPTH="1")),
}
assert RUN in RUNS, RUN

W = "/kaggle/working"
CODE = f"{W}/OpenPCDet"
IRP = f"{W}/irp_results"
INP = "/kaggle/input"


def sh(cmd, check=True, **kw):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, **kw)
    if check and r.returncode != 0:
        print(f"!! command failed with code {r.returncode}", flush=True)
        sys.exit(r.returncode)
    return r.returncode


def find_dir(root, name, must_have_parent=None, max_depth=6):
    """First directory called `name` under root (optionally whose parent is called must_have_parent)."""
    root = root.rstrip("/")
    for dp, dn, _ in os.walk(root):
        depth = dp[len(root):].count("/")
        if depth > max_depth:
            dn[:] = []
            continue
        for d in list(dn):
            if d == name and (must_have_parent is None or os.path.basename(dp) == must_have_parent):
                return os.path.join(dp, d)
        # never descend into huge leaf folders
        dn[:] = [d for d in dn if d not in ("velodyne", "image_2", "image_3", "label_2", "calib", "gt_database")]
    return None


print("=== 0. environment ===")
sh("nvidia-smi | head -15; python -V; python -c 'import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'; which nvcc; nvcc --version | tail -2", check=False)
print("input folders:", os.listdir(INP))

print("=== 1. locate inputs ===")
pcdet_src = None
for dp, dn, fn in os.walk(INP):
    if dp[len(INP):].count("/") > 4:
        dn[:] = []; continue
    if os.path.basename(dp) == "OpenPCDet" and "pcdet" in dn and "tools" in dn and "setup.py" in fn:
        pcdet_src = dp; break
    dn[:] = [d for d in dn if d not in ("data", "output", "build", ".git")]
irp_src = None
for dp, dn, fn in os.walk(INP):
    if dp[len(INP):].count("/") > 4:
        dn[:] = []; continue
    if "train_sparse.py" in fn and "joint_start.pth" in fn:
        irp_src = dp; break
    dn[:] = [d for d in dn if d not in ("data", "output", "build", ".git", "pcdet")]
print("OpenPCDet source:", pcdet_src)
print("irp_results source:", irp_src)
assert pcdet_src and irp_src, "could not find OpenPCDet code / irp_results (check the 'results' dataset is attached)"

kitti_roots = [os.path.join(INP, d) for d in os.listdir(INP)]
def locate(name):
    for r in kitti_roots:
        p = find_dir(r, name, must_have_parent="training")
        if p: return p
    return None
src = {n: locate(n) for n in ("velodyne", "image_2", "calib", "label_2")}
print("KITTI folders:", src)
missing = [k for k, v in src.items() if not v]
assert not missing, f"KITTI folder(s) not found under /kaggle/input: {missing}. Is 'object-detect' attached and does it have training/<name>?"

print("=== 2. copy code (no data/output/build) ===")
if os.path.exists(CODE): shutil.rmtree(CODE)
shutil.copytree(pcdet_src, CODE, ignore=shutil.ignore_patterns("data", "output", "build", ".git", "*.egg-info", "*.before_*", "__pycache__", "*.so"))
os.makedirs(f"{CODE}/data/kitti/training", exist_ok=True)
os.makedirs(f"{CODE}/output", exist_ok=True)
for n, p in src.items():
    dst = f"{CODE}/data/kitti/training/{n}"
    if not os.path.exists(dst): os.symlink(p, dst)
kd = os.path.join(pcdet_src, "data", "kitti")
shutil.copytree(f"{kd}/ImageSets", f"{CODE}/data/kitti/ImageSets", dirs_exist_ok=True)
for f in ("kitti_infos_train.pkl", "kitti_infos_val.pkl", "kitti_infos_trainval.pkl"):
    shutil.copy(f"{kd}/{f}", f"{CODE}/data/kitti/{f}")
if os.path.exists(IRP): shutil.rmtree(IRP)
os.makedirs(IRP)
for f in ("train_sparse.py", "eval_beams.py", "joint_start.pth", "final_table.py"):
    if os.path.exists(f"{irp_src}/{f}"): shutil.copy(f"{irp_src}/{f}", IRP)
print("sample check:", len(os.listdir(f"{CODE}/data/kitti/training/velodyne")), "velodyne files")

print("=== 3. python deps + compile pcdet CUDA ops ===")
sh("pip install -q easydict tensorboardX scikit-image shapely pyyaml tqdm numba pillow opencv-python-headless 2>&1 | tail -3", check=False)
import torch
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cc[0]}.{cc[1]}"
print("TORCH_CUDA_ARCH_LIST =", os.environ.get("TORCH_CUDA_ARCH_LIST"))
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
sh(f"cd {CODE} && python setup.py develop 2>&1 | tail -15")
sh(f"cd {CODE}/tools && python -c 'import pcdet; print(\"pcdet OK\", pcdet.__file__)'")

print("=== 4. resume checkpoints (optional) ===")
OUT = f"{CODE}/output/kitti_models/pointpillar_fusion/{RUN}"
os.makedirs(f"{OUT}/ckpt", exist_ok=True)
cands = glob.glob(f"{INP}/**/{RESUME_DATASET}/**/{RUN}/ckpt/checkpoint_epoch_*.pth", recursive=True) + \
        glob.glob(f"{INP}/{RESUME_DATASET}/**/checkpoint_epoch_*.pth", recursive=True)
if cands:
    for c in cands: shutil.copy(c, f"{OUT}/ckpt/")
    print("resuming from", sorted(os.listdir(f"{OUT}/ckpt")))
else:
    print("no resume checkpoints: starting from joint_start.pth")

print("=== 5. train ===")
cfg = RUNS[RUN]
env = dict(os.environ, FUSION_MODALITY_PROBS=cfg["probs"], **cfg["env"])
t0 = time.time()
train_cmd = (f"cd {CODE}/tools && timeout {int(TIME_BUDGET_H*3600)} python {IRP}/train_sparse.py "
             f"--cfg_file cfgs/kitti_models/pointpillar_fusion.yaml --batch_size {BATCH} --epochs {EPOCHS} "
             f"--extra_tag {RUN} --workers 4 --max_ckpt_save_num 3 --pretrained_model {IRP}/joint_start.pth "
             f"--set OPTIMIZATION.LR 0.0003")
rc = sh(train_cmd, check=False, env=env)
print(f"training exit code {rc} after {(time.time()-t0)/3600:.2f} h")
ck = sorted(glob.glob(f"{OUT}/ckpt/checkpoint_epoch_*.pth"), key=lambda p: int(p.split("_")[-1].split(".")[0]))
print("checkpoints:", [os.path.basename(c) for c in ck])
done = bool(ck) and int(ck[-1].split("_")[-1].split(".")[0]) >= EPOCHS

print("=== 6. evaluation ===")
if DO_EVAL and done:
    last = ck[-1]
    for b in (64, 32, 16):
        for m in ("both", "lidar", "camera"):
            e = dict(os.environ, MODALITY=m)
            sh(f"cd {CODE}/tools && python {IRP}/eval_beams.py --beams {b} --cfg cfgs/kitti_models/pointpillar_fusion.yaml --ckpt {last} 2>&1 | tail -5",
               check=False, env=e)
else:
    print("skipping eval (training not finished or DO_EVAL False). Re-run this script (with the saved checkpoints attached as dataset) to continue.")

print("=== 7. package results ===")
RES = f"{W}/results_{RUN}"
os.makedirs(RES, exist_ok=True)
for p in glob.glob(f"{CODE}/output/kitti_models/pointpillar_fusion/**/*.log", recursive=True) + \
         glob.glob(f"{CODE}/output/kitti_models/pointpillar_fusion/**/log_eval_*.txt", recursive=True):
    rel = os.path.relpath(p, f"{CODE}/output/kitti_models/pointpillar_fusion").replace("/", "__")
    shutil.copy(p, f"{RES}/{rel}")
shutil.copytree(f"{OUT}/ckpt", f"{RES}/ckpt", dirs_exist_ok=True)
if os.path.isdir(f"{OUT}/tensorboard"): shutil.copytree(f"{OUT}/tensorboard", f"{RES}/tensorboard", dirs_exist_ok=True)
# drop the big code copy from /kaggle/working so only results are returned as output
shutil.rmtree(CODE, ignore_errors=True); shutil.rmtree(IRP, ignore_errors=True)
print("done. outputs in", RES, os.listdir(RES))
