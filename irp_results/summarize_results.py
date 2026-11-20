"""
Print R40 3D / BEV AP (Easy/Mod/Hard) at the standard KITTI thresholds for every eval log found.
Run:  python ~/irp_results/summarize_results.py
"""
import re, glob, os
OUT = os.path.expanduser("~/OpenPCDet/output/kitti_models")
RUNS = [  # (label, glob of eval logs)
    ("pretrained 64", "pointpillar/default/eval/epoch_7728/val/default/log_eval_*.txt"),
    ("pretrained 32", "pointpillar/beams32/eval/epoch_7728/val/default/log_eval_*.txt"),
    ("pretrained 16", "pointpillar/beams16/eval/epoch_7728/val/default/log_eval_*.txt"),
    ("fine-tuned 64", "pointpillar_sparse/beams64/eval/epoch_20/val/default/log_eval_*.txt"),
    ("fine-tuned 32", "pointpillar_sparse/beams32/eval/epoch_20/val/default/log_eval_*.txt"),
    ("fine-tuned 16", "pointpillar_sparse/beams16/eval/epoch_20/val/default/log_eval_*.txt"),
]
STD = {"Car": "0.70, 0.70, 0.70", "Pedestrian": "0.50, 0.50, 0.50", "Cyclist": "0.50, 0.50, 0.50"}


def parse(path):
    txt = open(path, errors="ignore").read()
    res = {}
    for cls, thr in STD.items():
        m = re.search(rf"{cls} AP_R40@{re.escape(thr)}:\s*\n.*?bev\s+AP:([\d.]+), ([\d.]+), ([\d.]+)\s*\n\s*3d\s+AP:([\d.]+), ([\d.]+), ([\d.]+)",
                      txt.replace("INFO  ", ""), re.S)
        if m:
            v = [float(x) for x in m.groups()]
            res[cls] = (v[:3], v[3:])
    return res


rows = []
for label, pat in RUNS:
    files = sorted(glob.glob(os.path.join(OUT, pat)))
    if not files:
        rows.append((label, None)); continue
    rows.append((label, parse(files[-1])))

for kind, idx in (("3D AP_R40  Easy / Mod / Hard", 1), ("BEV AP_R40 Easy / Mod / Hard", 0)):
    print(f"\n{kind}")
    print(f"{'run':16}" + "".join(f"{c:>26}" for c in STD))
    for label, r in rows:
        if r is None:
            print(f"{label:16}  (no log found)"); continue
        cells = []
        for c in STD:
            cells.append("   n/a" if c not in r else "  " + " / ".join(f"{x:5.1f}" for x in r[c][idx]))
        print(f"{label:16}" + "".join(f"{x:>26}" for x in cells))
