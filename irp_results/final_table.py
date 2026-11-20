"""
Final comparison table: fine-tuned LiDAR-only baseline vs the fusion model (both / lidar / camera), 64/32/16 beams.
Run:  python ~/irp_results/final_table.py      (writes ~/irp_results/final_table.md as well)
"""
import re, glob, os
OUT = os.path.expanduser("~/OpenPCDet/output/kitti_models")
EP = os.environ.get("EPOCH", "20")
STD = {"Car": "0.70, 0.70, 0.70", "Pedestrian": "0.50, 0.50, 0.50", "Cyclist": "0.50, 0.50, 0.50"}


def parse(path):
    txt = open(path, errors="ignore").read().replace("INFO  ", "")
    res = {}
    for cls, thr in STD.items():
        m = re.search(rf"{cls} AP_R40@{re.escape(thr)}:\s*\n.*?bev\s+AP:([\d.]+), ([\d.]+), ([\d.]+)\s*\n\s*3d\s+AP:([\d.]+), ([\d.]+), ([\d.]+)", txt, re.S)
        if m:
            v = [float(x) for x in m.groups()]
            res[cls] = {"bev": v[:3], "3d": v[3:]}
    return res


def load(pattern):
    for f in sorted(glob.glob(os.path.join(OUT, pattern)), reverse=True):
        r = parse(f)
        if r:
            return r
    return None


rows = []
for b in (64, 32, 16):
    rows.append((f"LiDAR-only baseline (fine-tuned), {b}-beam", f"pointpillar_sparse/beams{b}/eval/epoch_20/val/default/log_eval_*.txt"))
for b in (64, 32, 16):
    rows.append((f"Fusion, both sensors, {b}-beam", f"pointpillar_fusion/beams{b}_both/eval/epoch_{EP}/val/default/log_eval_*.txt"))
for b in (64, 32, 16):
    rows.append((f"Fusion, LiDAR only, {b}-beam", f"pointpillar_fusion/beams{b}_lidar/eval/epoch_{EP}/val/default/log_eval_*.txt"))
rows.append(("Fusion, camera only", f"pointpillar_fusion/beams64_camera/eval/epoch_{EP}/val/default/log_eval_*.txt"))

data = [(label, load(p)) for label, p in rows]
base = {b: data[i][1] for i, b in enumerate((64, 32, 16))}
lines = []
for kind, key in (("3D AP_R40", "3d"), ("BEV AP_R40", "bev")):
    lines.append(f"\n### {kind}: Easy / Moderate / Hard   (delta = Moderate vs LiDAR-only baseline at the same beam count)\n")
    lines.append("| Condition | Car | Pedestrian | Cyclist |")
    lines.append("|---|---|---|---|")
    for label, r in data:
        if r is None:
            lines.append(f"| {label} | (no log) | | |"); continue
        b = next((x for x in (64, 32, 16) if f"{x}-beam" in label), None)
        cells = []
        for c in STD:
            if c not in r:
                cells.append("n/a"); continue
            v = r[c][key]; txt = " / ".join(f"{x:.1f}" for x in v)
            if b and base.get(b) and c in base[b] and not label.startswith("LiDAR-only baseline"):
                d = v[1] - base[b][c][key][1]; txt += f"  ({d:+.1f})"
            cells.append(txt)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
out = "\n".join(lines)
print(out)
open(os.path.expanduser("~/irp_results/final_table.md"), "w").write(out + "\n")
