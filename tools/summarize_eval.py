"""Markdown summary of logs/eval/*.json (held-out test tracks)."""
import glob, json, os, sys

rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1] if len(sys.argv) > 1 else "logs/eval", "*.json"))):
    r = json.load(open(f))
    laps = [t["lap_mean_s"] for t in r["tracks"].values() if t["lap_mean_s"]]
    fails = [n for n, t in r["tracks"].items() if t["finished"] < t["robots"]]
    cond = "camera OFF" if r["camera_off"] else ("LiDAR OFF" if r["lidar_off"] else "normal")
    rows.append((r["name"], r["speed"], cond, r["success_rate"], r["fall_rate"], sum(laps) / len(laps) if laps else None,
                 ", ".join(fails) or "-"))
order = {"Teacher": 0, "A": 1, "B": 2, "C": 3}
rows.sort(key=lambda x: (order.get(x[0], 9), x[2] != "normal", x[2], x[1]))
print("| Model | Speed | Sensors | Success | Falls | Mean lap (finished) | Tracks not 100% |")
print("|---|---|---|---|---|---|---|")
for n, sp, c, s, fa, lap, fl in rows:
    print(f"| {n} | {sp:g}x | {c} | {s * 100:.0f}% | {fa * 100:.0f}% | {f'{lap:.1f} s' if lap else '-'} | {fl} |")
