"""Top-down SVG/PNG preview of the train and test track libraries (no Isaac Sim).

    ~/isaacsim/kit/python/bin/python3 tools/preview_library.py   -> library_preview.svg (+ .png if ImageMagick)
"""

import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jetbot_ramp.tracks import TEST_LIBRARY, TRAIN_LIBRARY, build_library  # noqa: E402


def srgb(c):
    return "#" + "".join(f"{int(255 * min(1, max(0, x)) ** (1 / 2.2)):02x}" for x in c)


def panel(lib, title, x_off, scale, H):
    parts = []
    allp = np.concatenate([t.points for t in lib.tracks])
    lo, hi = allp.min(0) - 0.5, allp.max(0) + 0.5
    W = (hi[0] - lo[0]) * scale

    def xy(p):
        return x_off + (p[0] - lo[0]) * scale, 50 + (hi[1] - p[1]) * scale

    for t, s in zip(lib.tracks, lib.shapes):
        nrm = np.stack([-np.sin(t.yaw), np.cos(t.yaw)], -1)
        poly = np.concatenate([t.points + s.width / 2 * nrm, (t.points - s.width / 2 * nrm)[::-1]])
        parts.append(f'<polygon points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in map(xy, poly))}" fill="{srgb(s.color)}"/>')
        sx, sy = xy(t.points[0])
        parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="5" fill="#2a9d3a"/>')
        ex, ey = xy(t.points[-1])
        parts.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="5" fill="#1f5fbf"/>')
        label = s.name.replace("rand_", "r") + f" ({t.length:.1f} m, w {s.width * 100:.0f} cm)"
        parts.append(f'<text x="{sx + 8:.1f}" y="{sy + 16:.1f}" fill="#fff" font-size="12">{label}</text>')
    parts.append(f'<text x="{x_off + 10}" y="30" fill="#fff" font-size="20">{title}</text>')
    return parts, W


scale = 32
train, test = build_library(TRAIN_LIBRARY), build_library(TEST_LIBRARY)
p1, w1 = panel(train, f"TRAIN library: {len(train.tracks)} random tracks", 20, scale, 0)
p2, w2 = panel(test, f"TEST library (never trained on): {len(test.tracks)} tracks", 60 + w1, scale, 0)
H = 120 + max(np.ptp(np.concatenate([t.points for t in lib.tracks])[:, 1]) + 1 for lib in (train, test)) * scale
W = 80 + w1 + w2
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" font-family="sans-serif">',
       f'<rect width="{W:.0f}" height="{H:.0f}" fill="#3a3a3a"/>', *p1, *p2,
       f'<text x="20" y="{H - 15:.0f}" fill="#ddd" font-size="14">green = start, blue = finish. '
       f'Each environment holds the whole library; every reset puts the robot on a random track.</text>', "</svg>"]
out = os.path.join(os.path.dirname(__file__), "..", "library_preview.svg")
open(out, "w").write("\n".join(svg))
try:
    subprocess.run(["convert", out, out.replace(".svg", ".png")], check=True)
except Exception:  # noqa: BLE001
    pass
print(os.path.abspath(out))
