"""Top-down SVG preview of both tracks (no Isaac Sim needed).

    ~/isaacsim/kit/python/bin/python3 ~/jetbot_ramp/tools/preview_ramp.py [out.svg]
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jetbot_ramp.ramp import TrackSpec, build_tracks  # noqa: E402

spec = TrackSpec()
tracks = build_tracks(spec)
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "ramp_preview.svg")
hw = spec.width / 2
scale, pad = 150.0, 0.4
allp = np.concatenate([t.points for t in tracks])
lo, hi = allp.min(0) - hw - pad, allp.max(0) + hw + pad
W, H = (hi - lo) * scale


def xy(p):
    return (p[0] - lo[0]) * scale, (hi[1] - p[1]) * scale


parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" font-family="sans-serif">',
         f'<rect width="{W:.0f}" height="{H:.0f}" fill="#3a3a3a"/>']
for k, t in enumerate(tracks):
    nrm = np.stack([-np.sin(t.yaw), np.cos(t.yaw)], -1)
    left, right = t.points + hw * nrm, t.points - hw * nrm
    poly = np.concatenate([left, right[::-1]])
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(xy, poly))
    parts.append(f'<polygon points="{pts}" fill="#c8c8c0" stroke="#222" stroke-width="1"/>')
    cl = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(xy, t.points))
    parts.append(f'<polyline points="{cl}" fill="none" stroke="#d04040" stroke-width="1.5" stroke-dasharray="6,4"/>')
    sx, sy = xy(t.points[0])
    ex, ey = xy(t.points[-1])
    parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="6" fill="#2a9d3a"/>')
    parts.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="6" fill="#1f5fbf"/>')
    name = "track 0: S (left, then right)" if k == 0 else "track 1: mirrored S (right, then left)"
    parts.append(f'<text x="{sx + 10:.1f}" y="{sy + 25:.1f}" fill="#fff" font-size="14">{name}, {t.length:.2f} m</text>')
ox, oy = xy((0.0, 0.0))
parts.append(f'<path d="M{ox - 8:.1f},{oy:.1f} H{ox + 8:.1f} M{ox:.1f},{oy - 8:.1f} V{oy + 8:.1f}" stroke="#ff0" stroke-width="2"/>')
parts.append(f'<text x="10" y="20" fill="#fff" font-size="13">green = start, blue = finish, red = centreline, yellow + = env origin. '
             f'width {spec.width} m, top {spec.top_z} m, U-turn radius {spec.radius} m</text>')
parts.append("</svg>")
with open(out, "w") as f:
    f.write("\n".join(parts))
print(os.path.abspath(out))
