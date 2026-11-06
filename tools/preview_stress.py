"""Top-down preview of the stress-test tracks (stripes drawn), one tile per track -> stress_preview.png"""
import os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jetbot_ramp.tracks import STRESS_LIBRARY, build_library, sample_pieces  # noqa: E402

def srgb(c): return "#" + "".join(f"{int(255 * min(1, max(0, x)) ** (1 / 2.2)):02x}" for x in c)
lib = build_library(STRESS_LIBRARY)
TW, TH, cols = 420, 330, 4
rows = (len(lib.shapes) + cols - 1) // cols
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{TW * cols}" height="{TH * rows}" font-family="sans-serif">',
       f'<rect width="{TW * cols}" height="{TH * rows}" fill="#9a9a9a"/>']
for i, sh in enumerate(lib.shapes):
    t = sample_pieces(sh.pieces, 0.01)
    ox, oy = (i % cols) * TW, (i // cols) * TH
    nrm = np.stack([-np.sin(t.yaw), np.cos(t.yaw)], -1)
    allp = np.concatenate([t.points + sh.width / 2 * nrm, t.points - sh.width / 2 * nrm])
    lo, hi = allp.min(0), allp.max(0)
    sc = min((TW - 40) / (hi[0] - lo[0]), (TH - 70) / (hi[1] - lo[1]))
    f = lambda p: (ox + 20 + (p[0] - lo[0]) * sc, oy + 45 + (hi[1] - p[1]) * sc)  # noqa: E731
    svg.append(f'<rect x="{ox + 4}" y="{oy + 4}" width="{TW - 8}" height="{TH - 8}" fill="#c9c9c6" rx="10"/>')
    step = int(round(sh.stripe_len / 0.01)) if sh.stripes else len(t.s)
    for k, a in enumerate(range(0, len(t.s) - 1, step)):
        b = min(a + step + 1, len(t.s))
        L, R = t.points[a:b] + sh.width / 2 * nrm[a:b], t.points[a:b] - sh.width / 2 * nrm[a:b]
        poly = np.concatenate([L, R[::-1]])
        col = sh.stripes[k % len(sh.stripes)] if sh.stripes else sh.color
        svg.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in map(f, poly))}" fill="{srgb(col)}"/>')
    sx, sy = f(t.points[0]); ex, ey = f(t.points[-1])
    svg += [f'<circle cx="{sx}" cy="{sy}" r="6" fill="#1a8a2a"/>', f'<circle cx="{ex}" cy="{ey}" r="6" fill="#1f5fbf"/>',
            f'<text x="{ox + 16}" y="{oy + 30}" font-size="17" font-weight="bold" fill="#222">{i + 1}. {sh.name}  ({t.length:.0f} m, {sh.width * 100:.0f} cm)</text>']
svg.append("</svg>")
out = os.path.join(os.path.dirname(__file__), "..", "stress_preview.svg")
open(out, "w").write("\n".join(svg))
subprocess.run(["convert", out, out.replace(".svg", ".png")], check=True)
print(os.path.abspath(out.replace(".svg", ".png")))
