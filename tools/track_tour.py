"""One video touring every track of a library: title card (map + facts + result) then a camera-safe single-robot run
with camera and LiDAR insets (tools/pip_replay.py), all joined.

    ~/isaacsim/kit/python/bin/python3 tools/track_tour.py --library stress --task Isaac-JetbotTracks-Camera-Stress \
        --agent finetune_cfg_entry_point --ckpt <model.pt> --eval logs/stress/C_2x.json --speed 2 \
        --model "Model C" --out videos/fusion/stress_tour_C.mp4

Clip length per track comes from the evaluation JSON (lap time + 2 s if finished, else time to fall + 3 s).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from jetbot_ramp.tracks import STRESS_LIBRARY, TEST_LIBRARY, build_library, sample_pieces  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
WHY = {
    "long_wiggle_60m": "58 m of constant wiggles: 7-13x longer than training",
    "serpentine_120m": "133 m serpentine: tests error compounding over distance",
    "white_s": "ramp almost the colour of the floor: weak camera contrast",
    "rainbow_s": "8-colour stripes every 40 cm: never seen in training",
    "hairpins_r03": "0.3 m U-turns: tighter than the 0.4 m training minimum",
    "sharp_zigzag": "90 deg near-sharp corners: training had only smooth arcs",
    "narrow_18cm_s": "18 cm wide: 3 cm margin each side",
    "narrow_chicane": "22 cm wide with rapid tight turns",
}


def srgb(c):
    return "#" + "".join(f"{int(255 * min(1, max(0, x)) ** (1 / 2.2)):02x}" for x in c)


def track_map(shape, out_png):
    t = sample_pieces(shape.pieces, 0.01)
    nrm = np.stack([-np.sin(t.yaw), np.cos(t.yaw)], -1)
    allp = np.concatenate([t.points + shape.width / 2 * nrm, t.points - shape.width / 2 * nrm])
    lo, hi = allp.min(0), allp.max(0)
    W, H = 480, 360
    sc = min((W - 40) / (hi[0] - lo[0]), (H - 40) / (hi[1] - lo[1]))
    f = lambda p: (20 + (p[0] - lo[0]) * sc, 20 + (hi[1] - p[1]) * sc)  # noqa: E731
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="{W}" height="{H}" fill="#202020"/>']
    step = int(round(shape.stripe_len / 0.01)) if shape.stripes else len(t.s)
    for k, a in enumerate(range(0, len(t.s) - 1, step)):
        b = min(a + step + 1, len(t.s))
        poly = np.concatenate([t.points[a:b] + shape.width / 2 * nrm[a:b], (t.points[a:b] - shape.width / 2 * nrm[a:b])[::-1]])
        col = shape.stripes[k % len(shape.stripes)] if shape.stripes else shape.color
        svg.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in map(f, poly))}" fill="{srgb(col)}"/>')
    (sx, sy), (ex, ey) = f(t.points[0]), f(t.points[-1])
    svg += [f'<circle cx="{sx}" cy="{sy}" r="8" fill="#2a9d3a"/>', f'<circle cx="{ex}" cy="{ey}" r="8" fill="#1f5fbf"/></svg>']
    svg_path = out_png.replace(".png", ".svg")
    open(svg_path, "w").write("".join(svg))
    subprocess.run(["convert", svg_path, out_png], check=True)


def esc(t):
    return t.replace(":", "\\:").replace(",", "\\,").replace("'", "")


def card(out, l1, l2, l3, l4="", map_png=None, d=3.5):
    txt = [f"drawtext=fontfile={FONTB}:expansion=none:text='{esc(l1)}':x=60:y=210:fontsize=42:fontcolor=white",
           f"drawtext=fontfile={FONT}:expansion=none:text='{esc(l2)}':x=60:y=290:fontsize=26:fontcolor=0xdddddd",
           f"drawtext=fontfile={FONT}:expansion=none:text='{esc(l3)}':x=60:y=335:fontsize=24:fontcolor=0xf0c070"]
    if l4:
        txt.append(f"drawtext=fontfile={FONT}:expansion=none:text='{esc(l4)}':x=60:y=385:fontsize=24:fontcolor=0x9fd39f")
    if map_png:
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x202020:s=1280x720:d={d}:r=30", "-i", map_png,
               "-filter_complex", "[0:v][1:v]overlay=760:200," + ",".join(txt)]
    else:
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x202020:s=1280x720:d={d}:r=30", "-vf", ",".join(txt)]
    subprocess.run(cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", choices=["stress", "test"], default="stress")
    ap.add_argument("--task", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval", required=True, help="evaluate.py JSON at the same speed (for clip lengths and results)")
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--model", default="Model C")
    ap.add_argument("--title", default="Stress test: 8 very difficult tracks")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lib = build_library(STRESS_LIBRARY if a.library == "stress" else TEST_LIBRARY)
    ev = json.load(open(a.eval))
    work = os.path.join(os.path.dirname(os.path.abspath(a.out)), "_raw", os.path.splitext(os.path.basename(a.out))[0])
    os.makedirs(work, exist_ok=True)
    parts = []
    intro = os.path.join(work, "card_00.mp4")
    card(intro, a.title, f"{a.model} - camera + LiDAR - {a.speed:g}x speed", "None of these shapes or colours were in training", d=4)
    parts.append(intro)
    v_mps = 0.36 * a.speed
    for i, sh in enumerate(lib.shapes):
        r = ev["tracks"][sh.name]
        length = lib.tracks[i].length
        if r["finished"] == r["robots"]:
            res = f"all {r['robots']} test robots finished (mean lap {r['lap_mean_s']:.1f} s)"
        elif r["finished"]:
            res = f"{r['finished']}/{r['robots']} test robots finished"
        else:
            res = f"0/{r['robots']} finished - median {r['progress_median'] * 100:.0f}% of the track"
        if r["lap_mean_s"]:
            secs = r["lap_mean_s"] + 2.5
        else:
            secs = min(r["progress_median"] * length / (0.6 * v_mps) + 4.0, length / (0.6 * v_mps) + 4.0)
        secs = float(min(secs, 400.0))
        mp = os.path.join(work, f"map_{i + 1}.png")
        track_map(sh, mp)
        c = os.path.join(work, f"card_{i + 1:02d}.mp4")
        card(c, f"Track {i + 1} of {len(lib.shapes)} - {sh.name}", f"{length:.0f} m long, {sh.width * 100:.0f} cm wide",
             WHY.get(sh.name, ""), f"evaluation - {res}", mp)
        clip = os.path.join(work, f"run_{i + 1:02d}.mp4")
        cmd = [os.path.join(ROOT, "run.sh"), os.path.join(HERE, "pip_replay.py"), "--task", a.task, "--agent", a.agent,
               "--ckpt", a.ckpt, "--track", str(i), "--speed", str(a.speed), "--seconds", f"{secs:.1f}",
               "--caption", f"{a.model} - {sh.name} - {a.speed:g}x", "--out", clip]
        log = open(os.path.join(work, f"run_{i + 1:02d}.log"), "w")
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        line = [ln for ln in open(log.name) if ln.startswith("[pip] ") and "wrote" not in ln and "t=" not in ln]
        print(f"[tour] track {i + 1} {sh.name}: {line[-1].strip() if line else 'rc=' + str(rc)} ({secs:.0f} s clip)", flush=True)
        parts += [c, clip] if rc == 0 and os.path.exists(clip) else [c]
    lst = os.path.join(work, "list.txt")
    open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", "-crf", "21", "-r", "30", "-movflags", "+faststart", os.path.abspath(a.out)], check=True)
    print(f"[tour] wrote {a.out}")


if __name__ == "__main__":
    main()
