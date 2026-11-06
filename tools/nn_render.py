"""Animate model C's network on a recorded drive (frames with Pillow, encoded with ffmpeg), next to the robot video.

    <venv python with numpy+Pillow> tools/nn_render.py logs/nn/C_triple_s_2x_act.npz videos/fusion/nn_run_C_triple_s_2x.mp4 \
        videos/fusion/nn_inside_model_C.mp4
"""

import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

act_path, robot_video, out = sys.argv[1:4]
D = np.load(act_path)
T = D["a"].shape[0]
W, H = 1920, 1080
BG = (21, 24, 29)
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
f12, f14, f16, f18, f22, f30 = (ImageFont.truetype(F, s) for s in (12, 14, 16, 18, 22, 30))
fb16, fb20, fb26, fb34 = (ImageFont.truetype(FB, s) for s in (16, 20, 26, 34))
ORANGE, BLUE, GREEN, PURPLE, GREY = (242, 160, 60), (80, 150, 240), (70, 190, 100), (170, 120, 230), (150, 156, 166)


def cmap(x):
    """0..1 -> dark purple / orange / yellow (magma-like)."""
    stops = np.array([[10, 8, 25], [90, 30, 110], [220, 90, 60], [250, 200, 80], [255, 250, 210]], float)
    x = np.clip(x, 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    w = (x - i)[..., None]
    return (stops[i] * (1 - w) + stops[i + 1] * w).astype(np.uint8)


def diverge(v):
    """-1..1 -> blue..dark..orange."""
    v = float(np.clip(v, -1, 1))
    base = np.array([45, 48, 56], float)
    tgt = np.array(ORANGE if v > 0 else BLUE, float)
    return tuple((base + (tgt - base) * abs(v)).astype(int))


# normalisers over the whole run
def scale(x, q=98):
    return np.percentile(np.abs(x), q) + 1e-6


s_h1, s_h2, s_in = scale(D["h1"]), scale(D["h2"]), np.percentile(D["node_in"], 98, axis=0) + 1e-6
s_c = [np.percentile(D[k].astype(np.float32), 99, axis=(0, 2, 3)) + 1e-6 for k in ("c1", "c2", "c3")]
names = [str(n) for n in D["in_names"]]
n_in, n1, n2 = len(names), D["h1"].shape[1], D["h2"].shape[1]

# geometry of the network panel
X_IN, X_H1, X_H2, X_OUT = 1110, 1420, 1660, 1840
TOP, BOT = 150, 1030


def ys(n):
    return np.linspace(TOP, BOT, n) if n > 1 else np.array([(TOP + BOT) / 2])


Y_IN, Y_H1, Y_H2, Y_OUT = ys(n_in), ys(n1), ys(n2), np.array([520.0, 700.0])
R_IN, R_H, R_OUT = 9, 10, 26


def edges(draw, x0, y0s, x1, y1s, contrib, top_k):
    """contrib: (n_to, n_from). Draw the top_k strongest |contribution| edges, orange = positive push, blue = negative."""
    flat = np.abs(contrib).ravel()
    if flat.max() <= 0:
        return
    idx = np.argsort(flat)[::-1][:top_k]
    m = flat[idx[0]]
    for k in idx[::-1]:
        to, fr = divmod(int(k), contrib.shape[1])
        v = contrib[to, fr]
        a = abs(v) / m
        col = ORANGE if v > 0 else BLUE
        draw.line([(x0, y0s[fr]), (x1, y1s[to])], fill=col + (int(40 + 200 * a),), width=max(1, int(1 + 4 * a)))


def heat_tile(arr, size):
    img = Image.fromarray(cmap(arr)).resize(size, Image.NEAREST)
    return img


def render(t):
    im = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(im)
    dr.text((30, 18), "Inside model C: what the neural network computes while it drives", font=fb34, fill=(240, 240, 240))
    dr.text((30, 62), "real inputs from the drive on the left, re-run through the trained network; every number shown is computed, not illustrative",
            font=f16, fill=GREY)
    # robot video placeholder frame (video is overlaid by ffmpeg at 30,100 size 800x450)
    dr.rectangle([28, 98, 832, 552], outline=(60, 64, 72), width=2)

    # ---- inputs row: camera, saliency, CNN maps
    y0 = 585
    cam = np.clip(D["camera"][t].astype(np.float32).transpose(1, 2, 0) + 0.5, 0, 1)
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((260, 195), Image.NEAREST)
    im.paste(cam_img, (30, y0 + 28))
    dr.text((30, y0), "camera input (80x60)", font=fb16, fill=(230, 230, 230))
    sal = D["sal"][t]
    sal = np.clip(sal / (np.percentile(sal, 99.5) + 1e-9), 0, 1)
    grey = cam.mean(-1, keepdims=True) * 0.6
    over = np.clip(grey + sal[..., None] * np.array([1.0, 0.35, 0.1]), 0, 1)
    im.paste(Image.fromarray((over * 255).astype(np.uint8)).resize((260, 195), Image.NEAREST), (305, y0 + 28))
    dr.text((305, y0), "where it looks (saliency)", font=fb16, fill=(230, 230, 230))
    dr.text((30, y0 + 228), "red = pixels that change the steering most", font=f14, fill=GREY)
    dr.text((580, y0), "CNN feature maps", font=fb16, fill=(230, 230, 230))
    for r, key in enumerate(("c1", "c2", "c3")):
        for c in range(4):
            m = D[key][t, c].astype(np.float32) / s_c[r][c]
            im.paste(heat_tile(m, (58, 44)), (580 + c * 63, y0 + 28 + r * 64))
        dr.text((580 + 4 * 63 + 2, y0 + 40 + r * 64), ["5x5", "3x3", "3x3"][r], font=f12, fill=GREY)
    dr.text((580, y0 + 228), "layer 1 / 2 / 3 (4 of 16-32 maps)", font=f14, fill=GREY)

    # ---- LiDAR ranges and action timeline
    y1 = 845
    dr.text((30, y1), "LiDAR (61 beams, left to right)", font=fb16, fill=(230, 230, 230))
    lid = D["lidar"][t]
    bw = 380 / len(lid)
    for i, v in enumerate(lid[::-1]):  # beam +60 deg (left) first
        h = 150 * float(v)
        col = GREEN if v < 0.3 else ORANGE
        dr.rectangle([30 + i * bw, y1 + 190 - h, 30 + (i + 1) * bw - 1, y1 + 190], fill=col)
    dr.text((30, y1 + 196), "green: hits the ramp (short)   orange: past the edge (long)", font=f14, fill=GREY)
    dr.text((450, y1), "wheel commands over the drive", font=fb16, fill=(230, 230, 230))
    a = np.clip(D["a"], -1, 1)
    px = lambda i: 450 + 380 * i / (T - 1)  # noqa: E731
    py = lambda v: y1 + 110 - 80 * v  # noqa: E731
    dr.line([(450, py(0)), (830, py(0))], fill=(70, 74, 82))
    for k, col in ((0, PURPLE), (1, GREEN)):
        pts = [(px(i), py(a[i, k])) for i in range(0, T, 2)]
        dr.line(pts, fill=col, width=2)
    dr.line([(px(t), y1 + 25), (px(t), y1 + 195)], fill=(240, 240, 240), width=2)
    dr.text((450, y1 + 196), "purple: left wheel   green: right wheel", font=f14, fill=GREY)

    # ---- network graph (edges on an RGBA layer)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    edges(ld, X_IN + R_IN, Y_IN, X_H1 - R_H, Y_H1, D["contrib_in"][t], 140)
    edges(ld, X_H1 + R_H, Y_H1, X_H2 - R_H, Y_H2, D["contrib_12"][t], 110)
    edges(ld, X_H2 + R_H, Y_H2, X_OUT - R_OUT, Y_OUT, D["contrib_2o"][t], 32)
    im = Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB")
    dr = ImageDraw.Draw(im)
    for hx, title in ((X_IN, "inputs"), (X_H1, "hidden 1"), (X_H2, "hidden 2"), (X_OUT, "wheels")):
        dr.text((hx, 105), title, font=fb20, fill=(235, 235, 235), anchor="mm")
    dr.text((X_IN, 128), "69 sensors + 1536 vision", font=f14, fill=GREY, anchor="mm")
    dr.text((X_H1, 128), f"{n1} of 256 neurons", font=f14, fill=GREY, anchor="mm")
    dr.text((X_H2, 128), f"{n2} of 128 neurons", font=f14, fill=GREY, anchor="mm")
    for i, (y, nm) in enumerate(zip(Y_IN, names)):
        v = float(D["node_in"][t, i] / s_in[i])
        col = tuple(int(c) for c in cmap(np.array([min(v, 1.0)]))[0])
        dr.ellipse([X_IN - R_IN, y - R_IN, X_IN + R_IN, y + R_IN], fill=col, outline=(90, 94, 102))
        tcol = (120, 200, 140) if nm.startswith("LiDAR") else ((240, 170, 90) if nm.startswith("vision") else (200, 200, 210))
        dr.text((X_IN - R_IN - 8, y), nm, font=f14, fill=tcol, anchor="rm")
    for X, Ys, key, s in ((X_H1, Y_H1, "h1", s_h1), (X_H2, Y_H2, "h2", s_h2)):
        for i, y in enumerate(Ys):
            dr.ellipse([X - R_H, y - R_H, X + R_H, y + R_H], fill=diverge(D[key][t, i] / s), outline=(90, 94, 102))
    for k, (y, nm) in enumerate(zip(Y_OUT, ["left wheel", "right wheel"])):
        v = float(a[t, k])
        dr.ellipse([X_OUT - R_OUT, y - R_OUT, X_OUT + R_OUT, y + R_OUT], fill=diverge(v), outline=(200, 200, 200), width=2)
        dr.text((X_OUT, y - R_OUT - 16), nm, font=fb16, fill=(235, 235, 235), anchor="mm")
        dr.text((X_OUT, y + R_OUT + 16), f"{v:+.2f}", font=f18, fill=(235, 235, 235), anchor="mm")
    steer = a[t, 1] - a[t, 0]
    msg = "turning LEFT" if steer > 0.15 else ("turning RIGHT" if steer < -0.15 else "straight")
    dr.text((X_OUT, 820), msg, font=fb20, fill=(250, 220, 120), anchor="mm")
    dr.text((X_OUT, 846), f"speed {(a[t, 0] + a[t, 1]) / 2:+.2f}", font=f16, fill=GREY, anchor="mm")
    # legend
    dr.text((1250, 1048), "edges: orange pushes the next neuron up, blue pushes it down; thicker = stronger (weight x activation)",
            font=f14, fill=GREY, anchor="mm")
    dr.text((W - 30, 18), f"t = {t / 30:5.2f} s", font=f22, fill=GREY, anchor="ra")
    return im


tmp = out.replace(".mp4", "_frames.mp4")
p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "30", "-i", "-",
                      "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", tmp], stdin=subprocess.PIPE)
for t in range(T):
    p.stdin.write(np.asarray(render(t)).tobytes())
    if t % 100 == 0:
        print(f"[nn_render] frame {t}/{T}", flush=True)
p.stdin.close()
p.wait()
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-i", robot_video, "-filter_complex",
                "[1:v]scale=800:450[r];[0:v][r]overlay=30:100:shortest=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
                "-movflags", "+faststart", out], check=True)
print(f"[nn_render] wrote {out}")
