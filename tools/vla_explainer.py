"""Animated explainer (~60 s): our vision->action policy vs a VLM vs a VLA. Pure Pillow frames -> ffmpeg.

    <venv python> tools/vla_explainer.py <robot_frame.jpg> <camera_frame.npz or ''> videos/fusion/vla_explainer.mp4
"""

import math
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

robot_jpg, act_npz, out = sys.argv[1:4]
W, H, FPS = 1920, 1080, 30
BG = (21, 24, 29)
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
font = lambda p, s: ImageFont.truetype(p, s)  # noqa: E731
f18, f22, f26, f30 = (font(F, s) for s in (18, 22, 26, 30))
b22, b26, b30, b44 = (font(FB, s) for s in (22, 26, 30, 44))
m20 = font(FM, 20)
WHITE, GREY = (238, 238, 238), (150, 156, 166)
AMBER, GREEN, BLUE, PURPLE, RED = (242, 170, 70), (80, 195, 110), (85, 150, 240), (170, 120, 230), (230, 90, 80)

robot = Image.open(robot_jpg).convert("RGB")
cam = None
if act_npz:
    d = np.load(act_npz)
    c = np.clip(d["camera"][200].astype(np.float32).transpose(1, 2, 0) + 0.5, 0, 1)
    cam = Image.fromarray((c * 255).astype(np.uint8))


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def box(dr, xy, fill, text=None, f=b22, tc=WHITE, r=14, outline=None):
    dr.rounded_rectangle(xy, r, fill=fill, outline=outline, width=2)
    if text:
        dr.multiline_text(((xy[0] + xy[2]) / 2, (xy[1] + xy[3]) / 2), text, font=f, fill=tc, anchor="mm", align="center")


def arrow(dr, p0, p1, t, col=GREY, pulses=3, speed=0.6):
    dr.line([p0, p1], fill=(70, 74, 84), width=4)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    tip = [p1, (p1[0] - 16 * math.cos(ang - 0.4), p1[1] - 16 * math.sin(ang - 0.4)),
           (p1[0] - 16 * math.cos(ang + 0.4), p1[1] - 16 * math.sin(ang + 0.4))]
    dr.polygon(tip, fill=(70, 74, 84))
    for k in range(pulses):
        s = (t * speed + k / pulses) % 1.0
        x, y = p0[0] + (p1[0] - p0[0]) * s, p0[1] + (p1[1] - p0[1]) * s
        dr.ellipse([x - 7, y - 7, x + 7, y + 7], fill=col)


def title(dr, text, sub):
    dr.text((60, 40), text, font=b44, fill=WHITE)
    dr.text((60, 102), sub, font=f26, fill=GREY)


def fade_text(im, xy, text, f, col, a, anchor="la"):
    if a <= 0:
        return
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).text(xy, text, font=f, fill=col + (int(255 * min(a, 1)),), anchor=anchor)
    im.alpha_composite(layer)


# ------------------------------------------------------------------ scene 1: our policy (12 s)
def scene_ours(t):
    im = Image.new("RGBA", (W, H), BG + (255,))
    dr = ImageDraw.Draw(im)
    title(dr, "1. Our robot's brain: a small vision → action network", "Sensors in, two wheel speeds out, 30 times per second. No words anywhere.")
    if cam is not None:
        im.paste(cam.resize((320, 240), Image.NEAREST), (80, 220))
    dr.text((80, 470), "camera 80×60 (2 frames)", font=f22, fill=WHITE)
    for i in range(30):
        h = 30 + 60 * (0.5 + 0.5 * math.sin(i * 0.5 + t * 2))
        dr.rectangle([80 + i * 10, 640 - h, 88 + i * 10, 640], fill=GREEN if 8 < i < 22 else AMBER)
    dr.text((80, 650), "LiDAR 61 ranges + cliff 3 + proprio 5", font=f22, fill=WHITE)
    box(dr, (560, 260, 820, 420), BLUE, "CNN\n3 conv layers", b26)
    box(dr, (560, 560, 820, 680), (90, 96, 110), "normalise\n69 numbers", b22)
    box(dr, (960, 380, 1260, 560), PURPLE, "MLP\n1605→256→128", b26)
    box(dr, (1400, 400, 1640, 540), AMBER, "2 numbers\nleft, right wheel", b22, (25, 25, 25))
    arrow(dr, (410, 340), (555, 340), t, BLUE)
    arrow(dr, (400, 610), (555, 620), t, GREEN)
    arrow(dr, (825, 340), (955, 440), t, BLUE)
    arrow(dr, (825, 620), (955, 500), t, GREEN)
    arrow(dr, (1265, 470), (1395, 470), t, PURPLE)
    im.paste(robot.resize((240, 135)), (1660, 400))
    pts = [("≈ 450 thousand weights (numbers it learned)", 3), ("30 decisions per second on one consumer GPU", 4.5),
           ("one skill: drive along a ramp, learned by RL in simulation", 6), ("vision + LiDAR → action: a 'VA' model, no language", 7.5)]
    for i, (p, ts) in enumerate(pts):
        fade_text(im, (560, 780 + i * 52), "• " + p, f30, WHITE, (t - ts) * 2)
    return im


# ------------------------------------------------------------------ token scenes (VLM / VLA)
PATCH_N = 4


def token_scene(t, kind):
    im = Image.new("RGBA", (W, H), BG + (255,))
    dr = ImageDraw.Draw(im)
    if kind == "vlm":
        title(dr, "2. VLM: vision-language model", "Image + question in, words out. The 'brain' of chat assistants that can see.")
        prompt = ["What", "is", "on", "the", "ramp", "?"]
        out_tokens = ["A", "small", "green", "robot", "on", "a", "wooden", "ramp", "."]
    else:
        title(dr, "3. VLA: vision-language-action model", "Image + instruction in, robot actions out. A VLM taught to answer with motor commands.")
        prompt = ["Drive", "to", "the", "end", "of", "the", "ramp"]
        out_tokens = ["<act 201>", "<act 236>", "<act 203>", "<act 240>", "<act 199>", "<act 238>"]
    # image + patch grid
    img = robot.resize((480, 270))
    ix, iy = 80, 190
    im.paste(img, (ix, iy))
    grid_a = ease((t - 0.5) / 1.5)
    if grid_a > 0:
        for k in range(1, PATCH_N):
            dr.line([(ix + k * 120, iy), (ix + k * 120, iy + 270)], fill=(255, 255, 255), width=2)
            dr.line([(ix, iy + k * 67.5), (ix + 480, iy + k * 67.5)], fill=(255, 255, 255), width=2)
    dr.text((ix, iy + 285), "image cut into patches", font=f22, fill=GREY)
    # token row
    row_y = 820
    tok_w, gap = 62, 8
    x0 = 80
    n_img = PATCH_N * PATCH_N
    for k in range(n_img):
        s = ease((t - 2.0 - k * 0.08) / 1.0)
        px, py = ix + (k % PATCH_N) * 120 + 30, iy + (k // PATCH_N) * 67.5 + 15
        tx, ty = x0 + k * (tok_w + gap) * 0.62, row_y
        x, y = px + (tx - px) * s, py + (ty - py) * s
        if s > 0:
            patch = img.crop((int((k % PATCH_N) * 120), int((k // PATCH_N) * 67.5), int((k % PATCH_N) * 120 + 120), int((k // PATCH_N) * 67.5 + 67.5)))
            im.paste(patch.resize((36, 36)), (int(x), int(y)))
    dr.text((x0, row_y + 50), "image tokens", font=f22, fill=BLUE)
    # prompt typed + tokens
    tx0 = x0 + n_img * 44 + 30
    typed = int(max(0, (t - 3.0) * 3))
    dr.text((620, 250), "prompt:", font=f22, fill=GREY)
    dr.text((620, 280), " ".join(prompt[:typed]), font=b30, fill=WHITE)
    for k, w in enumerate(prompt[:typed]):
        bw = max(60, 16 * len(w) + 20)
        xx = tx0 + sum(max(60, 16 * len(q) + 20) + 8 for q in prompt[:k])
        box(dr, (xx, row_y, xx + bw, row_y + 40), (70, 80, 100), w, f18, WHITE, 8)
    dr.text((tx0, row_y + 50), "text tokens", font=f22, fill=AMBER)
    # transformer block with attention
    tb_a = ease((t - 5.0) / 1.0)
    if tb_a > 0:
        box(dr, (620, 400, 1500, 740), (40, 44, 58), None, outline=(110, 120, 150))
        dr.text((1060, 430), "Transformer  (billions of weights)", font=b30, fill=WHITE, anchor="mm")
        dr.text((1060, 470), "every token looks at every other token (attention), layer after layer", font=f22, fill=GREY, anchor="mm")
        rng = np.random.default_rng(int(t * 6))
        n_tok = n_img + len(prompt)
        xs = [x0 + 18 + k * 44 for k in range(n_img)] + [tx0 + 30 + sum(max(60, 16 * len(q) + 20) + 8 for q in prompt[:k]) for k in range(len(prompt))]
        for layer_i in range(4):
            ly = 700 - layer_i * 50
            dr.line([(640, ly), (1480, ly)], fill=(60, 66, 84), width=2)
        for _ in range(26):
            i, j = rng.integers(0, n_tok, 2)
            a = rng.uniform(0.3, 1)
            xi, xj = 640 + (xs[i] - 80) / (xs[-1] - 80 + 1e-6) * 840, 640 + (xs[j] - 80) / (xs[-1] - 80 + 1e-6) * 840
            ly = 700 - rng.integers(0, 4) * 50
            hgt = max(8.0, min(abs(xj - xi) / 3, ly - 515))
            dr.arc([min(xi, xj), ly - hgt, max(xi, xj) + 1, ly + hgt], 180, 360,
                   fill=(int(120 + 120 * a), int(150 + 60 * a), 255), width=2)
        for k in range(len(xs)):
            dr.line([(xs[k], row_y - 4), (640 + (xs[k] - 80) / (xs[-1] - 80 + 1e-6) * 840, 742)], fill=(60, 66, 84), width=1)
    # outputs, one token at a time
    out_start = 8.0
    n_out = int(max(0, (t - out_start) * 1.6))
    ox, oy = 1560, 400
    dr.text((ox, oy - 40), "output, one token at a time", font=f22, fill=GREY)
    for k, w in enumerate(out_tokens[:n_out]):
        col = GREEN if kind == "vlm" else RED
        bw = max(70, 15 * len(w) + 24)
        yy = oy + k * 46
        box(dr, (ox, yy, ox + bw, yy + 38), col, w, f18 if kind == "vlm" else m20, (20, 20, 20), 8)
    if n_out > 0 and n_out <= len(out_tokens):
        arrow(dr, (1505, 560), (1555, 420 + (n_out - 1) * 46), t, GREEN if kind == "vlm" else RED, 1, 1.2)
    if kind == "vlm" and n_out >= len(out_tokens):
        fade_text(im, (620, 960), "Answer: \"A small green robot on a wooden ramp.\"  Text only: it cannot move anything.", f30, WHITE, (t - 14) * 2)
    if kind == "vla":
        if n_out >= len(out_tokens):
            a = (t - 12.0) * 1.5
            fade_text(im, (620, 960), "decode action tokens → left wheel +0.58, right wheel +0.85 → the robot moves", f30, WHITE, a)
        fade_text(im, (620, 1010), "Other designs skip tokens and add an action head that outputs a short chunk of continuous actions (e.g. π₀).",
                  f22, GREY, (t - 13.5) * 2)
    return im


# ------------------------------------------------------------------ scene 4: comparison table (14 s)
ROWS = [
    ("", "Our policy (model C)", "VLM", "VLA"),
    ("Input", "camera + LiDAR", "image + text", "image + text instruction"),
    ("Output", "2 wheel speeds", "text", "robot actions"),
    ("Size", "≈ 0.45 M weights", "≈ 1–80 B weights", "≈ 1–55 B (e.g. OpenVLA 7 B)"),
    ("Trained on", "≈ 1 h of sim training (RL + teacher)", "web images + text", "VLM + many robot demos"),
    ("Speed", "30 Hz on one consumer GPU", "a few words/s to 100s", "≈ 5–50 Hz on big GPUs"),
    ("Can do", "one skill, very fast", "describe, answer, reason", "many tasks from words"),
    ("Language?", "no", "yes", "yes"),
]


def scene_table(t):
    im = Image.new("RGBA", (W, H), BG + (255,))
    dr = ImageDraw.Draw(im)
    title(dr, "4. Same idea, different scale", "All three map sensor input to an output with a neural network. What changes is size, data and what comes out.")
    cx = [80, 420, 900, 1360]
    for r, row in enumerate(ROWS):
        a = ease((t - 0.6 * r) / 0.6)
        if a <= 0:
            continue
        y = 210 + r * 92
        if r == 0:
            for c in range(1, 4):
                col = [None, PURPLE, GREEN, RED][c]
                box(dr, (cx[c], y, cx[c] + 440, y + 64), col, row[c], b26, (20, 20, 20) if c > 1 else WHITE, 12)
            continue
        dr.rounded_rectangle((60, y, 1860, y + 76), 10, fill=(30, 34, 42))
        fade_text(im, (cx[0] + 20, y + 38), row[0], b26, AMBER, a, "lm")
        for c in range(1, 4):
            fade_text(im, (cx[c] + 10, y + 38), row[c], f26, WHITE, a, "lm")
    fade_text(im, (60, 1000), "VLA examples: RT-2 (2023), OpenVLA (2024), π₀ (2024). Our model is tiny, task-specific and trained from scratch.",
              f22, GREY, (t - 6) * 1.5)
    return im


SCENES = [(scene_ours, 12), (lambda t: token_scene(t, "vlm"), 17), (lambda t: token_scene(t, "vla"), 17), (scene_table, 14)]
p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                      "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
n = 0
for fn, dur in SCENES:
    for k in range(int(dur * FPS)):
        t = k / FPS
        im = fn(t)
        fade = min(1.0, t / 0.4, (dur - t) / 0.4)
        arr = np.asarray(im.convert("RGB")).astype(np.float32)
        if fade < 1:
            arr = arr * fade + np.array(BG, np.float32) * (1 - fade)
        p.stdin.write(arr.astype(np.uint8).tobytes())
        n += 1
p.stdin.close()
p.wait()
print(f"[vla] wrote {out}: {n / FPS:.0f} s")
