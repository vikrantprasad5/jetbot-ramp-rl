"""Video helpers: LiDAR top-down plot frames (numpy -> ffmpeg) and multi-inset composition.

No PIL/matplotlib needed (Isaac Sim's python has neither).
"""

from __future__ import annotations

import os
import subprocess

import numpy as np

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


class LidarPlotWriter:
    """Robot-centric top-down plot of LiDAR (and cliff) hits, one frame per policy step.

    Forward is up. Green = hit on the ramp surface, orange = hit below the ramp (floor / pillar),
    red ticks at max range = no return (drop-off with nothing within range).
    """

    def __init__(self, path: str, size: int = 320, fps: int = 30, fwd_range=(-0.12, 0.6), half_width: float = 0.36):
        self.path, self.S, self.fps = path, size, fps
        self.fwd_range, self.half_width = fwd_range, half_width
        self.frames: list[np.ndarray] = []

    def _px(self, xb: np.ndarray, yb: np.ndarray):
        S = self.S
        f0, f1 = self.fwd_range
        u = ((-yb + self.half_width) / (2 * self.half_width) * (S - 1)).astype(int)  # left of robot = left of image
        v = ((f1 - xb) / (f1 - f0) * (S - 1)).astype(int)
        return u, v

    def add(self, hits_b: np.ndarray, on_ramp: np.ndarray, valid: np.ndarray, cliff_b: np.ndarray | None = None,
            cliff_ok: np.ndarray | None = None):
        S = self.S
        img = np.full((S, S, 3), 28, np.uint8)
        # grid every 0.25 m forward
        for d in np.arange(0.0, self.fwd_range[1] + 1e-6, 0.1):
            _, v = self._px(np.array([d]), np.array([0.0]))
            if 0 <= v[0] < S:
                img[v[0], :, :] = 55
        # robot footprint (~0.14 x 0.13 m) at the origin
        u0, v0 = self._px(np.array([0.07, -0.07]), np.array([0.065, -0.065]))
        img[min(v0):max(v0) + 1, min(u0):max(u0) + 1] = (60, 200, 60)

        def dot(xb, yb, color, r=2):
            u, v = self._px(xb, yb)
            for uu, vv in zip(u, v):
                if 0 <= uu < S and 0 <= vv < S:
                    img[max(0, vv - r):vv + r + 1, max(0, uu - r):uu + r + 1] = color

        dot(hits_b[valid & on_ramp, 0], hits_b[valid & on_ramp, 1], (90, 230, 90))
        dot(hits_b[valid & ~on_ramp, 0], hits_b[valid & ~on_ramp, 1], (255, 150, 40))
        if cliff_b is not None:
            dot(cliff_b[cliff_ok, 0], cliff_b[cliff_ok, 1], (80, 160, 255), r=3)
            dot(cliff_b[~cliff_ok, 0], cliff_b[~cliff_ok, 1], (255, 0, 255), r=3)
        self.frames.append(img)

    def close(self):
        if not self.frames:
            return None
        S = self.S
        p = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{S}x{S}", "-r", str(self.fps),
             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", self.path],
            stdin=subprocess.PIPE,
        )
        for f in self.frames:
            p.stdin.write(f.tobytes())
        p.stdin.close()
        p.wait()
        return self.path


def lidar_sample(env, idx: int = 0):
    """Body-frame LiDAR / cliff hit points of env ``idx`` (numpy), for LidarPlotWriter.add."""
    import torch

    u = env
    yaw = float(u.yaw[idx])
    base = u.robot.data.root_pos_w.torch[idx]
    c, s = np.cos(yaw), np.sin(yaw)
    top = u.scene.env_origins[idx, 2] + u.cfg.library.top_z

    def body(sensor, max_range):
        hits = sensor.data.ray_hits_w.torch[idx]
        origin = sensor.data.pos_w.torch[idx]
        finite = torch.isfinite(hits).all(-1)
        d = (hits - origin).norm(dim=-1)
        valid = finite & (d < max_range * 0.999)
        h = torch.where(valid[:, None], hits, origin.expand_as(hits))  # misses are not drawn
        rel = (h - base).cpu().numpy()
        xb = c * rel[:, 0] + s * rel[:, 1]
        yb = -s * rel[:, 0] + c * rel[:, 1]
        on_ramp = (torch.abs(hits[:, 2] - top) < 0.05).cpu().numpy()
        return np.stack([xb, yb], -1), on_ramp, valid.cpu().numpy()

    lb, lr, lv = body(u._lidar, 3.0)
    cb = cok = None
    if u._cliff is not None:
        cb, cok, cv = body(u._cliff, 1.0)
        cok = cok & cv
    return lb, lr, lv, cb, cok


def compose(viewer: str, out: str, cam: str | None = None, lidar: str | None = None, caption: str = "",
            stats_text: str = ""):
    """Viewer 1280x720 + optional camera inset (top-right quadrant) + optional LiDAR inset (bottom-right)."""
    inputs = ["-i", viewer]
    chains, last = [], "0:v"
    n = 1
    if cam:
        inputs += ["-i", cam]
        chains.append(f"[{n}:v]scale=480:360:flags=bicubic,pad=486:366:3:3:color=white[cam]")
        chains.append(f"[{last}][cam]overlay=W-w-9:9:shortest=1,drawtext=fontfile={FONT}:text='robot camera':x=W-480:y=340:"
                      f"fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=4[v{n}]")
        last = f"v{n}"
        n += 1
    if lidar:
        inputs += ["-i", lidar]
        chains.append(f"[{n}:v]scale=280:280:flags=neighbor,pad=286:286:3:3:color=white[li]")
        y = "H-h-60"
        chains.append(f"[{last}][li]overlay=W-w-9:{y}:shortest=1,drawtext=fontfile={FONT}:text='LiDAR top view (grid 10 cm)':x=w-283:"
                      f"y=h-340:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=4[v{n}]")
        last = f"v{n}"
        n += 1
    tail = []
    if caption:
        tail.append(f"drawtext=fontfile={FONT}:text='{caption}':x=16:y=h-40:fontsize=24:fontcolor=white:box=1:"
                    f"boxcolor=black@0.55:boxborderw=6")
    if stats_text:
        tail.append(f"drawtext=fontfile={FONT}:text='{stats_text}':x=16:y=16:fontsize=22:fontcolor=white:box=1:"
                    f"boxcolor=black@0.55:boxborderw=6")
    filt = ";".join(chains)
    if tail:
        filt = (filt + ";" if filt else "") + f"[{last}]" + ",".join(tail) + "[vout]"
        last = "vout"
    cmd = ["ffmpeg", "-v", "error", "-y", *inputs]
    if filt:
        cmd += ["-filter_complex", filt, "-map", f"[{last}]"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    return out


def concat(parts: list[str], out: str):
    lst = out + ".txt"
    with open(lst, "w") as f:
        for p in parts:
            f.write(f"file '{os.path.abspath(p)}'\n")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "21", "-movflags", "+faststart", out], check=True)
    os.remove(lst)
    return out
