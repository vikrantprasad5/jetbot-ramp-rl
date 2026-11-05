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
    u = env
    q, pos = u._sensor_pose()
    yaw = float(u.yaw[idx])
    c, s = np.cos(yaw), np.sin(yaw)
    base = pos[idx].cpu().numpy()

    def body(hits, kind):
        rel = hits[idx].cpu().numpy() - base
        xb = c * rel[:, 0] + s * rel[:, 1]
        yb = -s * rel[:, 0] + c * rel[:, 1]
        k = kind[idx].cpu().numpy()
        return np.stack([xb, yb], -1), k == 1, k >= 0

    lb, lr, lv = body(u.lidar_hits, u.lidar_kind)
    cb, cr, cv = body(u.cliff_hits, u.cliff_kind)
    return lb, lr, lv, cb, cr & cv


def compose(viewer: str, out: str, cam: str | None = None, lidar: str | None = None, caption: str = "",
            stats_text: str = ""):
    """Viewer 1280x720 + optional camera inset (top-right quadrant) + optional LiDAR inset (bottom-right)."""
    inputs = ["-i", viewer]
    chains, last = [], "0:v"
    n = 1
    if cam:
        inputs += ["-i", cam]
        chains.append(f"[{n}:v]scale=340:255:flags=bicubic,pad=344:259:2:2:color=white[cam]")
        chains.append(f"[{last}][cam]overlay=W-w:0:shortest=1,drawtext=fontfile={FONT}:text='robot camera':x=w-336:y=232:"
                      f"fontsize=15:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=3[v{n}]")
        last = f"v{n}"
        n += 1
    if lidar:
        inputs += ["-i", lidar]
        y0 = 263 if cam else 0
        chains.append(f"[{n}:v]scale=240:240:flags=neighbor,pad=244:244:2:2:color=white[li]")
        chains.append(f"[{last}][li]overlay=W-w:{y0}:shortest=1,drawtext=fontfile={FONT}:text='LiDAR top view':x=w-238:"
                      f"y={y0 + 4}:fontsize=14:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=3[v{n}]")
        last = f"v{n}"
        n += 1
    tail = []
    if caption:
        tail.append(f"drawtext=fontfile={FONT}:expansion=none:text='{caption}':x=16:y=h-40:fontsize=24:fontcolor=white:box=1:"
                    f"boxcolor=black@0.55:boxborderw=6")
    if stats_text:
        tail.append(f"drawtext=fontfile={FONT}:expansion=none:text='{stats_text}':x=16:y=16:fontsize=22:fontcolor=white:box=1:"
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
