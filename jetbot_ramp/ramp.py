"""Elevated S-ramp geometry for the Jetbot ramp task.

Pure numpy/torch module: it does NOT import Isaac Lab or Isaac Sim, so it can be unit-tested
without launching the simulator.

Each environment holds two tracks side by side:

* track 0: an "S" (left U-turn, then right U-turn)
* track 1: the mirror image (right U-turn, then left U-turn)

On every reset the robot is placed on one of the two tracks, so the policy has to see both turn orders.
Driving a point-symmetric S in reverse gives the *same* turn order, so mirroring needs separate
geometry, which is why there are two tracks.

Each track is written as a closed triangle mesh (top, bottom, sides and end caps) in a generated
``.usda`` file that is used as a static collider. Thin pillars under the ramp are visual only.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class TrackSpec:
    """Geometry of one S track. Units are metres and degrees."""

    straight_in: float = 0.4
    """Length of the straight section at the start [m]."""
    radius: float = 0.6
    """Radius of both U-turns along the centreline [m]."""
    arc_deg: float = 180.0
    """Turn angle of each arc [deg]. 180 gives a letter-S."""
    straight_out: float = 0.4
    """Length of the straight section at the end [m]."""
    width: float = 0.26
    """Ramp width [m]. The Jetbot is about 0.12 m wide."""
    thickness: float = 0.03
    """Ramp slab thickness [m]."""
    top_z: float = 0.30
    """Height of the ramp's top surface above the ground [m]."""
    point_spacing: float = 0.01
    """Spacing of the centreline samples used for projection and the mesh [m]."""
    track_gap: float = 1.2
    """Clear gap between the two tracks' bounding boxes [m]."""
    pillar_spacing: float = 0.6
    """Distance between the visual support pillars [m]."""
    pillar_radius: float = 0.02
    """Radius of the visual support pillars [m]."""
    ramp_color: tuple[float, float, float] = (0.42, 0.27, 0.14)  # plywood tone: contrasts with the grey floor
    pillar_color: tuple[float, float, float] = (0.35, 0.35, 0.38)


@dataclass
class Track:
    """Sampled centreline of one track, in the environment's local frame."""

    points: np.ndarray  # (N, 2) xy
    yaw: np.ndarray  # (N,) tangent heading [rad]
    s: np.ndarray  # (N,) arc length from the start [m]

    @property
    def length(self) -> float:
        return float(self.s[-1])


# ---------------------------------------------------------------------------
# Centreline
# ---------------------------------------------------------------------------


def _pieces(spec: TrackSpec, mirror: bool) -> list[tuple[float, float]]:
    """(length, curvature) pieces. Positive curvature turns left."""
    arc_len = math.radians(spec.arc_deg) * spec.radius
    k = 1.0 / spec.radius
    sign = -1.0 if mirror else 1.0
    return [
        (spec.straight_in, 0.0),
        (arc_len, sign * k),
        (arc_len, -sign * k),
        (spec.straight_out, 0.0),
    ]


def build_centreline(spec: TrackSpec, mirror: bool = False) -> Track:
    """Sample the centreline exactly, integrating constant-curvature pieces in closed form.

    The track starts at the origin, heading +x.
    """
    pieces = _pieces(spec, mirror)
    total = sum(length for length, _ in pieces)
    n = int(math.ceil(total / spec.point_spacing)) + 1
    s_all = np.linspace(0.0, total, n)

    pts = np.zeros((n, 2))
    yaw = np.zeros(n)
    # piece start states
    x0, y0, th0, s0 = 0.0, 0.0, 0.0, 0.0
    starts = []
    for length, k in pieces:
        starts.append((s0, x0, y0, th0, length, k))
        if abs(k) < 1e-12:
            x0 += length * math.cos(th0)
            y0 += length * math.sin(th0)
        else:
            th1 = th0 + k * length
            x0 += (math.sin(th1) - math.sin(th0)) / k
            y0 += -(math.cos(th1) - math.cos(th0)) / k
            th0 = th1
        s0 += length

    for i, s in enumerate(s_all):
        # find the piece containing s (last piece takes s == total)
        for ps, px, py, pth, plen, pk in starts:
            if s <= ps + plen + 1e-12:
                break
        u = s - ps
        if abs(pk) < 1e-12:
            pts[i] = (px + u * math.cos(pth), py + u * math.sin(pth))
            yaw[i] = pth
        else:
            th = pth + pk * u
            pts[i] = (px + (math.sin(th) - math.sin(pth)) / pk, py - (math.cos(th) - math.cos(pth)) / pk)
            yaw[i] = th
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
    return Track(points=pts, yaw=yaw, s=s_all)


def _bbox(track: Track, half_width: float) -> tuple[np.ndarray, np.ndarray]:
    lo = track.points.min(axis=0) - half_width
    hi = track.points.max(axis=0) + half_width
    return lo, hi


def build_tracks(spec: TrackSpec) -> list[Track]:
    """Build [S, mirrored S], placed side by side along x and centred on the env origin."""
    a = build_centreline(spec, mirror=False)
    b = build_centreline(spec, mirror=True)
    hw = spec.width / 2
    a_lo, a_hi = _bbox(a, hw)
    b_lo, b_hi = _bbox(b, hw)
    # put b to the +x side of a with the requested gap, aligning the y-centres
    b_shift = np.array([a_hi[0] + spec.track_gap - b_lo[0], (a_lo[1] + a_hi[1]) / 2 - (b_lo[1] + b_hi[1]) / 2])
    b = Track(points=b.points + b_shift, yaw=b.yaw, s=b.s)
    # centre the pair's combined bbox on the origin
    lo = np.minimum(a_lo, b_lo + b_shift)
    hi = np.maximum(a_hi, b_hi + b_shift)
    centre = (lo + hi) / 2
    return [Track(points=t.points - centre, yaw=t.yaw, s=t.s) for t in (a, b)]


def footprint_size(spec: TrackSpec) -> tuple[float, float]:
    """Size (x, y) of the area both tracks occupy, used to choose the env spacing."""
    tracks = build_tracks(spec)
    hw = spec.width / 2
    lo = np.min([_bbox(t, hw)[0] for t in tracks], axis=0)
    hi = np.max([_bbox(t, hw)[1] for t in tracks], axis=0)
    return float(hi[0] - lo[0]), float(hi[1] - lo[1])


# ---------------------------------------------------------------------------
# USD writer (plain text, no pxr dependency)
# ---------------------------------------------------------------------------


def _slab_mesh(track: Track, spec: TrackSpec, stride: int = 2) -> tuple[np.ndarray, list[int], list[int]]:
    """Closed slab along the centreline. Returns (points (M,3), faceVertexCounts, faceVertexIndices).

    Faces are wound counter-clockwise seen from outside (USD's default right-handed orientation).
    """
    idx = list(range(0, len(track.s), stride))
    if idx[-1] != len(track.s) - 1:
        idx.append(len(track.s) - 1)
    c = track.points[idx]
    th = track.yaw[idx]
    nrm = np.stack([-np.sin(th), np.cos(th)], axis=-1)  # left normal
    hw = spec.width / 2
    zt, zb = spec.top_z, spec.top_z - spec.thickness
    L = c + hw * nrm
    R = c - hw * nrm
    n = len(idx)
    # vertex layout per sample i: 4*i + {0: Lt, 1: Rt, 2: Lb, 3: Rb}
    pts = np.zeros((4 * n, 3))
    pts[0::4, :2], pts[0::4, 2] = L, zt
    pts[1::4, :2], pts[1::4, 2] = R, zt
    pts[2::4, :2], pts[2::4, 2] = L, zb
    pts[3::4, :2], pts[3::4, 2] = R, zb

    def v(i: int, k: int) -> int:
        return 4 * i + k

    LT, RT, LB, RB = 0, 1, 2, 3
    faces: list[list[int]] = []
    for i in range(n - 1):
        j = i + 1
        faces.append([v(i, RT), v(j, RT), v(j, LT), v(i, LT)])  # top, normal +z
        faces.append([v(i, LB), v(j, LB), v(j, RB), v(i, RB)])  # bottom, normal -z
        faces.append([v(i, RB), v(j, RB), v(j, RT), v(i, RT)])  # right side, normal -left
        faces.append([v(i, LB), v(i, LT), v(j, LT), v(j, LB)])  # left side, normal +left
    faces.append([v(0, RB), v(0, RT), v(0, LT), v(0, LB)])  # start cap, normal -tangent
    e = n - 1
    faces.append([v(e, LB), v(e, LT), v(e, RT), v(e, RB)])  # end cap, normal +tangent
    counts = [4] * len(faces)
    indices = [k for f in faces for k in f]
    return pts, counts, indices


def _fmt_points(pts: np.ndarray) -> str:
    return ", ".join(f"({p[0]:.5f}, {p[1]:.5f}, {p[2]:.5f})" for p in pts)


def _fmt_color(c: tuple[float, float, float]) -> str:
    return f"[({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})]"


def ramp_usda_text(spec: TrackSpec) -> str:
    """USD (ASCII) for both tracks plus visual pillars. Default prim is ``/Ramp``."""
    tracks = build_tracks(spec)
    out = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Ramp"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "Ramp"',
        "{",
    ]
    for t_i, track in enumerate(tracks):
        pts, counts, indices = _slab_mesh(track, spec)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        out += [
            f'    def Xform "Track{t_i}"',
            "    {",
            '        def Mesh "slab" (',
            '            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]',
            "        )",
            "        {",
            f"            float3[] extent = [({lo[0]:.5f}, {lo[1]:.5f}, {lo[2]:.5f}), ({hi[0]:.5f}, {hi[1]:.5f}, {hi[2]:.5f})]",
            f"            int[] faceVertexCounts = [{', '.join(map(str, counts))}]",
            f"            int[] faceVertexIndices = [{', '.join(map(str, indices))}]",
            f"            point3f[] points = [{_fmt_points(pts)}]",
            f"            color3f[] primvars:displayColor = {_fmt_color(spec.ramp_color)}",
            '            uniform token subdivisionScheme = "none"',
            '            uniform token physics:approximation = "none"',
            "            bool physics:collisionEnabled = 1",
            "        }",
        ]
        # visual pillars (no collision) from the ground to the slab bottom
        pillar_h = spec.top_z - spec.thickness
        n_p = max(2, int(track.length // spec.pillar_spacing) + 1)
        for p_i, s in enumerate(np.linspace(0.05, track.length - 0.05, n_p)):
            k = int(np.searchsorted(track.s, s))
            x, y = track.points[min(k, len(track.s) - 1)]
            out += [
                f'        def Cylinder "pillar_{p_i:02d}"',
                "        {",
                '            uniform token axis = "Z"',
                f"            double height = {pillar_h:.5f}",
                f"            double radius = {spec.pillar_radius:.5f}",
                f"            color3f[] primvars:displayColor = {_fmt_color(spec.pillar_color)}",
                f"            double3 xformOp:translate = ({x:.5f}, {y:.5f}, {pillar_h / 2:.5f})",
                '            uniform token[] xformOpOrder = ["xformOp:translate"]',
                "        }",
            ]
        out.append("    }")
    out.append("}")
    out.append("")
    return "\n".join(out)


def spec_hash(spec: TrackSpec) -> str:
    return hashlib.sha1(repr(sorted(asdict(spec).items())).encode()).hexdigest()[:10]


def write_ramp_usd(spec: TrackSpec, directory: str | None = None) -> str:
    """Write the ramp USD for ``spec`` (content-addressed file name) and return its path."""
    directory = directory or os.path.join(os.path.expanduser("~"), ".cache", "jetbot_ramp")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"ramp_{spec_hash(spec)}.usda")
    if not os.path.exists(path):
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(ramp_usda_text(spec))
        os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# Batched torch helpers used by the environment
# ---------------------------------------------------------------------------


class TrackTensors:
    """All tracks stacked as tensors, for batched projection of robot positions onto the centreline."""

    def __init__(self, tracks: list[Track], device: str = "cpu"):
        import torch  # local import keeps the numpy-only parts importable without torch

        n = len(tracks[0].s)
        if any(len(t.s) != n for t in tracks):
            raise ValueError("All tracks must have the same number of centreline samples.")
        self.torch = torch
        self.device = device
        self.points = torch.tensor(np.stack([t.points for t in tracks]), dtype=torch.float32, device=device)  # (K,N,2)
        self.yaw = torch.tensor(np.stack([t.yaw for t in tracks]), dtype=torch.float32, device=device)  # (K,N)
        self.s = torch.tensor(np.stack([t.s for t in tracks]), dtype=torch.float32, device=device)  # (K,N)
        self.tangent = torch.stack([torch.cos(self.yaw), torch.sin(self.yaw)], dim=-1)  # (K,N,2)
        self.length = self.s[:, -1].clone()  # (K,)
        self.num_tracks = len(tracks)
        self.num_points = n
        self.spacing = float(tracks[0].s[1] - tracks[0].s[0])

    def project(self, track_id, pos_xy):
        """Project positions onto their track's centreline.

        Args:
            track_id: (E,) long tensor.
            pos_xy: (E, 2) positions in the env-local frame.

        Returns:
            idx (E,) nearest sample, s (E,) arc length, lateral (E,) signed offset (+ = left of the
            track direction), track_yaw (E,) tangent heading at the nearest sample.
        """
        torch = self.torch
        pts = self.points[track_id]  # (E,N,2)
        d2 = ((pts - pos_xy[:, None, :]) ** 2).sum(-1)  # (E,N)
        idx = torch.argmin(d2, dim=1)
        ar = torch.arange(len(track_id), device=pos_xy.device)
        p = pts[ar, idx]
        t = self.tangent[track_id, idx]
        rel = pos_xy - p
        along = (rel * t).sum(-1)
        s = (self.s[track_id, idx] + along).clamp(min=0.0)
        s = torch.minimum(s, self.length[track_id])
        lateral = t[:, 0] * rel[:, 1] - t[:, 1] * rel[:, 0]
        return idx, s, lateral, self.yaw[track_id, idx]

    def pose_at(self, track_id, s):
        """Centreline position (E,2) and heading (E,) at arc length ``s`` (E,)."""
        torch = self.torch
        f = (s / self.spacing).clamp(0, self.num_points - 1)
        i0 = f.floor().long().clamp(max=self.num_points - 2)
        w = (f - i0.float())[:, None]
        p = self.points[track_id, i0] * (1 - w) + self.points[track_id, i0 + 1] * w
        y0 = self.yaw[track_id, i0]
        dy = torch.atan2(torch.sin(self.yaw[track_id, i0 + 1] - y0), torch.cos(self.yaw[track_id, i0 + 1] - y0))
        return p, y0 + dy * w[:, 0]

    def lookahead_body(self, track_id, s, pos_xy, yaw, distances):
        """Centreline points ``distances`` metres ahead of ``s``, expressed in the robot's body frame.

        Returns (E, M, 2). Points past the end are clamped to the last sample.
        """
        torch = self.torch
        E, M = len(track_id), len(distances)
        d = torch.as_tensor(distances, dtype=torch.float32, device=pos_xy.device)
        s_q = (s[:, None] + d[None, :]).reshape(-1)
        tid = track_id[:, None].expand(E, M).reshape(-1)
        p, _ = self.pose_at(tid, s_q)
        rel = p.view(E, M, 2) - pos_xy[:, None, :]
        c, sn = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        x_b = c * rel[..., 0] + sn * rel[..., 1]
        y_b = -sn * rel[..., 0] + c * rel[..., 1]
        return torch.stack([x_b, y_b], dim=-1)
