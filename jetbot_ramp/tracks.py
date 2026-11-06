"""Procedurally generated elevated tracks ("track library") for generalization experiments.

Pure numpy/torch: no Isaac Lab imports, testable offline.

A track is a sequence of constant-curvature pieces (straights and arcs, either direction) with its
own width and colour. Each environment holds a *library* of tracks laid out on a grid; every reset puts
the robot on a random one. Training and test libraries come from different seeds, plus a few named
test tracks (original S, triple S, long straight + S, tight S, narrow S).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from .ramp import Track

# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


@dataclass
class TrackShape:
    """One track: pieces of (length [m], curvature [1/m], + = left) plus width and colour."""

    pieces: list[tuple[float, float]]
    width: float = 0.26
    color: tuple[float, float, float] = (0.42, 0.27, 0.14)
    name: str = ""
    stripes: list | None = None
    """Optional list of colours used in turn along the track (multi-colour ramps)."""
    stripe_len: float = 0.4


@dataclass
class GeneratorSpec:
    """Ranges for random tracks."""

    start_straight: tuple[float, float] = (0.35, 0.6)
    end_straight: tuple[float, float] = (0.3, 0.5)
    num_middle: tuple[int, int] = (3, 7)
    straight_prob: float = 0.35
    straight_len: tuple[float, float] = (0.3, 2.0)
    arc_radius: tuple[float, float] = (0.4, 1.2)
    arc_deg: tuple[float, float] = (40.0, 180.0)
    width: tuple[float, float] = (0.22, 0.32)
    max_length: float = 9.0
    min_length: float = 3.5
    min_gap: float = 0.35
    """Minimum clear gap between non-adjacent parts of a track (and between tracks) [m]."""


@dataclass
class LibrarySpec:
    """A library of tracks shared by every environment."""

    seed: int = 0
    num_random: int = 12
    named: list[str] = field(default_factory=list)
    """Named tracks to include (see NAMED_TRACKS)."""
    generator: GeneratorSpec = field(default_factory=GeneratorSpec)
    thickness: float = 0.03
    top_z: float = 0.30
    point_spacing: float = 0.01
    cell_gap: float = 1.2
    """Gap between neighbouring tracks' bounding boxes in the grid layout [m]."""
    pillar_spacing: float = 0.6
    pillar_radius: float = 0.02
    pillar_color: tuple[float, float, float] = (0.35, 0.35, 0.38)


# plywood / painted-wood palette (linear RGB), all clearly darker than the light grey floor
PALETTE = [
    (0.42, 0.27, 0.14),
    (0.30, 0.18, 0.09),
    (0.55, 0.38, 0.20),
    (0.12, 0.22, 0.40),
    (0.35, 0.10, 0.08),
    (0.10, 0.28, 0.14),
    (0.20, 0.20, 0.22),
    (0.50, 0.30, 0.05),
]


def _arc(radius: float, deg: float, left: bool) -> tuple[float, float]:
    k = 1.0 / radius
    return (math.radians(deg) * radius, k if left else -k)


NAMED_TRACKS: dict[str, TrackShape] = {
    "s_original": TrackShape([(0.4, 0), _arc(0.6, 180, True), _arc(0.6, 180, False), (0.4, 0)], 0.26, PALETTE[0]),
    "triple_s": TrackShape(
        [(0.4, 0), _arc(0.6, 180, True), _arc(0.6, 180, False), _arc(0.6, 180, True), _arc(0.6, 180, False),
         _arc(0.6, 180, True), _arc(0.6, 180, False), (0.4, 0)],
        0.26, PALETTE[2],
    ),
    "long_straight_s": TrackShape(
        [(2.5, 0), _arc(0.6, 180, False), _arc(0.6, 180, True), (1.0, 0)], 0.26, PALETTE[3]
    ),
    "tight_s": TrackShape([(0.4, 0), _arc(0.4, 180, True), _arc(0.4, 180, False), (0.4, 0)], 0.26, PALETTE[4]),
    "narrow_s": TrackShape([(0.4, 0), _arc(0.6, 180, True), _arc(0.6, 180, False), (0.4, 0)], 0.20, PALETTE[5]),
}

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def sample_pieces(pieces: list[tuple[float, float]], spacing: float) -> Track:
    """Exact sampling of a constant-curvature piece sequence starting at the origin heading +x."""
    total = sum(p[0] for p in pieces)
    n = int(math.ceil(total / spacing)) + 1
    s_all = np.linspace(0.0, total, n)
    starts = []
    x0 = y0 = th0 = s0 = 0.0
    for length, k in pieces:
        starts.append((s0, x0, y0, th0, length, k))
        if abs(k) < 1e-12:
            x0 += length * math.cos(th0)
            y0 += length * math.sin(th0)
        else:
            th1 = th0 + k * length
            x0 += (math.sin(th1) - math.sin(th0)) / k
            y0 -= (math.cos(th1) - math.cos(th0)) / k
            th0 = th1
        s0 += length
    pts = np.zeros((n, 2))
    yaw = np.zeros(n)
    piece_idx = 0
    for i, s in enumerate(s_all):
        while piece_idx < len(starts) - 1 and s > starts[piece_idx][0] + starts[piece_idx][4] + 1e-12:
            piece_idx += 1
        ps, px, py, pth, _, pk = starts[piece_idx]
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


def self_clearance(track: Track, width: float, stride: int = 3) -> float:
    """Smallest edge-to-edge gap between parts of the track more than ~(width + 0.6 m) apart along s."""
    p = track.points[::stride]
    s = track.s[::stride]
    d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    far_along = np.abs(s[:, None] - s[None, :]) > (width + 0.6)
    # neighbours along a tight U-turn are close in space but also close along s; only count far ones
    if not far_along.any():
        return float("inf")
    return float(d[far_along].min() - width)


def generate_random_track(rng: np.random.Generator, gen: GeneratorSpec, spacing: float, max_tries: int = 500) -> TrackShape:
    for _ in range(max_tries):
        pieces = [(rng.uniform(*gen.start_straight), 0.0)]
        for _ in range(rng.integers(gen.num_middle[0], gen.num_middle[1] + 1)):
            if rng.random() < gen.straight_prob and pieces[-1][1] != 0.0:
                pieces.append((rng.uniform(*gen.straight_len), 0.0))
            else:
                pieces.append(_arc(rng.uniform(*gen.arc_radius), rng.uniform(*gen.arc_deg), bool(rng.random() < 0.5)))
        pieces.append((rng.uniform(*gen.end_straight), 0.0))
        total = sum(p[0] for p in pieces)
        if not (gen.min_length <= total <= gen.max_length):
            continue
        width = float(rng.uniform(*gen.width))
        track = sample_pieces(pieces, spacing)
        if self_clearance(track, width) < gen.min_gap:
            continue
        color = PALETTE[int(rng.integers(len(PALETTE)))]
        return TrackShape(pieces=[(float(a), float(b)) for a, b in pieces], width=width, color=color)
    raise RuntimeError("Could not generate a non-self-intersecting track; relax GeneratorSpec.")


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------


@dataclass
class Library:
    shapes: list[TrackShape]
    tracks: list[Track]  # centrelines in the env-local frame (laid out on the grid)
    spec: LibrarySpec

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.shapes]

    def footprint(self) -> tuple[float, float]:
        lo = np.min([t.points.min(0) - s.width / 2 for t, s in zip(self.tracks, self.shapes)], axis=0)
        hi = np.max([t.points.max(0) + s.width / 2 for t, s in zip(self.tracks, self.shapes)], axis=0)
        return float(hi[0] - lo[0]), float(hi[1] - lo[1])


def build_library(spec: LibrarySpec) -> Library:
    rng = np.random.default_rng(spec.seed)
    shapes: list[TrackShape] = []
    for name in spec.named:
        s = NAMED_TRACKS[name] if name in NAMED_TRACKS else STRESS_TRACKS[name]
        shapes.append(TrackShape(list(s.pieces), s.width, s.color, name, s.stripes, s.stripe_len))
    for i in range(spec.num_random):
        s = generate_random_track(rng, spec.generator, spec.point_spacing)
        s.name = f"rand_{spec.seed}_{i:02d}"
        shapes.append(s)
    raw = [sample_pieces(s.pieces, spec.point_spacing) for s in shapes]

    # grid layout: rows of tracks, each bbox separated by cell_gap; whole layout centred on the origin
    n = len(raw)
    cols = int(math.ceil(math.sqrt(n)))
    boxes = []
    for t, s in zip(raw, shapes):
        lo = t.points.min(0) - s.width / 2
        hi = t.points.max(0) + s.width / 2
        boxes.append((lo, hi))
    cell_w = max(hi[0] - lo[0] for lo, hi in boxes) + spec.cell_gap
    cell_h = max(hi[1] - lo[1] for lo, hi in boxes) + spec.cell_gap
    placed = []
    for i, (t, (lo, hi)) in enumerate(zip(raw, boxes)):
        r, c = divmod(i, cols)
        cell_centre = np.array([c * cell_w, -r * cell_h])
        shift = cell_centre - (lo + hi) / 2
        placed.append(Track(points=t.points + shift, yaw=t.yaw, s=t.s))
    lo = np.min([t.points.min(0) for t in placed], axis=0)
    hi = np.max([t.points.max(0) for t in placed], axis=0)
    centre = (lo + hi) / 2
    tracks = [Track(points=t.points - centre, yaw=t.yaw, s=t.s) for t in placed]
    return Library(shapes=shapes, tracks=tracks, spec=spec)


# ---------------------------------------------------------------------------
# USD (plain text)
# ---------------------------------------------------------------------------


def _slab(track: Track, width: float, top_z: float, thickness: float, stride: int = 2):
    idx = list(range(0, len(track.s), stride))
    if idx[-1] != len(track.s) - 1:
        idx.append(len(track.s) - 1)
    c, th = track.points[idx], track.yaw[idx]
    nrm = np.stack([-np.sin(th), np.cos(th)], -1)
    L, R = c + width / 2 * nrm, c - width / 2 * nrm
    n = len(idx)
    zt, zb = top_z, top_z - thickness
    pts = np.zeros((4 * n, 3))
    pts[0::4, :2], pts[0::4, 2] = L, zt
    pts[1::4, :2], pts[1::4, 2] = R, zt
    pts[2::4, :2], pts[2::4, 2] = L, zb
    pts[3::4, :2], pts[3::4, 2] = R, zb
    faces = []
    for i in range(n - 1):
        a, b = 4 * i, 4 * (i + 1)
        faces += [[a + 1, b + 1, b, a], [a + 2, b + 2, b + 3, a + 3], [a + 3, b + 3, b + 1, a + 1], [a + 2, a, b, b + 2]]
    e = 4 * (n - 1)
    faces += [[3, 1, 0, 2], [e + 2, e, e + 1, e + 3]]
    return pts, faces


def _f3(p) -> str:
    return f"({p[0]:.5f}, {p[1]:.5f}, {p[2]:.5f})"


def library_usda_text(lib: Library) -> str:
    sp = lib.spec
    out = ["#usda 1.0", "(", '    defaultPrim = "Ramp"', "    metersPerUnit = 1", '    upAxis = "Z"', ")", "",
           'def Xform "Ramp"', "{"]
    for i, (track, shape) in enumerate(zip(lib.tracks, lib.shapes)):
        out += [f'    def Xform "Track{i:02d}"', "    {"]
        if shape.stripes:
            step = max(2, int(round(shape.stripe_len / sp.point_spacing)))
            chunks = [(a, min(a + step, len(track.s) - 1)) for a in range(0, len(track.s) - 1, step)]
            parts = [(f"slab_{k:03d}", Track(track.points[a:b + 1], track.yaw[a:b + 1], track.s[a:b + 1]),
                      shape.stripes[k % len(shape.stripes)]) for k, (a, b) in enumerate(chunks)]
        else:
            parts = [("slab", track, shape.color)]
        for pname, sub, col in parts:
            pts, faces = _slab(sub, shape.width, sp.top_z, sp.thickness, stride=2 if len(sub.s) > 4 else 1)
            lo, hi = pts.min(0), pts.max(0)
            out += [
                f'        def Mesh "{pname}" (',
                '            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]',
                "        )",
                "        {",
                f"            float3[] extent = [{_f3(lo)}, {_f3(hi)}]",
                f"            int[] faceVertexCounts = [{', '.join(['4'] * len(faces))}]",
                f"            int[] faceVertexIndices = [{', '.join(str(k) for f in faces for k in f)}]",
                f"            point3f[] points = [{', '.join(_f3(p) for p in pts)}]",
                f"            color3f[] primvars:displayColor = [({col[0]:.3f}, {col[1]:.3f}, {col[2]:.3f})]",
                '            uniform token subdivisionScheme = "none"',
                '            uniform token physics:approximation = "none"',
                "        }",
            ]
        h = sp.top_z - sp.thickness
        n_p = max(2, int(track.length // sp.pillar_spacing) + 1)
        for p_i, s in enumerate(np.linspace(0.05, track.length - 0.05, n_p)):
            k = min(int(np.searchsorted(track.s, s)), len(track.s) - 1)
            x, y = track.points[k]
            pc = sp.pillar_color
            out += [
                f'        def Cylinder "pillar_{p_i:02d}"',
                "        {",
                '            uniform token axis = "Z"',
                f"            double height = {h:.5f}",
                f"            double radius = {sp.pillar_radius:.5f}",
                f"            color3f[] primvars:displayColor = [({pc[0]:.3f}, {pc[1]:.3f}, {pc[2]:.3f})]",
                f"            double3 xformOp:translate = ({x:.5f}, {y:.5f}, {h / 2:.5f})",
                '            uniform token[] xformOpOrder = ["xformOp:translate"]',
                "        }",
            ]
        out.append("    }")
    out += ["}", ""]
    return "\n".join(out)


def library_hash(spec: LibrarySpec) -> str:
    return hashlib.sha1(json.dumps(asdict(spec), sort_keys=True, default=str).encode()).hexdigest()[:10]


def write_library_usd(lib: Library, directory: str | None = None) -> str:
    directory = directory or os.path.join(os.path.expanduser("~"), ".cache", "jetbot_ramp")
    os.makedirs(directory, exist_ok=True)
    text = library_usda_text(lib)
    path = os.path.join(directory, f"library_{hashlib.sha1(text.encode()).hexdigest()[:12]}.usda")
    if not os.path.exists(path):
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# Batched torch helpers (tracks of different lengths, padded)
# ---------------------------------------------------------------------------


class LibraryTensors:
    """Padded tensors for projecting robots onto their assigned track's centreline."""

    def __init__(self, lib: Library, device: str = "cpu"):
        import torch

        self.torch = torch
        K = len(lib.tracks)
        n_max = max(len(t.s) for t in lib.tracks)
        pts = np.zeros((K, n_max, 2), np.float32)
        yaw = np.zeros((K, n_max), np.float32)
        s = np.zeros((K, n_max), np.float32)
        n_valid = np.zeros(K, np.int64)
        for k, t in enumerate(lib.tracks):
            n = len(t.s)
            pts[k, :n], yaw[k, :n], s[k, :n] = t.points, t.yaw, t.s
            pts[k, n:], yaw[k, n:], s[k, n:] = t.points[-1], t.yaw[-1], t.s[-1]  # pad with the end point
            n_valid[k] = n
        self.points = torch.tensor(pts, device=device)
        self.yaw = torch.tensor(yaw, device=device)
        self.s = torch.tensor(s, device=device)
        self.n_valid = torch.tensor(n_valid, device=device)
        self.tangent = torch.stack([torch.cos(self.yaw), torch.sin(self.yaw)], -1)
        self.length = torch.tensor([t.length for t in lib.tracks], device=device, dtype=torch.float32)
        self.half_width = torch.tensor([sh.width / 2 for sh in lib.shapes], device=device, dtype=torch.float32)
        self.num_tracks = K
        self.spacing = float(lib.spec.point_spacing)

    def project(self, track_id, pos_xy):
        torch = self.torch
        pts = self.points[track_id]
        idx = torch.argmin(((pts - pos_xy[:, None, :]) ** 2).sum(-1), dim=1)
        ar = torch.arange(len(track_id), device=pos_xy.device)
        p = pts[ar, idx]
        t = self.tangent[track_id, idx]
        rel = pos_xy - p
        s = (self.s[track_id, idx] + (rel * t).sum(-1)).clamp(min=0.0)
        s = torch.minimum(s, self.length[track_id])
        lateral = t[:, 0] * rel[:, 1] - t[:, 1] * rel[:, 0]
        return idx, s, lateral, self.yaw[track_id, idx]

    def pose_at(self, track_id, s):
        torch = self.torch
        last = (self.n_valid[track_id] - 2).clamp(min=0)
        f = (s / self.spacing).clamp(min=0.0)
        i0 = torch.minimum(f.floor().long(), last)
        w = (f - i0.float()).clamp(0.0, 1.0)[:, None]
        p = self.points[track_id, i0] * (1 - w) + self.points[track_id, i0 + 1] * w
        y0 = self.yaw[track_id, i0]
        y1 = self.yaw[track_id, i0 + 1]
        dy = torch.atan2(torch.sin(y1 - y0), torch.cos(y1 - y0))
        return p, y0 + dy * w[:, 0]

    def lookahead_body(self, track_id, s, pos_xy, yaw, distances):
        torch = self.torch
        E, M = len(track_id), len(distances)
        d = torch.as_tensor(distances, dtype=torch.float32, device=pos_xy.device)
        s_q = torch.minimum(s[:, None] + d[None, :], self.length[track_id][:, None]).reshape(-1)
        tid = track_id[:, None].expand(E, M).reshape(-1)
        p, _ = self.pose_at(tid, s_q)
        rel = p.view(E, M, 2) - pos_xy[:, None, :]
        c, sn = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        return torch.stack([c * rel[..., 0] + sn * rel[..., 1], -sn * rel[..., 0] + c * rel[..., 1]], -1)


def rasterize_library(lib: Library, res: float = 0.01, pad: float = 1.0):
    """Boolean top-view occupancy map of all ramp surfaces (env-local frame).

    Returns (grid[H, W] bool, origin_xy (x0, y0) of cell [0, 0], res). Cell (i, j) covers
    x in [x0 + j*res, ...), y in [y0 + i*res, ...).
    """
    lo = np.min([t.points.min(0) for t in lib.tracks], axis=0) - pad
    hi = np.max([t.points.max(0) for t in lib.tracks], axis=0) + pad
    W = int(np.ceil((hi[0] - lo[0]) / res)) + 1
    H = int(np.ceil((hi[1] - lo[1]) / res)) + 1
    grid = np.zeros((H, W), bool)
    for t, sh in zip(lib.tracks, lib.shapes):
        hw = sh.width / 2
        # 4x denser centreline (linear interpolation) so the outside of tight curves has no gaps
        f = np.linspace(0, len(t.s) - 1, 4 * len(t.s))
        i0 = np.minimum(np.floor(f).astype(int), len(t.s) - 2)
        w = (f - i0)[:, None]
        pts = t.points[i0] * (1 - w) + t.points[i0 + 1] * w
        dy = np.arctan2(np.sin(t.yaw[i0 + 1] - t.yaw[i0]), np.cos(t.yaw[i0 + 1] - t.yaw[i0]))
        yaw = t.yaw[i0] + dy * w[:, 0]
        nrm = np.stack([-np.sin(yaw), np.cos(yaw)], -1)
        for off in np.arange(-hw, hw + 1e-9, res / 2):
            p = pts + off * nrm
            j = np.floor((p[:, 0] - lo[0]) / res).astype(int)
            i = np.floor((p[:, 1] - lo[1]) / res).astype(int)
            grid[i, j] = True
    return grid, (float(lo[0]), float(lo[1])), res


# ---------------------------------------------------------------------------
# Stress-test tracks (deliberately harder than anything in training)
# ---------------------------------------------------------------------------


def _serpentine(rows: int, row_len: float, turn_r: float) -> list[tuple[float, float]]:
    pieces = []
    for r in range(rows):
        pieces.append((row_len, 0.0))
        if r < rows - 1:
            pieces.append(_arc(turn_r, 180, r % 2 == 0))
    return pieces


def _long_wiggle(seed: int = 7, rows: int = 7, turn_r: float = 0.6, width: float = 0.26) -> list[tuple[float, float]]:
    """Serpentine whose rows are made of small alternating arcs (constant wiggling), ~60 m."""
    rng = np.random.default_rng(seed)
    for _ in range(200):
        pieces = [(0.4, 0.0)]
        for r in range(rows):
            for k in range(3):  # 3 wiggle pairs per row; pair order alternates so the row comes back to centre
                R = rng.uniform(1.0, 2.0)
                th = rng.uniform(12, 26)
                a, b = (_arc(R, th, True), _arc(R, th, False)) if k % 2 == 0 else (_arc(R, th, False), _arc(R, th, True))
                pieces += [a, b, b, a]
            pieces.append((rng.uniform(0.3, 1.0), 0.0))
            if r < rows - 1:
                pieces.append(_arc(turn_r, 180, r % 2 == 0))
        pieces.append((0.4, 0.0))
        t = sample_pieces(pieces, 0.01)
        if self_clearance(t, width, stride=4) > 0.3:
            return pieces
    raise RuntimeError("long wiggle generation failed")


RAINBOW = [(0.80, 0.08, 0.08), (0.90, 0.45, 0.04), (0.90, 0.82, 0.08), (0.08, 0.55, 0.12),
           (0.06, 0.55, 0.65), (0.08, 0.18, 0.80), (0.45, 0.12, 0.65), (0.85, 0.85, 0.85)]
_S = [(0.4, 0), _arc(0.6, 180, True), _arc(0.6, 180, False), (0.4, 0)]

STRESS_TRACKS: dict[str, TrackShape] = {
    "long_wiggle_60m": TrackShape(_long_wiggle(), 0.26, PALETTE[0]),
    "serpentine_120m": TrackShape([(0.4, 0.0)] + _serpentine(17, 6.0, 0.6) + [(0.4, 0.0)], 0.26, PALETTE[2]),
    "white_s": TrackShape(list(_S), 0.26, (0.80, 0.80, 0.78)),
    "rainbow_s": TrackShape(list(_S), 0.26, PALETTE[0], stripes=RAINBOW, stripe_len=0.4),
    "hairpins_r03": TrackShape([(0.4, 0)] + [_arc(0.3, 180, True), (0.3, 0), _arc(0.3, 180, False), (0.3, 0)] * 2
                               + [(0.4, 0)], 0.26, PALETTE[4]),
    "sharp_zigzag": TrackShape([(0.5, 0)] + [_arc(0.15, 90, True), (0.6, 0), _arc(0.15, 90, False), (0.6, 0)] * 4
                               + [(0.4, 0)], 0.26, PALETTE[3]),
    "narrow_18cm_s": TrackShape(list(_S), 0.18, PALETTE[5]),
    "narrow_chicane": TrackShape([(0.4, 0), _arc(0.4, 120, True), _arc(0.4, 120, False), (0.3, 0), _arc(0.35, 150, True),
                                  _arc(0.45, 90, False), (0.3, 0), _arc(0.4, 120, True), (0.4, 0)], 0.22, PALETTE[1]),
}


# train / test libraries
TRAIN_LIBRARY = LibrarySpec(seed=1, num_random=12, named=[])
TEST_LIBRARY = LibrarySpec(seed=1001, num_random=3, named=list(NAMED_TRACKS))
STRESS_LIBRARY = LibrarySpec(seed=2001, num_random=0, named=list(STRESS_TRACKS), cell_gap=1.5)
