"""Offline checks for ramp geometry and track maths. No Isaac Sim needed.

Run with the Isaac Sim python, which has torch, on CPU only:

    CUDA_VISIBLE_DEVICES= ~/isaacsim/kit/python/bin/python3 tests/test_ramp.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jetbot_ramp.ramp import (  # noqa: E402
    TrackSpec,
    TrackTensors,
    build_centreline,
    build_tracks,
    footprint_size,
    ramp_usda_text,
    write_ramp_usd,
)

SPEC = TrackSpec()


def test_centreline_shape():
    t = build_centreline(SPEC)
    R = SPEC.radius
    expected_len = SPEC.straight_in + SPEC.straight_out + 2 * math.radians(SPEC.arc_deg) * R
    assert abs(t.length - expected_len) < 1e-9, t.length
    # letter S: start (0,0) heading +x; end at (straight_in + straight_out, 4R) heading +x
    assert np.allclose(t.points[0], [0, 0])
    assert np.allclose(t.points[-1], [SPEC.straight_in + SPEC.straight_out, 4 * R], atol=1e-6), t.points[-1]
    assert abs(t.yaw[0]) < 1e-9 and abs(math.remainder(t.yaw[-1], 2 * math.pi)) < 1e-6
    # samples are evenly spaced along the curve (chord ~= spacing)
    seg = np.linalg.norm(np.diff(t.points, axis=0), axis=1)
    assert np.all(np.abs(seg - (t.s[1] - t.s[0])) < 1e-4)
    # first arc turns left (yaw rises to +pi/2 midway), second turns right
    mid1 = np.searchsorted(t.s, SPEC.straight_in + math.radians(SPEC.arc_deg) * R / 2)
    assert abs(t.yaw[mid1] - math.pi / 2) < 0.02


def test_mirror():
    a = build_centreline(SPEC, mirror=False)
    b = build_centreline(SPEC, mirror=True)
    assert np.allclose(a.points[:, 0], b.points[:, 0])
    assert np.allclose(a.points[:, 1], -b.points[:, 1])


def test_tracks_separated():
    a, b = build_tracks(SPEC)
    hw = SPEC.width / 2
    gap = b.points[:, 0].min() - hw - (a.points[:, 0].max() + hw)
    assert abs(gap - SPEC.track_gap) < 1e-6, gap
    fx, fy = footprint_size(SPEC)
    print(f"  footprint {fx:.2f} x {fy:.2f} m, track length {a.length:.2f} m, samples {len(a.s)}")


def test_projection_and_lookahead():
    tracks = build_tracks(SPEC)
    tt = TrackTensors(tracks)
    E = 400
    g = torch.Generator().manual_seed(0)
    tid = torch.randint(0, 2, (E,), generator=g)
    s_true = torch.rand(E, generator=g) * (tt.length[tid] - 0.02) + 0.01
    lat_true = (torch.rand(E, generator=g) - 0.5) * SPEC.width
    p, yaw = tt.pose_at(tid, s_true)
    nrm = torch.stack([-torch.sin(yaw), torch.cos(yaw)], -1)
    pos = p + lat_true[:, None] * nrm
    _, s, lat, tyaw = tt.project(tid, pos)
    assert (s - s_true).abs().max() < 0.01, (s - s_true).abs().max()
    assert (lat - lat_true).abs().max() < 0.005, (lat - lat_true).abs().max()
    dyaw = torch.atan2(torch.sin(tyaw - yaw), torch.cos(tyaw - yaw))
    assert dyaw.abs().max() < 0.05
    # lookahead at distance 0 from the true pose is the robot itself (body origin)
    la = tt.lookahead_body(tid, s_true, p, yaw, [0.0, 0.3])
    assert la[:, 0].abs().max() < 1e-3
    # 0.3 m ahead is in front (x > 0) and within 0.3 m (away from the end, where points clamp to the last sample)
    far = s_true < tt.length[tid] - 0.3
    assert (la[far, 1, 0] > 0.15).all() and (la[:, 1].norm(dim=-1) <= 0.3 + 1e-3).all()


def test_unicycle_pure_pursuit_completes_track():
    """Kinematic check that a simple controller on the teacher's inputs can finish both tracks.

    This exercises progress, lateral offset, the fall check (|lateral| > half width) and finishing.
    """
    tracks = build_tracks(SPEC)
    tt = TrackTensors(tracks)
    tid = torch.tensor([0, 1])
    s0 = torch.tensor([0.1, 0.1])
    pos, yaw = tt.pose_at(tid, s0)
    dt, v_max, wheel_base = 1 / 30, 0.35, 0.1125
    done = torch.zeros(2, dtype=torch.bool)
    fell = torch.zeros(2, dtype=torch.bool)
    max_lat = torch.zeros(2)
    for _ in range(int(30 / dt)):
        _, s, lat, _ = tt.project(tid, pos)
        max_lat = torch.maximum(max_lat, lat.abs())
        fell |= lat.abs() > SPEC.width / 2
        done |= s >= tt.length[tid] - 0.15
        if (done | fell).all():
            break
        tgt = tt.lookahead_body(tid, s, pos, yaw, [0.25])[:, 0]  # (E,2)
        curvature = 2 * tgt[:, 1] / (tgt.norm(dim=-1) ** 2)
        v = torch.full((2,), v_max)
        omega = v * curvature
        # convert to wheel speeds and saturate like the real robot
        vl, vr = v - omega * wheel_base / 2, v + omega * wheel_base / 2
        m = torch.maximum(vl.abs(), vr.abs()).clamp(min=v_max) / v_max
        vl, vr = vl / m, vr / m
        v, omega = (vl + vr) / 2, (vr - vl) / wheel_base
        yaw = yaw + omega * dt
        pos = pos + torch.stack([torch.cos(yaw), torch.sin(yaw)], -1) * v[:, None] * dt
    print(f"  pure pursuit: finished={done.tolist()} fell={fell.tolist()} max|lat|={max_lat.tolist()}")
    assert done.all() and not fell.any()


def test_usd_text():
    txt = ramp_usda_text(SPEC)
    assert txt.startswith("#usda 1.0")
    assert txt.count('def Mesh "slab"') == 2
    assert txt.count("{") == txt.count("}")
    assert txt.count("(") == txt.count(")")
    with tempfile.TemporaryDirectory() as d:
        p1 = write_ramp_usd(SPEC, d)
        p2 = write_ramp_usd(SPEC, d)
        assert p1 == p2 and os.path.getsize(p1) > 1000
    print(f"  usda size {len(txt) / 1e3:.0f} kB")


def test_mesh_closed_and_outward():
    """Every edge is shared by exactly two faces (watertight) and the signed volume is positive (outward)."""
    from jetbot_ramp.ramp import _slab_mesh

    track = build_tracks(SPEC)[0]
    pts, counts, idx = _slab_mesh(track, SPEC)
    faces = np.array(idx).reshape(-1, 4)
    edges = {}
    for f in faces:
        for a, b in zip(f, np.roll(f, -1)):
            edges.setdefault((min(a, b), max(a, b)), []).append((a, b))
    assert all(len(v) == 2 for v in edges.values()), "mesh is not closed"
    assert all(v[0] != v[1] for v in edges.values()), "inconsistent winding"
    vol = 0.0
    for f in faces:
        p = pts[f]
        for tri in ((0, 1, 2), (0, 2, 3)):
            a, b, c = p[list(tri)]
            vol += np.dot(a, np.cross(b, c)) / 6
    length = track.length
    expected = length * SPEC.width * SPEC.thickness
    assert vol > 0, "faces point inwards"
    assert abs(vol - expected) / expected < 0.02, (vol, expected)


if __name__ == "__main__":
    torch.set_default_device("cpu")
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
