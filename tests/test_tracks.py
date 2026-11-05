"""Offline checks for the procedural track library (no Isaac Sim).

    CUDA_VISIBLE_DEVICES= ~/isaacsim/kit/python/bin/python3 tests/test_tracks.py
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jetbot_ramp.tracks import (  # noqa: E402
    TEST_LIBRARY, TRAIN_LIBRARY, LibraryTensors, _slab, build_library, library_usda_text, self_clearance,
)


def check_library(spec, label):
    t0 = time.time()
    lib = build_library(spec)
    fx, fy = lib.footprint()
    print(f"  {label}: {len(lib.tracks)} tracks, lengths {min(t.length for t in lib.tracks):.1f}-"
          f"{max(t.length for t in lib.tracks):.1f} m, footprint {fx:.1f} x {fy:.1f} m ({time.time() - t0:.1f} s)")
    # no self-intersection, and tracks don't overlap each other
    for t, s in zip(lib.tracks, lib.shapes):
        assert self_clearance(t, s.width) >= spec.generator.min_gap - 1e-6 or s.name, s.name
    for i in range(len(lib.tracks)):
        for j in range(i + 1, len(lib.tracks)):
            a, b = lib.tracks[i].points[::5], lib.tracks[j].points[::5]
            d = np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)).min()
            assert d > (lib.shapes[i].width + lib.shapes[j].width) / 2 + 0.5, (i, j, d)
    # meshes watertight and outward
    for t, s in zip(lib.tracks, lib.shapes):
        pts, faces = _slab(t, s.width, spec.top_z, spec.thickness)
        edges = {}
        for f in faces:
            for a, b in zip(f, f[1:] + f[:1]):
                edges.setdefault((min(a, b), max(a, b)), []).append((a, b))
        assert all(len(v) == 2 and v[0] != v[1] for v in edges.values()), s.name
        vol = sum(np.dot(pts[f[0]], np.cross(pts[f[k]], pts[f[k + 1]])) / 6 for f in faces for k in (1, 2))
        exp = t.length * s.width * spec.thickness
        assert vol > 0 and abs(vol - exp) / exp < 0.03, (s.name, vol, exp)
    txt = library_usda_text(lib)
    assert txt.count("{") == txt.count("}")
    # projection on padded tensors
    lt = LibraryTensors(lib)
    g = torch.Generator().manual_seed(0)
    E = 2000
    tid = torch.randint(0, lt.num_tracks, (E,), generator=g)
    s_true = torch.rand(E, generator=g) * (lt.length[tid] - 0.02) + 0.01
    lat_true = (torch.rand(E, generator=g) - 0.5) * 2 * lt.half_width[tid] * 0.9
    p, yaw = lt.pose_at(tid, s_true)
    pos = p + lat_true[:, None] * torch.stack([-torch.sin(yaw), torch.cos(yaw)], -1)
    _, s, lat, _ = lt.project(tid, pos)
    assert (s - s_true).abs().max() < 0.02, (s - s_true).abs().max()
    assert (lat - lat_true).abs().max() < 0.01, (lat - lat_true).abs().max()
    return lib


if __name__ == "__main__":
    ok = True
    for spec, label in [(TRAIN_LIBRARY, "train"), (TEST_LIBRARY, "test")]:
        try:
            check_library(spec, label)
            print(f"PASS {label} library")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"FAIL {label} library: {type(e).__name__}: {e}")
    sys.exit(0 if ok else 1)
