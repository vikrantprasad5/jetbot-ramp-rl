"""Mosaic video: one robot per track (all tracks of a library at once), each with a follow camera, tiled.

    ./run.sh tools/mosaic.py --task Isaac-JetbotTracks-Eval --agent teacher_cfg_entry_point \
        --ckpt logs/.../model_1999.pt --speed 3 --seconds 25 --title "Teacher - 8 held-out tracks, 3x" \
        --out videos/fusion/teacher_test_mosaic.mp4
"""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
import jetbot_ramp  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz import FONT  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-JetbotTracks-Eval")
parser.add_argument("--agent", default="teacher_cfg_entry_point")
parser.add_argument("--ckpt", required=True)
parser.add_argument("--speed", type=float, default=3.0)
parser.add_argument("--seconds", type=float, default=25.0)
parser.add_argument("--title", default="")
parser.add_argument("--out", required=True)
add_launcher_args(parser)
parser.set_defaults(device=None)
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + hydra_args
args_cli.enable_cameras = True

env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
from jetbot_ramp.tracks import build_library  # noqa: E402

lib = build_library(env_cfg.library)
K = len(lib.tracks)
env_cfg.scene.num_envs = K
env_cfg.track_per_env = True
env_cfg.showcase = True
env_cfg.fixed_speed_scale = args_cli.speed
env_cfg.random_start_prob = 0.0
env_cfg.camera_dropout_prob = 0.0
env_cfg.lidar_dropout_prob = 0.0
env_cfg.domain_randomization = False
env_cfg.events = None
env_cfg.episode_length_s = args_cli.seconds + 5.0
args_cli.device = env_cfg.sim.device
from isaaclab_visualizers.kit import KitVisualizerCfg  # noqa: E402

env_cfg.sim.visualizer_cfgs = [
    KitVisualizerCfg(headless=True, window_width=1280, window_height=720, streaming_view=True,
                     streaming_cam_target_prim_path="/World/envs/env_.*/Robot", streaming_cam_eye=(-0.8, -0.9, 0.9),
                     streaming_envs=list(range(K)))
]

with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_rl.rsl_rl.utils import check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    agent_cfg.device = env_cfg.sim.device
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(wenv, agent_cfg)
    runner.load(os.path.abspath(args_cli.ckpt))
    policy = runner.alg.get_policy()
    policy.eval()
    viz = [v for v in u.sim.visualizers if getattr(v.cfg, "visualizer_type", "") == "kit"][0]
    frames = []
    with torch.inference_mode():
        obs, _ = wenv.reset()
        for step in range(int(args_cli.seconds / u.step_dt)):
            obs, _, _, _ = wenv.step(policy(obs))
            if not u.sim.is_rendering:
                u.sim.forward()
            f = viz.render_tiled_rgb_array()
            if f is not None:
                frames.append(np.ascontiguousarray(np.asarray(f)[..., :3]).astype(np.uint8))
    n_fin = int((u.status == 1).sum())
    n_fell = int((u.status == 2).sum())
    lens = u.lib.length
    prog = (u.best_progress / lens[u.track_id]).cpu().numpy()
    names = [s.name for s in lib.shapes]
    per = [f"{names[i]}: {'FINISH ' + format(float(u.finish_time[i]), '.1f') + ' s' if int(u.status[i]) == 1 else ('FELL at ' if int(u.status[i]) == 2 else 'reached ') + format(prog[i] * 100, '.0f') + '%'}" for i in range(K)]
    summary = f"{args_cli.title}  |  {n_fin}/{K} finished, {n_fell} fell"
    print("[mosaic] " + summary)
    for p in per:
        print("[mosaic]   " + p)
    H, W = frames[0].shape[:2]
    out = os.path.abspath(args_cli.out)
    txt = summary.replace(":", "\\:").replace(",", "\\,")
    # infer the tile grid (16:9 tiles, row-major in env order) and label each tile with its track result
    rows = next(r for r in range(1, K + 1) if round(W / (H / r * 16 / 9)) * r >= K)
    cols = round(W / (H / rows * 16 / 9))
    tw, th = W / cols, H / rows
    labels = []
    for i, txt_i in enumerate(per):
        r, c = divmod(i, cols)
        t_i = txt_i.replace(":", "\\:").replace(",", "\\,")
        labels.append(f"drawtext=fontfile={FONT}:expansion=none:text='{t_i}':x={int(c * tw) + 6}:y={int(r * th) + 6}:"
                      f"fontsize=15:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=4")
    vf = ",".join(labels + [f"drawtext=fontfile={FONT}:expansion=none:text='{txt}':x=16:y=h-40:fontsize=22:"
                            f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8"])
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r",
                          str(round(1 / u.step_dt)), "-i", "-", "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                          "-crf", "21", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    with open(out.replace(".mp4", ".txt"), "w") as fh:
        fh.write(summary + "\n" + "\n".join(per) + "\n")
    print(f"[mosaic] wrote {out} ({W}x{H}, {len(frames)} frames)")
    env.close()
