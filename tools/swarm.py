"""Swarm "generations" video: for each checkpoint, N robots start together on the same track (stacked envs,
no inter-robot collisions), drive with the policy's own sampling noise, fallen robots stay down, finished
ones stop. One captioned segment per generation, then all segments are joined.

    ./run.sh tools/swarm.py --task Isaac-JetbotTracks-Eval --agent teacher_cfg_entry_point \
        --ckpt "Gen 1 (iter 0)=logs/.../model_0.pt" --ckpt "Gen 2 (iter 20)=logs/.../model_20.pt" ... \
        --track 1 --speed 1 --out videos/fusion/teacher_generations.mp4

Camera models need a camera task (Isaac-JetbotTracks-Camera-Eval / -Camera-Train-Show).
Also: --layout grid tiles one robot per track (final model tour) instead of a swarm on one track.
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
from viz import FONT, concat  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-JetbotTracks-Eval")
parser.add_argument("--agent", default="teacher_cfg_entry_point")
parser.add_argument("--ckpt", action="append", required=True, help="'label=path' (repeatable, in order)")
parser.add_argument("--track", type=int, default=1)
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--robots", type=int, default=60)
parser.add_argument("--max_seconds", type=float, default=14.0)
parser.add_argument("--hold_seconds", type=float, default=1.0)
parser.add_argument("--mean_action", action="store_true", help="use the mean action instead of sampling")
parser.add_argument("--title", default="")
parser.add_argument("--out", required=True)
add_launcher_args(parser)
parser.set_defaults(device=None)
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + hydra_args
args_cli.enable_cameras = True

env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
env_cfg.scene.num_envs = args_cli.robots
env_cfg.scene.env_spacing = 0.0  # every robot on the same copy of the track
env_cfg.showcase = True
env_cfg.fixed_track = args_cli.track
env_cfg.fixed_speed_scale = args_cli.speed
env_cfg.random_start_prob = 0.0
env_cfg.reset_lateral_noise = 0.05
env_cfg.reset_yaw_noise = 0.3
env_cfg.camera_dropout_prob = 0.0
env_cfg.lidar_dropout_prob = 0.0
env_cfg.domain_randomization = False
env_cfg.events = None
env_cfg.episode_length_s = args_cli.max_seconds + args_cli.hold_seconds + 5.0
args_cli.device = env_cfg.sim.device

# frame the chosen track from above-behind its start
from jetbot_ramp.tracks import build_library  # noqa: E402

lib = build_library(env_cfg.library)
pts = lib.tracks[args_cli.track].points
lo, hi = pts.min(0), pts.max(0)
c = (lo + hi) / 2
span = float(max(hi - lo)) + 0.6
from isaaclab_visualizers.kit import KitVisualizerCfg  # noqa: E402

start = pts[0]
# camera behind-above the start, looking at the track centre, far enough to contain the whole track
d = c - start
dn = d / (np.linalg.norm(d) + 1e-6)
look = start + 0.55 * d
eye = start - dn * (0.55 * span)
env_cfg.sim.visualizer_cfgs = [
    KitVisualizerCfg(headless=True, origin_type="world", eye=(float(eye[0]), float(eye[1]), 0.3 + 0.75 * span),
                     lookat=(float(look[0]), float(look[1]), 0.3), window_width=1280, window_height=720)
]

out = os.path.abspath(args_cli.out)
parts_dir = os.path.join(os.path.dirname(out), "_raw")
os.makedirs(parts_dir, exist_ok=True)
tag = os.path.splitext(os.path.basename(out))[0]

with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_rl.rsl_rl.utils import check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    agent_cfg.device = env_cfg.sim.device
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(wenv, agent_cfg)
    viz = [v for v in u.sim.visualizers if getattr(v.cfg, "visualizer_type", "") == "kit"][0]
    length = float(u.lib.length[args_cli.track])
    max_steps = int(args_cli.max_seconds / u.step_dt)
    hold = int(args_cli.hold_seconds / u.step_dt)
    parts, summary = [], []

    for gi, spec in enumerate(args_cli.ckpt):
        label, path = spec.split("=", 1)
        runner.load(os.path.abspath(path))
        policy = runner.alg.get_policy()
        policy.eval()
        frames = []
        done_at = None
        with torch.inference_mode():  # reset inside too: env buffers touched during inference stay inference tensors
            obs, _ = wenv.reset()
            for step in range(max_steps):
                obs, _, _, _ = wenv.step(policy(obs, stochastic_output=not args_cli.mean_action))
                if not u.sim.is_rendering:
                    u.sim.forward()
                f = viz.render_rgb_array()
                if f is not None:
                    frames.append(np.ascontiguousarray(f[..., :3]))
                if done_at is None and bool((u.status != 0).all()):
                    done_at = step
                if done_at is not None and step - done_at >= hold:
                    break
        n_fin = int((u.status == 1).sum())
        n_fell = int((u.status == 2).sum())
        med = float(torch.median(u.best_progress / length)) * 100
        ft = u.finish_time[u.status == 1]
        best = f", best lap {float(ft.min()):.1f} s" if len(ft) else ""
        stats = f"{label}  |  {args_cli.robots} robots: {n_fin} finished, {n_fell} fell, median {med:.0f}% of track{best}"
        summary.append(stats)
        print(f"[swarm] {stats}")
        # encode this generation with its caption
        seg = os.path.join(parts_dir, f"{tag}_{gi:02d}.mp4")
        H, W = frames[0].shape[:2]
        txt = stats.replace(":", "\\:").replace(",", "\\,")
        title = args_cli.title.replace(":", "\\:")
        vf = (f"drawtext=fontfile={FONT}:expansion=none:text='{txt}':x=16:y=h-44:fontsize=22:fontcolor=white:box=1:"
              f"boxcolor=black@0.6:boxborderw=8")
        if title:
            vf += f",drawtext=fontfile={FONT}:expansion=none:text='{title}':x=16:y=16:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8"
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                              "-r", str(round(1 / u.step_dt)), "-i", "-", "-vf", vf, "-c:v", "libx264", "-pix_fmt",
                              "yuv420p", "-crf", "20", seg], stdin=subprocess.PIPE)
        for f in frames:
            p.stdin.write(f.astype(np.uint8).tobytes())
        p.stdin.close()
        p.wait()
        parts.append(seg)

    concat(parts, out)
    with open(out.replace(".mp4", ".txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
    print(f"[swarm] wrote {out}")
    env.close()
