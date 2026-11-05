"""In-sim smoke test for the procedural-track env: scripted pure pursuit, sensor sanity, optional PiP video.

    ./run.sh tools/tracks_smoke.py --task Isaac-JetbotTracks-Teacher --num_envs 24 --speed 3
    ./run.sh tools/tracks_smoke.py --task Isaac-JetbotTracks-Camera-Eval --num_envs 1 --track 1 --speed 2 --record videos/x.mp4
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import os
import sys

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
import jetbot_ramp  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import viz  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-JetbotTracks-Teacher")
parser.add_argument("--num_envs", type=int, default=24)
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--track", type=int, default=-1)
parser.add_argument("--lookahead", type=float, default=0.3)
parser.add_argument("--record", default="")
parser.add_argument("--caption", default="")
add_launcher_args(parser)
parser.set_defaults(device=None)
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + hydra_args

env_cfg, _ = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.domain_randomization = False
env_cfg.events = None
env_cfg.random_start_prob = 0.0
env_cfg.fixed_speed_scale = args_cli.speed
env_cfg.fixed_track = args_cli.track
args_cli.device = env_cfg.sim.device
if args_cli.record:
    args_cli.enable_cameras = True
    from isaaclab_visualizers.kit import KitVisualizerCfg

    from isaaclab.envs import VideoRecorderCfg

    raw = os.path.join(os.path.dirname(os.path.abspath(args_cli.record)), "_raw")
    os.makedirs(raw, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args_cli.record))[0]
    for f in glob.glob(os.path.join(raw, f"{tag}_*.mp4")):
        os.remove(f)
    env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(headless=True, origin_type="asset", origin_track_path="robot",
                                                    eye=(-0.9, -1.3, 1.1), lookat=(0.25, 0.0, 0.0),
                                                    window_width=1280, window_height=720)]
    env_cfg.video_recorders = [VideoRecorderCfg(source="visualizer:kit", output_dir=raw, video_length=args_cli.steps,
                                                output_filename_prefix=f"{tag}_viewer")]
    if getattr(env_cfg, "use_camera", False):
        env_cfg.video_recorders.append(VideoRecorderCfg(source="sensor:front_cam", output_dir=raw,
                                                        video_length=args_cli.steps, output_filename_prefix=f"{tag}_cam"))

with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    obs, _ = env.reset()
    print("[tsmoke] obs groups:", {k: tuple(v.shape) for k, v in obs.items()})
    print("[tsmoke] tracks:", [s.name for s in u.library.shapes])
    lidar_w = viz.LidarPlotWriter(os.path.join(raw, f"{tag}_lidar.mp4")) if args_cli.record and u._lidar else None
    WB, WR = 0.1125, 0.03
    n_fin = n_fall = 0
    lap_times = []
    for step in range(args_cli.steps + 2):
        tgt = u.lib.lookahead_body(u.track_id, u.progress, u.pos_xy, u.yaw, [args_cli.lookahead * max(1.0, args_cli.speed / 1.5)])[:, 0]
        curv = 2 * tgt[:, 1] / tgt.norm(dim=-1).square().clamp(min=1e-4)
        vmax = u.cfg.base_wheel_speed * u.speed_scale * WR * 0.8
        v = vmax / (1 + 0.25 * curv.abs())  # slow down in tight turns
        w = v * curv
        act = torch.stack([(v - w * WB / 2), (v + w * WB / 2)], -1) / WR / (u.cfg.base_wheel_speed * u.speed_scale[:, None])
        act = act / act.abs().max(-1, keepdim=True).values.clamp(min=1.0)
        obs, rew, term, trunc, _ = env.step(act)
        if term.any():
            fin = term & (rew > 5)
            n_fin += int(fin.sum())
            n_fall += int((term & (rew < -5)).sum())
        if lidar_w is not None:
            lidar_w.add(*viz.lidar_sample(u, 0))
        if step == 60:
            print(f"[tsmoke] lidar[0] ranges m: centre={float(u.lidar_range[0, 30]):.3f} min={float(u.lidar_range[0].min()):.3f} "
                  f"kinds ramp/floor/none={[(u.lidar_kind[0] == k).sum().item() for k in (1, 0, -1)]}")
            print(f"[tsmoke] cliff[0] ranges m: {[round(float(x), 3) for x in u.cliff_range[0]]} kinds={u.cliff_kind[0].tolist()}")
        if step % 60 == 0:
            sp = u.robot.data.root_lin_vel_b.torch[:, 0]
            print(f"[tsmoke] t={step / 30:4.1f}s fwd speed mean={sp.mean():.2f} max={sp.max():.2f} m/s "
                  f"lat|max|={u.lateral.abs().max():.3f}")
    print(f"[tsmoke] speed {args_cli.speed}x: finished={n_fin} fell={n_fall}")
    if args_cli.record:
        viewer = sorted(glob.glob(os.path.join(raw, f"{tag}_viewer_*.mp4")))[-1]
        cams = sorted(glob.glob(os.path.join(raw, f"{tag}_cam_*.mp4")))
        lid = lidar_w.close() if lidar_w else None
        cap = (args_cli.caption + f"  |  finished {n_fin}, fell {n_fall}") if args_cli.caption else ""
        viz.compose(viewer, os.path.abspath(args_cli.record), cam=cams[-1] if cams else None, lidar=lid, caption=cap)
        print(f"[tsmoke] wrote {args_cli.record}")
    env.close()
