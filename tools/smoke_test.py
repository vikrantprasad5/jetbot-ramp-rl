"""In-sim smoke test: drive the Jetbot with a scripted pure-pursuit controller.

Checks that the scene builds, the wheels drive forward for positive actions, progress and
termination behave, and (for the camera task) saves camera frames so you can check the view.

Run only once no other Isaac Sim session needs the GPU:

    cd ~/jetbot_ramp
    ./run.sh tools/smoke_test.py --task Isaac-JetbotRamp-Teacher-Direct --num_envs 8
    ./run.sh tools/smoke_test.py --task Isaac-JetbotRamp-Distill-Direct --num_envs 4

Headless is the default in Isaac Lab 3.0 and cameras turn on automatically; add ``--viz kit`` to watch.

Pass ``--mode zero`` to only check spawning (robots should sit still on the ramp), or
``--mode forward`` to drive straight (robots should drive off the ramp at the first turn).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
import jetbot_ramp  # noqa: F401  (registers the Isaac-JetbotRamp-* tasks)
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description="Jetbot ramp smoke test")
parser.add_argument("--task", default="Isaac-JetbotRamp-Teacher-Direct")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=900, help="policy steps (30 Hz)")
parser.add_argument("--mode", choices=["pursuit", "forward", "zero"], default="pursuit")
parser.add_argument("--out", default=os.path.expanduser("~/jetbot_ramp/smoke_out"))
parser.add_argument("--record", action="store_true", help="record mp4 of env 0 (viewer + robot camera if present)")
parser.add_argument("--tag", default="smoke", help="filename prefix for recorded clips")
add_launcher_args(parser)
parser.set_defaults(device=None)
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + hydra_args

env_cfg, _ = resolve_task_config(args_cli.task, "")
env_cfg.scene.num_envs = args_cli.num_envs
if hasattr(env_cfg, "domain_randomization"):
    env_cfg.domain_randomization = False
    env_cfg.events = None
if hasattr(env_cfg, "random_start_prob"):
    env_cfg.random_start_prob = 0.0

args_cli.device = env_cfg.sim.device
os.makedirs(args_cli.out, exist_ok=True)

if args_cli.record:
    args_cli.enable_cameras = True  # load the rendering Kit experience (replicator) for frame capture
    from isaaclab_visualizers.kit import KitVisualizerCfg

    from isaaclab.envs import VideoRecorderCfg

    # overview of env 0: both tracks are ~4.1 x 2.7 m around the env origin
    if args_cli.num_envs == 1:
        # single robot: follow it so it stays visible (the PiP inset covers the top-right quadrant)
        viz = KitVisualizerCfg(
            headless=True, origin_type="asset", origin_track_path="robot", eye=(-0.9, -1.3, 1.1),
            lookat=(0.25, 0.0, 0.0), window_width=1280, window_height=720,
        )
    else:
        # several robots: overview of env 0 (both tracks are ~4.1 x 2.7 m around the env origin)
        viz = KitVisualizerCfg(
            headless=True, origin_type="env", origin_env_index=0, eye=(0.0, -3.6, 3.2), lookat=(0.0, 0.0, 0.2),
            window_width=1280, window_height=720,
        )
    env_cfg.sim.visualizer_cfgs = [viz]
    video_dir = os.path.join(args_cli.out, "videos")
    # recorders number clips instead of overwriting: clear this tag's old clips so _0000 is always the new take
    import glob

    for old in glob.glob(os.path.join(video_dir, f"{args_cli.tag}_*.mp4")):
        os.remove(old)
    env_cfg.video_recorders = [
        VideoRecorderCfg(source="visualizer:kit", output_dir=video_dir, video_length=args_cli.steps,
                         output_filename_prefix=f"{args_cli.tag}_viewer")
    ]
    if hasattr(env_cfg.scene, "front_cam"):
        env_cfg.video_recorders.append(
            VideoRecorderCfg(source="sensor:front_cam", output_dir=video_dir, video_length=args_cli.steps,
                             output_filename_prefix=f"{args_cli.tag}_robotcam")
        )

with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
    env = gym.make(args_cli.task, cfg=env_cfg)
    cleanup.callback(env.close)
    u = env.unwrapped
    obs, _ = env.reset()
    print("[smoke] obs groups:", {k: tuple(v.shape) for k, v in obs.items()})

    from jetbot_ramp.jetbot_cfg import JETBOT_WHEEL_BASE as WHEEL_BASE
    from jetbot_ramp.jetbot_cfg import JETBOT_WHEEL_RADIUS as WHEEL_RADIUS

    v_max = u.cfg.max_wheel_speed * WHEEL_RADIUS * 0.8
    n_success = n_fall = n_timeout = 0
    max_progress = torch.zeros(u.num_envs, device=u.device)
    for step in range(args_cli.steps):
        if args_cli.mode == "zero":
            act = torch.zeros(u.num_envs, 2, device=u.device)
        elif args_cli.mode == "forward":
            act = torch.full((u.num_envs, 2), 0.8, device=u.device)
        else:
            tgt = u.track.lookahead_body(u.track_id, u.progress, u.pos_xy, u.yaw, [0.25])[:, 0]
            curv = 2 * tgt[:, 1] / tgt.norm(dim=-1).square().clamp(min=1e-4)
            v = torch.full((u.num_envs,), v_max, device=u.device)
            w = v * curv
            wl = (v - w * WHEEL_BASE / 2) / WHEEL_RADIUS
            wr = (v + w * WHEEL_BASE / 2) / WHEEL_RADIUS
            act = torch.stack([wl, wr], -1) / u.cfg.max_wheel_speed
            act = act / act.abs().max(dim=-1, keepdim=True).values.clamp(min=1.0)
        obs, rew, term, trunc, extras = env.step(act)
        max_progress = torch.maximum(max_progress, u.progress)
        done = term | trunc
        if done.any():
            # flags are cleared by the auto-reset inside step(); the terminal reward tells finish (+10) from fall (-10)
            n_success += int((term & (rew > 5.0)).sum())
            n_fall += int((term & (rew < -5.0)).sum())
            n_timeout += int((trunc & ~term).sum())
        if step % 30 == 0:
            print(
                f"[smoke] t={step / 30:5.1f}s progress={u.progress[:4].tolist()} lat={u.lateral[:4].tolist()}"
                f" rew={rew[:4].tolist()}"
            )
        if "camera" in obs and step % 15 == 0:
            from isaaclab.sensors import save_images_to_file

            raw = u._camera.data.output["rgb"].torch[:1, ..., :3].float() / 255.0
            save_images_to_file(raw, os.path.join(args_cli.out, f"cam_{step:04d}.png"))

    print(f"[smoke] episodes: success={n_success} fall={n_fall} timeout={n_timeout}")
    print(f"[smoke] max progress per env [m]: {[round(x, 2) for x in max_progress.tolist()]}")
    print(f"[smoke] track length [m]: {u.track.length.tolist()}")
    if args_cli.mode == "pursuit" and n_success == 0:
        print("[smoke] WARNING: no env finished. Check wheel direction (positive action = forward?) and actuator gains.")
