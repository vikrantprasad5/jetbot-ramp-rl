"""Swarm "generations" video: for each checkpoint, N robots start together and drive with the policy's own
sampling noise; fallen robots stay down, finished ones stop. One captioned segment per generation, joined.

Two separate processes (run automatically):
  1. drive  - robots drive on their OWN copy of the track (normal env spacing), NO viewer. This matches
              evaluation conditions exactly: an active viewer changes what camera sensors render, and stacked
              robots would see each other. Trajectories + stats are saved to disk.
  2. render - a camera-free session replays all trajectories onto one track (kinematic, no physics) and
              records the viewer.

    ./run.sh tools/swarm.py --task Isaac-JetbotTracks-Eval --agent teacher_cfg_entry_point \
        --ckpt "Gen 1 (iter 0)=logs/.../model_0.pt" --ckpt "Gen 2 (iter 20)=logs/.../model_20.pt" ... \
        --track 1 --speed 2 --out videos/fusion/teacher_generations.mp4
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-JetbotTracks-Eval")
parser.add_argument("--render_task", default="", help="camera-free task for replay (default: derived from --task)")
parser.add_argument("--agent", default="teacher_cfg_entry_point")
parser.add_argument("--ckpt", action="append", default=[], help="'label=path' (repeatable, in order)")
parser.add_argument("--track", type=int, default=1)
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--robots", type=int, default=60)
parser.add_argument("--max_seconds", type=float, default=14.0)
parser.add_argument("--hold_seconds", type=float, default=1.0)
parser.add_argument("--mean_action", action="store_true", help="use the mean action instead of sampling")
parser.add_argument("--title", default="")
parser.add_argument("--out", required=True)
parser.add_argument("--phase", choices=["all", "drive", "render"], default="all")
parser.add_argument("--work", default="")

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..", "run.sh")


def main_all(args):
    out = os.path.abspath(args.out)
    work = args.work or os.path.join(os.path.dirname(out), "_raw", os.path.splitext(os.path.basename(out))[0] + "_traj")
    os.makedirs(work, exist_ok=True)
    base = [RUN, os.path.abspath(__file__), *sys.argv[1:], "--work", work]
    for phase in ("drive", "render"):
        r = subprocess.run(base + ["--phase", phase])
        if r.returncode != 0:
            sys.exit(f"[swarm] {phase} phase failed ({r.returncode})")


def _common_cfg(task, agent, args, num_envs):
    from isaaclab_tasks.utils import resolve_task_config

    env_cfg, agent_cfg = resolve_task_config(task, agent)
    env_cfg.scene.num_envs = num_envs
    env_cfg.showcase = True
    env_cfg.fixed_track = args.track
    env_cfg.fixed_speed_scale = args.speed
    env_cfg.random_start_prob = 0.0
    env_cfg.reset_lateral_noise = 0.05
    env_cfg.reset_yaw_noise = 0.3
    env_cfg.camera_dropout_prob = 0.0
    env_cfg.lidar_dropout_prob = 0.0
    env_cfg.domain_randomization = False
    env_cfg.events = None
    env_cfg.episode_length_s = args.max_seconds + args.hold_seconds + 5.0
    return env_cfg, agent_cfg


def main_drive(args):
    import gymnasium as gym
    import numpy as np
    import torch

    from isaaclab.app import launch_simulation

    env_cfg, agent_cfg = _common_cfg(args.task, args.agent, args, args.robots)
    args.device = env_cfg.sim.device
    with launch_simulation(env_cfg, args), contextlib.ExitStack():
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_rl.rsl_rl.utils import check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
        agent_cfg.device = env_cfg.sim.device
        env = gym.make(args.task, cfg=env_cfg)
        u = env.unwrapped
        wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(wenv, agent_cfg)
        length = float(u.lib.length[args.track])
        max_steps = int(args.max_seconds / u.step_dt)
        hold = int(args.hold_seconds / u.step_dt)
        meta = []
        for gi, spec in enumerate(args.ckpt):
            label, path = spec.split("=", 1)
            runner.load(os.path.abspath(path))
            policy = runner.alg.get_policy()
            policy.eval()
            poses, joints = [], []
            done_at = None
            with torch.inference_mode():
                obs, _ = wenv.reset()
                for step in range(max_steps):
                    obs, _, _, _ = wenv.step(policy(obs, stochastic_output=not args.mean_action))
                    p = u.robot.data.root_pose_w.torch.clone()
                    p[:, :3] -= u.scene.env_origins
                    poses.append(p.cpu().numpy())
                    joints.append(u.robot.data.joint_pos.torch.cpu().numpy())
                    if done_at is None and bool((u.status != 0).all()):
                        done_at = step
                    if done_at is not None and step - done_at >= hold:
                        break
            n_fin, n_fell = int((u.status == 1).sum()), int((u.status == 2).sum())
            med = float(torch.median(u.best_progress / length)) * 100
            ft = u.finish_time[u.status == 1]
            best = f", best lap {float(ft.min()):.1f} s" if len(ft) else ""
            stats = f"{label}  |  {args.robots} robots: {n_fin} finished, {n_fell} fell, median {med:.0f}% of track{best}"
            print(f"[swarm] {stats}", flush=True)
            np.savez_compressed(os.path.join(args.work, f"gen_{gi:02d}.npz"), poses=np.stack(poses), joints=np.stack(joints))
            meta.append({"label": label, "stats": stats, "dt": u.step_dt})
        json.dump(meta, open(os.path.join(args.work, "meta.json"), "w"), indent=1)
        env.close()


def main_render(args):
    import gymnasium as gym
    import numpy as np
    import torch

    from isaaclab.app import launch_simulation

    sys.path.insert(0, HERE)
    from viz import FONT, concat

    from jetbot_ramp.tracks import build_library

    task = args.render_task or args.task.replace("-Camera-", "-")
    env_cfg, _ = _common_cfg(task, "rsl_rl_cfg_entry_point", args, args.robots)
    env_cfg.use_camera = False
    lib = build_library(env_cfg.library)
    pts = lib.tracks[args.track].points
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    span = float(max(hi - lo)) + 0.6
    start = pts[0]
    d = c - start
    dn = d / (np.linalg.norm(d) + 1e-6)
    look = start + 0.55 * d
    eye = start - dn * (0.55 * span)
    from isaaclab_visualizers.kit import KitVisualizerCfg

    env_cfg.sim.visualizer_cfgs = [
        KitVisualizerCfg(headless=True, origin_type="env", origin_env_index=0,
                         eye=(float(eye[0]), float(eye[1]), 0.3 + 0.75 * span),
                         lookat=(float(look[0]), float(look[1]), 0.3), window_width=1280, window_height=720)
    ]
    args.enable_cameras = True
    args.device = env_cfg.sim.device
    meta = json.load(open(os.path.join(args.work, "meta.json")))
    out = os.path.abspath(args.out)
    with launch_simulation(env_cfg, args), contextlib.ExitStack():
        env = gym.make(task, cfg=env_cfg)
        u = env.unwrapped
        env.reset()
        viz = [v for v in u.sim.visualizers if getattr(v.cfg, "visualizer_type", "") == "kit"][0]
        ids = torch.arange(u.num_envs, device=u.device)
        o0 = u.scene.env_origins[0]
        parts = []
        for gi, m in enumerate(meta):
            data = np.load(os.path.join(args.work, f"gen_{gi:02d}.npz"))
            seg = os.path.join(args.work, f"seg_{gi:02d}.mp4")
            txt = m["stats"].replace(":", "\\:").replace(",", "\\,")
            vf = (f"drawtext=fontfile={FONT}:expansion=none:text='{txt}':x=16:y=h-44:fontsize=22:fontcolor=white:box=1:"
                  f"boxcolor=black@0.6:boxborderw=8")
            if args.title:
                t = args.title.replace(":", "\\:").replace(",", "\\,")
                vf += (f",drawtext=fontfile={FONT}:expansion=none:text='{t}':x=16:y=16:fontsize=24:fontcolor=white:box=1:"
                       f"boxcolor=black@0.6:boxborderw=8")
            proc = None
            with torch.inference_mode():
                for pose, jp in zip(data["poses"], data["joints"]):
                    p = torch.tensor(pose, device=u.device)
                    p[:, :3] += o0
                    u.robot.write_root_pose_to_sim_index(root_pose=p, env_ids=ids)
                    u.robot.write_joint_position_to_sim_index(position=torch.tensor(jp, device=u.device), env_ids=ids)
                    u.scene.write_data_to_sim()
                    u.sim.forward()
                    f = viz.render_rgb_array()
                    if f is None:
                        continue
                    f = np.ascontiguousarray(np.asarray(f)[..., :3]).astype(np.uint8)
                    if proc is None:
                        H, W = f.shape[:2]
                        proc = subprocess.Popen(
                            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r",
                             str(round(1 / m["dt"])), "-i", "-", "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                             "-crf", "20", seg], stdin=subprocess.PIPE)
                    proc.stdin.write(f.tobytes())
            proc.stdin.close()
            proc.wait()
            parts.append(seg)
        concat(parts, out)
        with open(out.replace(".mp4", ".txt"), "w") as fh:
            fh.write("\n".join(m["stats"] for m in meta) + "\n")
        print(f"[swarm] wrote {out}", flush=True)
        env.close()


if __name__ == "__main__":
    from isaaclab.app import add_launcher_args

    add_launcher_args(parser)
    parser.set_defaults(device=None)
    from isaaclab_tasks.utils import setup_preset_cli

    args, hydra_args = setup_preset_cli(parser, sys.argv[1:])
    if args.phase == "all":
        main_all(args)
        sys.exit(0)
    sys.argv = [sys.argv[0]] + hydra_args
    import isaaclab_tasks  # noqa: F401
    import jetbot_ramp  # noqa: F401

    os.makedirs(args.work, exist_ok=True)
    if args.phase == "drive":
        main_drive(args)
    else:
        main_render(args)
