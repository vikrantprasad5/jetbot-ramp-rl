"""Evaluate a checkpoint on every track of the held-out TEST library (no viewer, no respawns).

N robots per track drive once with the mean (deterministic) action; reports per-track success / fall rates and
lap times. Sensor-failure tests: --camera_off / --lidar_off blank that sensor for every robot.

    ./run.sh tools/evaluate.py --task Isaac-JetbotTracks-Eval --agent lidar_cfg_entry_point \
        --ckpt logs/.../model_2999.pt --speed 3 --name A --out logs/eval/A_3x.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
import jetbot_ramp  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-JetbotTracks-Eval")
parser.add_argument("--agent", default="teacher_cfg_entry_point")
parser.add_argument("--ckpt", required=True)
parser.add_argument("--name", default="")
parser.add_argument("--speed", type=float, default=3.0)
parser.add_argument("--per_track", type=int, default=8)
parser.add_argument("--max_seconds", type=float, default=60.0)
parser.add_argument("--camera_off", action="store_true")
parser.add_argument("--lidar_off", action="store_true")
parser.add_argument("--sample", action="store_true", help="sample actions instead of the mean")
parser.add_argument("--out", required=True)
add_launcher_args(parser)
parser.set_defaults(device=None)
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + hydra_args

env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
from jetbot_ramp.tracks import build_library  # noqa: E402

lib = build_library(env_cfg.library)
K = len(lib.tracks)
env_cfg.scene.num_envs = K * args_cli.per_track
env_cfg.track_per_env = True
env_cfg.showcase = True
env_cfg.fixed_speed_scale = args_cli.speed
env_cfg.random_start_prob = 0.0
env_cfg.camera_dropout_prob = 1.0 if args_cli.camera_off else 0.0
env_cfg.lidar_dropout_prob = 1.0 if args_cli.lidar_off else 0.0
env_cfg.domain_randomization = False
env_cfg.events = None
env_cfg.episode_length_s = args_cli.max_seconds + 5.0
args_cli.device = env_cfg.sim.device

with launch_simulation(env_cfg, args_cli), contextlib.ExitStack():
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_rl.rsl_rl.utils import check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    agent_cfg.device = env_cfg.sim.device
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    if args_cli.lidar_off:  # lidar dropout is only applied when the camera isn't dropped; force it
        u.cfg.camera_dropout_prob = 0.0
    wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(wenv, agent_cfg)
    runner.load(os.path.abspath(args_cli.ckpt))
    policy = runner.alg.get_policy()
    policy.eval()
    with torch.inference_mode():
        obs, _ = wenv.reset()
        if args_cli.lidar_off:
            u._lidar_drop[:] = True
        if args_cli.camera_off:
            u._cam_drop[:] = True
        for _ in range(int(args_cli.max_seconds / u.step_dt)):
            obs, _, _, _ = wenv.step(policy(obs, stochastic_output=args_cli.sample))
            if bool((u.status != 0).all()):
                break
    res = {"name": args_cli.name, "ckpt": args_cli.ckpt, "speed": args_cli.speed, "camera_off": args_cli.camera_off,
           "lidar_off": args_cli.lidar_off, "tracks": {}}
    tot_fin = tot_fell = 0
    for k in range(K):
        m = u.track_id == k
        st = u.status[m]
        fin, fell = int((st == 1).sum()), int((st == 2).sum())
        tot_fin += fin
        tot_fell += fell
        laps = u.finish_time[m][st == 1]
        res["tracks"][lib.shapes[k].name] = {
            "robots": int(m.sum()), "finished": fin, "fell": fell,
            "lap_mean_s": float(laps.mean()) if len(laps) else None,
            "progress_median": float(torch.median(u.best_progress[m] / u.lib.length[k])),
        }
    n = u.num_envs
    res["success_rate"] = tot_fin / n
    res["fall_rate"] = tot_fell / n
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    json.dump(res, open(args_cli.out, "w"), indent=1)
    print(f"[eval] {args_cli.name} speed {args_cli.speed}x cam_off={args_cli.camera_off} lidar_off={args_cli.lidar_off}: "
          f"success {res['success_rate'] * 100:.0f}%  falls {res['fall_rate'] * 100:.0f}%")
    for name, r in res["tracks"].items():
        lap = f"{r['lap_mean_s']:.1f} s" if r["lap_mean_s"] else "-"
        print(f"[eval]   {name:16s} {r['finished']}/{r['robots']} finished, {r['fell']} fell, lap {lap}")
    env.close()
