"""Score every checkpoint of a run (no rendering) and pick generations for a swarm video.

Each checkpoint: N robots, showcase mode (no respawn), same track and speed as the video, sampled actions.
Selection: a few generations before the swarm reaches --dense_from (fraction of track), then one generation
per --step gain in median progress, plus the final checkpoint.

    ./run.sh tools/progress_scan.py --task Isaac-JetbotTracks-Eval --agent teacher_cfg_entry_point \
        --run logs/rsl_rl/tracks_teacher_progress/<run> --track 1 --speed 2 --out logs/scan_teacher.json
Prints a ready-to-use list of --ckpt arguments for tools/swarm.py.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import re
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
parser.add_argument("--run", required=True)
parser.add_argument("--track", type=int, default=1)
parser.add_argument("--speed", type=float, default=2.0)
parser.add_argument("--robots", type=int, default=64)
parser.add_argument("--max_seconds", type=float, default=14.0)
parser.add_argument("--every", type=int, default=1, help="scan every k-th checkpoint")
parser.add_argument("--dense_from", type=float, default=0.30)
parser.add_argument("--step", type=float, default=0.05)
parser.add_argument("--early", type=int, default=4)
parser.add_argument("--max_gens", type=int, default=20)
parser.add_argument("--out", required=True)
add_launcher_args(parser)
parser.set_defaults(device=None)
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + hydra_args

env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
env_cfg.scene.num_envs = args_cli.robots
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
env_cfg.episode_length_s = args_cli.max_seconds + 5.0
args_cli.device = env_cfg.sim.device

ckpts = sorted(glob.glob(os.path.join(args_cli.run, "model_*.pt")), key=lambda p: int(re.findall(r"model_(\d+)", p)[0]))
ckpts = ckpts[:: args_cli.every] + ([ckpts[-1]] if (len(ckpts) - 1) % args_cli.every else [])

with launch_simulation(env_cfg, args_cli), contextlib.ExitStack():
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_rl.rsl_rl.utils import check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    agent_cfg.device = env_cfg.sim.device
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(wenv, agent_cfg)
    length = float(u.lib.length[args_cli.track])
    rows = []
    for path in ckpts:
        it = int(re.findall(r"model_(\d+)", path)[0])
        runner.load(os.path.abspath(path))
        pol = runner.alg.get_policy()
        pol.eval()
        with torch.inference_mode():
            obs, _ = wenv.reset()
            for _ in range(int(args_cli.max_seconds / u.step_dt)):
                obs, _, _, _ = wenv.step(pol(obs, stochastic_output=True))
                if bool((u.status != 0).all()):
                    break
        med = float(torch.median(u.best_progress / length))
        fin = float((u.status == 1).float().mean())
        rows.append({"iter": it, "path": path, "median": med, "finished": fin})
        print(f"[scan] iter {it:5d}: median progress {med * 100:5.1f}%  finished {fin * 100:5.1f}%")

    # selection
    sel = []
    pre = [r for r in rows if r["median"] < args_cli.dense_from]
    if pre:
        idx = sorted({round(i * (len(pre) - 1) / max(1, args_cli.early - 1)) for i in range(args_cli.early)})
        sel += [pre[i] for i in idx]
    last = None
    for r in rows:
        if r["median"] < args_cli.dense_from:
            continue
        score = r["median"] + r["finished"]  # keeps advancing after median saturates (more robots finishing)
        if last is None or score - last >= args_cli.step:
            sel.append(r)
            last = score
    if rows and sel[-1] is not rows[-1]:
        sel.append(rows[-1])
    if len(sel) > args_cli.max_gens:  # thin evenly, keep first and last
        keep = sorted({round(i * (len(sel) - 1) / (args_cli.max_gens - 1)) for i in range(args_cli.max_gens)})
        sel = [sel[i] for i in keep]
    json.dump({"rows": rows, "selected": sel}, open(args_cli.out, "w"), indent=1)
    print("[scan] selected iterations:", [r["iter"] for r in sel])
    print("[scan] SWARM_ARGS " + " ".join(f'--ckpt "Gen {k + 1} (iter {r["iter"]})={r["path"]}"' for k, r in enumerate(sel)))
    env.close()
