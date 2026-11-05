"""Record a trained (or partly trained) policy driving ONE robot, as a picture-in-picture mp4.

The viewer follows the robot; the robot's camera is inset in the top-right quadrant.

    cd ~/jetbot_ramp
    ./run.sh tools/record_policy.py --task Isaac-JetbotRamp-Teacher-Direct-Play \
        --checkpoint logs/rsl_rl/jetbot_ramp_teacher/<run>/model_40.pt --steps 600 \
        --out videos/phase2_teacher_iter040.mp4 --caption "Teacher after 40 PPO iterations"

Works for the student too (``--task Isaac-JetbotRamp-Distill-Direct-Play`` and a distillation checkpoint).
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import os
import subprocess
import sys

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
import jetbot_ramp  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Record one robot driving with a checkpoint (PiP video)")
parser.add_argument("--task", default="Isaac-JetbotRamp-Teacher-Direct-Play")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--steps", type=int, default=600, help="policy steps at 30 Hz (600 = 20 s)")
parser.add_argument("--out", required=True, help="output mp4 path")
parser.add_argument("--caption", default="")
parser.add_argument("--track", type=int, default=-1, help="force track 0 (S) or 1 (mirrored); -1 = random")
parser.add_argument("--seed", type=int, default=0)
add_launcher_args(parser)
parser.set_defaults(device=None)
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + hydra_args
args_cli.enable_cameras = True

env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
env_cfg.scene.num_envs = 1
env_cfg.seed = args_cli.seed
env_cfg.random_start_prob = 0.0
env_cfg.fixed_track = args_cli.track
args_cli.device = env_cfg.sim.device

from isaaclab_visualizers.kit import KitVisualizerCfg  # noqa: E402

from isaaclab.envs import VideoRecorderCfg  # noqa: E402

out = os.path.abspath(args_cli.out)
tmp_dir = os.path.join(os.path.dirname(out), "_raw")
tag = os.path.splitext(os.path.basename(out))[0]
os.makedirs(tmp_dir, exist_ok=True)
for old in glob.glob(os.path.join(tmp_dir, f"{tag}_*.mp4")):
    os.remove(old)
env_cfg.sim.visualizer_cfgs = [
    KitVisualizerCfg(
        headless=True, origin_type="asset", origin_track_path="robot", eye=(-0.9, -1.3, 1.1),
        lookat=(0.25, 0.0, 0.0), window_width=1280, window_height=720,
    )
]
env_cfg.video_recorders = [
    VideoRecorderCfg(source="visualizer:kit", output_dir=tmp_dir, video_length=args_cli.steps,
                     output_filename_prefix=f"{tag}_viewer"),
    VideoRecorderCfg(source="sensor:front_cam", output_dir=tmp_dir, video_length=args_cli.steps,
                     output_filename_prefix=f"{tag}_robotcam"),
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
    runner.load(os.path.abspath(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=u.device)
    obs = wenv.get_observations()
    n_finish = n_fall = 0
    with torch.inference_mode():
        for _ in range(args_cli.steps + 2):  # +2: the recorder closes the clip after video_length steps
            obs, rew, dones, _ = wenv.step(policy(obs))
            policy.reset(dones)
            if dones.any():
                n_finish += int((rew > 5).sum())
                n_fall += int((rew < -5).sum())
    print(f"[record] {tag}: finished={n_finish} fell={n_fall} in {args_cli.steps / 30:.0f} s")

    # Kit exits the process when the simulation context closes, so compose here (clips are already closed)
    viewer = sorted(glob.glob(os.path.join(tmp_dir, f"{tag}_viewer_*.mp4")))[-1]
    cam = sorted(glob.glob(os.path.join(tmp_dir, f"{tag}_robotcam_*.mp4")))[-1]
    caption = args_cli.caption
    if caption:
        caption = f"{caption}  |  finished {n_finish}, fell {n_fall}"
    subprocess.run([os.path.join(HERE, "pip_compose.sh"), viewer, cam, out, caption], check=True)
    print(f"[record] wrote {out}")
    env.close()
