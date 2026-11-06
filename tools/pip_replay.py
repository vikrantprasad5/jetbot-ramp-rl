"""Single-robot clip with robot-camera + LiDAR insets, camera-safe (drive without a viewer, then replay).

    ./run.sh tools/pip_replay.py --agent finetune_cfg_entry_point --ckpt logs/.../model_499.pt \
        --track 1 --speed 3 --seconds 15 --caption "Model C - triple S at 3x" --out videos/fusion/C_pip.mp4

The drive phase uses the camera eval task for every model (non-camera models simply ignore the image), so the
inset always shows what the robot's camera sees. Extra flags: --camera_off / --lidar_off (sensor-failure demo).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-JetbotTracks-Camera-Eval")
parser.add_argument("--agent", required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--track", type=int, default=1)
parser.add_argument("--speed", type=float, default=3.0)
parser.add_argument("--seconds", type=float, default=15.0)
parser.add_argument("--camera_off", action="store_true")
parser.add_argument("--lidar_off", action="store_true")
parser.add_argument("--caption", default="")
parser.add_argument("--save_obs", default="", help="save every network input + action of the drive to this .npz")
parser.add_argument("--out", required=True)
parser.add_argument("--phase", choices=["all", "drive", "render"], default="all")
parser.add_argument("--work", default="")
HERE = os.path.dirname(os.path.abspath(__file__))


def cfg_for(task, agent, args):
    from isaaclab_tasks.utils import resolve_task_config

    env_cfg, agent_cfg = resolve_task_config(task, agent)
    env_cfg.scene.num_envs = 1
    env_cfg.showcase = True
    env_cfg.fixed_track = args.track
    env_cfg.fixed_speed_scale = args.speed
    env_cfg.random_start_prob = 0.0
    env_cfg.reset_lateral_noise = 0.0
    env_cfg.reset_yaw_noise = 0.0
    env_cfg.camera_dropout_prob = 0.0
    env_cfg.lidar_dropout_prob = 0.0
    env_cfg.domain_randomization = False
    env_cfg.events = None
    env_cfg.episode_length_s = args.seconds + 5.0
    return env_cfg, agent_cfg


def drive(args):
    import gymnasium as gym
    import numpy as np
    import torch

    from isaaclab.app import launch_simulation

    sys.path.insert(0, HERE)
    import viz

    env_cfg, agent_cfg = cfg_for(args.task, args.agent, args)
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
        runner.load(os.path.abspath(args.ckpt))
        policy = runner.alg.get_policy()
        policy.eval()
        lw = viz.LidarPlotWriter(os.path.join(args.work, "lidar.mp4"))
        cam = None
        poses, joints = [], []
        rec = {"camera": [], "lidar": [], "cliff": [], "proprio": [], "action": []}
        with torch.inference_mode():
            obs, _ = wenv.reset()
            u._cam_drop[:] = args.camera_off
            u._lidar_drop[:] = args.lidar_off
            for _ in range(int(args.seconds / u.step_dt)):
                act = policy(obs)
                if args.save_obs:
                    for k in ("camera", "lidar", "cliff", "proprio"):
                        if k in obs.keys():
                            rec[k].append(obs[k][0].cpu().numpy().astype(np.float32))
                    rec["action"].append(act[0].cpu().numpy())
                obs, _, _, _ = wenv.step(act)
                p = u.robot.data.root_pose_w.torch.clone()
                p[:, :3] -= u.scene.env_origins
                poses.append(p.cpu().numpy())
                joints.append(u.robot.data.joint_pos.torch.cpu().numpy())
                lw.add(*viz.lidar_sample(u, 0))
                if len(poses) % 30 == 0:
                    print(f"[pip] t={len(poses) * u.step_dt:4.1f}s progress={float(u.progress[0]):.2f}/{float(u.lib.length[u.track_id[0]]):.2f} "
                          f"v={float(u.robot.data.root_lin_vel_b.torch[0, 0]):.2f} status={int(u.status[0])} speed_scale={float(u.speed_scale[0])}", flush=True)
                img = u._camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy().astype(np.uint8)
                if args.camera_off:
                    img = (img * 0.15).astype(np.uint8)  # show it's blanked for the policy
                if cam is None:
                    h, w = img.shape[:2]
                    cam = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                                            "-r", str(round(1 / u.step_dt)), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                            "-crf", "18", os.path.join(args.work, "cam.mp4")], stdin=subprocess.PIPE)
                cam.stdin.write(np.ascontiguousarray(img).tobytes())
        cam.stdin.close()
        cam.wait()
        lw.close()
        st = int(u.status[0])
        res = "finished in " + format(float(u.finish_time[0]), ".1f") + " s" if st == 1 else ("fell" if st == 2 else "still driving")
        np.savez_compressed(os.path.join(args.work, "traj.npz"), poses=np.stack(poses), joints=np.stack(joints))
        if args.save_obs:
            np.savez_compressed(args.save_obs, **{k: np.stack(v) for k, v in rec.items() if v})
        json.dump({"result": res, "dt": u.step_dt}, open(os.path.join(args.work, "meta.json"), "w"))
        print(f"[pip] {res}", flush=True)
        env.close()


def render(args):
    import gymnasium as gym
    import numpy as np
    import torch

    from isaaclab.app import launch_simulation

    sys.path.insert(0, HERE)
    import viz

    task = args.task.replace("-Camera-", "-")
    env_cfg, _ = cfg_for(task, "rsl_rl_cfg_entry_point", args)
    from isaaclab_visualizers.kit import KitVisualizerCfg

    env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(headless=True, origin_type="asset", origin_track_path="robot",
                                                    eye=(-0.9, -1.3, 1.1), lookat=(0.25, 0.0, 0.0),
                                                    window_width=1280, window_height=720)]
    args.enable_cameras = True
    args.device = env_cfg.sim.device
    meta = json.load(open(os.path.join(args.work, "meta.json")))
    data = np.load(os.path.join(args.work, "traj.npz"))
    with launch_simulation(env_cfg, args), contextlib.ExitStack():
        env = gym.make(task, cfg=env_cfg)
        u = env.unwrapped
        env.reset()
        viz_ = [v for v in u.sim.visualizers if getattr(v.cfg, "visualizer_type", "") == "kit"][0]
        ids = torch.arange(1, device=u.device)
        o = u.scene.env_origins[0]
        view = os.path.join(args.work, "viewer.mp4")
        proc = None
        with torch.inference_mode():
            for pose, jp in zip(data["poses"], data["joints"]):
                p = torch.tensor(pose, device=u.device)
                p[:, :3] += o
                u.robot.write_root_pose_to_sim_index(root_pose=p, env_ids=ids)
                u.robot.write_joint_position_to_sim_index(position=torch.tensor(jp, device=u.device), env_ids=ids)
                u.scene.write_data_to_sim()
                u.sim.forward()
                u.scene.update(dt=u.physics_dt)  # refresh asset data so the follow camera tracks the replayed pose
                f = viz_.render_rgb_array()
                if f is None:
                    continue
                f = np.ascontiguousarray(np.asarray(f)[..., :3]).astype(np.uint8)
                if proc is None:
                    H, W = f.shape[:2]
                    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                             "-r", str(round(1 / meta["dt"])), "-i", "-", "-c:v", "libx264", "-pix_fmt",
                                             "yuv420p", "-crf", "18", view], stdin=subprocess.PIPE)
                proc.stdin.write(f.tobytes())
        proc.stdin.close()
        proc.wait()
        cap = f"{args.caption}  |  {meta['result']}" if args.caption else meta["result"]
        viz.compose(view, os.path.abspath(args.out), cam=os.path.join(args.work, "cam.mp4"),
                    lidar=os.path.join(args.work, "lidar.mp4"), caption=cap.replace(":", "\\:").replace(",", "\\,"))
        print(f"[pip] wrote {args.out}", flush=True)
        env.close()


if __name__ == "__main__":
    from isaaclab.app import add_launcher_args

    add_launcher_args(parser)
    parser.set_defaults(device=None)
    from isaaclab_tasks.utils import setup_preset_cli

    args, hydra_args = setup_preset_cli(parser, sys.argv[1:])
    if args.phase == "all":
        out = os.path.abspath(args.out)
        work = args.work or os.path.join(os.path.dirname(out), "_raw", os.path.splitext(os.path.basename(out))[0] + "_pip")
        os.makedirs(work, exist_ok=True)
        for ph in ("drive", "render"):
            r = subprocess.run([os.path.join(HERE, "..", "run.sh"), os.path.abspath(__file__), *sys.argv[1:], "--work", work,
                                "--phase", ph])
            if r.returncode:
                sys.exit(f"[pip] {ph} failed")
        sys.exit(0)
    sys.argv = [sys.argv[0]] + hydra_args
    import isaaclab_tasks  # noqa: F401
    import jetbot_ramp  # noqa: F401

    drive(args) if args.phase == "drive" else render(args)
