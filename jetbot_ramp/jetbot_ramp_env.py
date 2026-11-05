"""Direct-workflow environment: Jetbot driving along an elevated S-ramp.

Per policy step (30 Hz):

1. ``_pre_physics_step``: clip actions, add noise and random latency, convert to wheel speeds
2. physics x decimation
3. ``_get_dones``: project the robot onto its track's centreline, detect fall / flip / finish
4. ``_get_rewards``: progress along the centreline + shaping + terminal bonus / penalty
5. ``_reset_idx`` for finished envs: sample a track, start position and pose
6. ``_get_observations``: teacher state (+ camera image and proprio for the student)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import wrap_to_pi

from .ramp import TrackTensors, build_tracks, write_ramp_usd

if TYPE_CHECKING:
    from .jetbot_ramp_env_cfg import JetbotRampTeacherEnvCfg


class JetbotRampEnv(DirectRLEnv):
    cfg: JetbotRampTeacherEnvCfg

    def __init__(self, cfg: JetbotRampTeacherEnvCfg, render_mode: str | None = None, **kwargs):
        # generate the ramp USD from the same TrackSpec the reward maths uses
        cfg.scene.ramp.spawn.usd_path = write_ramp_usd(cfg.track)

        expected_obs = 1 + 2 + 2 * len(cfg.lookahead_distances) + 2 + 3 + 2
        if cfg.observation_space != expected_obs:
            raise ValueError(f"observation_space must be {expected_obs} for the configured lookahead.")
        cfg.state_space = expected_obs

        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene["robot"]
        wheel_ids, _ = self.robot.find_joints(self.cfg.wheel_joint_names, preserve_order=True)
        self._wheel_ids = torch.tensor(wheel_ids, device=self.device)

        self.track = TrackTensors(build_tracks(self.cfg.track), device=self.device)
        E, dev = self.num_envs, self.device
        self._half_width = self.cfg.track.width / 2

        # per-env track state
        self.track_id = torch.zeros(E, dtype=torch.long, device=dev)
        self.progress = torch.zeros(E, device=dev)  # s along the centreline [m]
        self.prev_progress = torch.zeros(E, device=dev)
        self.lateral = torch.zeros(E, device=dev)
        self.heading_err = torch.zeros(E, device=dev)
        self.pos_xy = torch.zeros(E, 2, device=dev)  # env-local
        self.yaw = torch.zeros(E, device=dev)
        self._fell = torch.zeros(E, dtype=torch.bool, device=dev)
        self._finished = torch.zeros(E, dtype=torch.bool, device=dev)
        self._just_reset = torch.zeros(E, dtype=torch.bool, device=dev)  # sim velocities are stale until stepped

        # actions
        self.actions = torch.zeros(E, 2, device=dev)
        self.prev_actions = torch.zeros(E, 2, device=dev)
        n_hist = self.cfg.max_action_delay + 1
        self._action_hist = torch.zeros(E, n_hist, 2, device=dev)
        self._delay = torch.zeros(E, dtype=torch.long, device=dev)
        self._wheel_targets = torch.zeros(E, 2, device=dev)

        # camera (student)
        self._camera = None
        if self.cfg.use_camera:
            self._camera = self.scene["front_cam"]
            self._frames = None  # lazily sized on the first image
            self._aug_gain = torch.ones(E, 1, 1, 1, device=dev)
            self._aug_contrast = torch.ones(E, 1, 1, 1, device=dev)
            self._aug_shift = torch.zeros(E, 1, 1, 3, device=dev)

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions[:] = self.actions
        self.actions[:] = actions.clamp(-1.0, 1.0)
        a = self.actions
        if self.cfg.domain_randomization:
            a = (a + self.cfg.action_noise_std * torch.randn_like(a)).clamp(-1.0, 1.0)
        # latency: slot 0 holds the newest action; each env reads its own delay slot
        self._action_hist = torch.roll(self._action_hist, shifts=1, dims=1)
        self._action_hist[:, 0] = a
        delayed = self._action_hist[torch.arange(self.num_envs, device=self.device), self._delay]
        self._wheel_targets[:] = delayed * self.cfg.max_wheel_speed

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target_index(target=self._wheel_targets, joint_ids=self._wheel_ids)

    # ------------------------------------------------------------------
    # track state, dones, rewards
    # ------------------------------------------------------------------

    def _update_track_state(self) -> torch.Tensor:
        """Refresh pose/track quantities from the sim. Returns root height above the env origin."""
        pos_w = self.robot.data.root_pos_w.torch
        origins = self.scene.env_origins
        self.pos_xy[:] = pos_w[:, :2] - origins[:, :2]
        self.yaw[:] = self.robot.data.heading_w.torch
        _, s, lat, tyaw = self.track.project(self.track_id, self.pos_xy)
        self.progress[:] = s
        self.lateral[:] = lat
        self.heading_err[:] = wrap_to_pi(self.yaw - tyaw)
        return pos_w[:, 2] - origins[:, 2]

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._just_reset[:] = False
        z = self._update_track_state()
        fell = z < (self.cfg.track.top_z - self.cfg.fall_margin)
        flipped = self.robot.data.projected_gravity_b.torch[:, 2] > -0.5
        # a wheel past the edge counts as a fall: the chassis can hang on the lip instead of dropping
        off_edge = self.lateral.abs() > (self._half_width - self.cfg.edge_margin)
        fell = fell | off_edge
        finished = self.progress >= (self.track.length[self.track_id] - self.cfg.finish_margin)
        self._fell = fell | flipped
        self._finished = finished & ~self._fell
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self._fell | self._finished, time_out

    def _get_rewards(self) -> torch.Tensor:
        dt = self.step_dt
        ds = (self.progress - self.prev_progress).clamp(-0.1, 0.1)
        self.prev_progress[:] = self.progress
        r_progress = self.cfg.rew_progress * ds
        r_lateral = self.cfg.rew_lateral * self.lateral.square() * dt
        r_heading = self.cfg.rew_heading * (1.0 - torch.cos(self.heading_err)) * dt
        r_action = self.cfg.rew_action_rate * (self.actions - self.prev_actions).square().sum(-1)
        r_term = self.cfg.rew_fall * self._fell.float() + self.cfg.rew_finish * self._finished.float()

        log = self.extras.setdefault("log", {})
        log["Episode_Reward/progress"] = r_progress.mean()
        log["Episode_Reward/lateral"] = r_lateral.mean()
        log["Episode_Reward/heading"] = r_heading.mean()
        log["Episode_Reward/action_rate"] = r_action.mean()
        return r_progress + r_lateral + r_heading + r_action + r_term

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # episode metrics for the envs being reset (no .item(): avoids a GPU sync)
        if self.common_step_counter > 0:
            log = self.extras.setdefault("log", {})
            frac = self.progress[env_ids] / self.track.length[self.track_id[env_ids]]
            log["Metrics/success_rate"] = self._finished[env_ids].float().mean()
            log["Metrics/fall_rate"] = self._fell[env_ids].float().mean()
            log["Metrics/progress_fraction"] = frac.mean()

        super()._reset_idx(env_ids)
        n = len(env_ids)
        dev = self.device

        # track and start position
        if self.cfg.fixed_track >= 0:
            self.track_id[env_ids] = self.cfg.fixed_track
        elif self.cfg.randomize_track:
            self.track_id[env_ids] = torch.randint(0, self.track.num_tracks, (n,), device=dev)
        else:
            self.track_id[env_ids] = 0
        tid = self.track_id[env_ids]
        length = self.track.length[tid]
        s0 = torch.full((n,), self.cfg.start_s, device=dev)
        rand_start = torch.rand(n, device=dev) < self.cfg.random_start_prob
        s_rand = self.cfg.start_s + torch.rand(n, device=dev) * (self.cfg.random_start_max_frac * length - self.cfg.start_s)
        s0 = torch.where(rand_start, s_rand, s0)

        xy, tyaw = self.track.pose_at(tid, s0)
        lat0 = (torch.rand(n, device=dev) * 2 - 1) * self.cfg.reset_lateral_noise
        nrm = torch.stack([-torch.sin(tyaw), torch.cos(tyaw)], dim=-1)
        xy = xy + lat0[:, None] * nrm
        yaw0 = tyaw + (torch.rand(n, device=dev) * 2 - 1) * self.cfg.reset_yaw_noise

        # write root pose (x, y, z, qx, qy, qz, qw) and zero velocities
        pose = torch.zeros(n, 7, device=dev)
        pose[:, :2] = xy + self.scene.env_origins[env_ids, :2]
        pose[:, 2] = self.scene.env_origins[env_ids, 2] + self.cfg.track.top_z + self.cfg.spawn_height
        pose[:, 5] = torch.sin(yaw0 / 2)
        pose[:, 6] = torch.cos(yaw0 / 2)
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=env_ids)
        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        # cached track state for the first observation (sim buffers refresh on the next step)
        self.pos_xy[env_ids] = xy
        self.yaw[env_ids] = yaw0
        self.progress[env_ids] = s0
        self.prev_progress[env_ids] = s0
        self.lateral[env_ids] = lat0
        self.heading_err[env_ids] = wrap_to_pi(yaw0 - tyaw)
        self._fell[env_ids] = False
        self._finished[env_ids] = False
        self._just_reset[env_ids] = True

        # actions / latency
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._action_hist[env_ids] = 0.0
        if self.cfg.domain_randomization and self.cfg.max_action_delay > 0:
            self._delay[env_ids] = torch.randint(0, self.cfg.max_action_delay + 1, (n,), device=dev)
        else:
            self._delay[env_ids] = 0

        # camera: frame stack and per-episode image augmentation
        if self._camera is not None:
            if self._frames is not None:
                self._frames[env_ids] = 0.0
            if self.cfg.image_aug:
                c = self.cfg
                self._aug_gain[env_ids] = _uniform(n, *c.aug_brightness, dev).view(n, 1, 1, 1)
                self._aug_contrast[env_ids] = _uniform(n, *c.aug_contrast, dev).view(n, 1, 1, 1)
                self._aug_shift[env_ids] = _uniform(n * 3, -c.aug_color_shift, c.aug_color_shift, dev).view(n, 1, 1, 3)

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------

    def _teacher_obs(self) -> torch.Tensor:
        la = self.track.lookahead_body(
            self.track_id, self.progress, self.pos_xy, self.yaw, self.cfg.lookahead_distances
        )  # (E, M, 2)
        wheel_vel = self.robot.data.joint_vel.torch[:, self._wheel_ids] / self.cfg.max_wheel_speed
        lin_b = self.robot.data.root_lin_vel_b.torch[:, :2]
        ang_z = self.robot.data.root_ang_vel_b.torch[:, 2:3]
        fresh = (~self._just_reset).float()[:, None]
        wheel_vel, lin_b, ang_z = wheel_vel * fresh, lin_b * fresh, ang_z * fresh
        return torch.cat(
            [
                (self.lateral / self._half_width)[:, None],
                torch.sin(self.heading_err)[:, None],
                torch.cos(self.heading_err)[:, None],
                la.reshape(self.num_envs, -1) / 0.5,
                wheel_vel,
                lin_b / 0.4,
                ang_z / 4.0,
                self.actions,
            ],
            dim=-1,
        )

    def _camera_obs(self) -> torch.Tensor:
        img = self._camera.data.output["rgb"].torch[..., :3].float() / 255.0  # (E, H, W, 3)
        k = self.cfg.camera_downsample
        if k > 1:
            img = F.avg_pool2d(img.permute(0, 3, 1, 2), k).permute(0, 2, 3, 1)
        if self.cfg.image_aug:
            mean = img.mean(dim=(1, 2, 3), keepdim=True)
            img = (img - mean) * self._aug_contrast + mean
            img = img * self._aug_gain + self._aug_shift
            img = img + self.cfg.aug_noise_std * torch.randn_like(img)
            img = img.clamp(0.0, 1.0)
        if self.cfg.write_image_to_file:
            from isaaclab.sensors import save_images_to_file

            save_images_to_file(img[:1], f"jetbot_cam_{int(self.common_step_counter):05d}.png")
        # per-image mean removal, channel-first
        img = (img - img.mean(dim=(1, 2), keepdim=True)).permute(0, 3, 1, 2)  # (E, 3, H, W)
        if self.cfg.frame_stack <= 1:
            return img
        if self._frames is None:
            E, C, H, W = img.shape
            self._frames = torch.zeros(E, C * self.cfg.frame_stack, H, W, device=self.device)
            self._frames[:] = img.repeat(1, self.cfg.frame_stack, 1, 1)
        else:
            self._frames = torch.roll(self._frames, shifts=img.shape[1], dims=1)
            self._frames[:, : img.shape[1]] = img
        return self._frames.clone()

    def _get_observations(self) -> dict:
        state = self._teacher_obs()
        obs = {"policy": state, "critic": state}
        if self._camera is not None:
            proprio = [self.actions]
            if self.cfg.student_use_wheel_vel:
                proprio.append(self.robot.data.joint_vel.torch[:, self._wheel_ids] / self.cfg.max_wheel_speed)
            obs["camera"] = self._camera_obs()
            obs["proprio"] = torch.cat(proprio, dim=-1)
        return obs


def _uniform(n: int, lo: float, hi: float, device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, device=device)
