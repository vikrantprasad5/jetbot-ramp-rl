"""Direct-workflow env: Jetbot on a library of procedural elevated tracks, with LiDAR / cliff / camera sensors,
a speed curriculum (1x -> 3x) and a showcase mode for swarm videos and evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, wrap_to_pi

from .tracks import LibraryTensors, build_library, rasterize_library, write_library_usd

if TYPE_CHECKING:
    from .tracks_env_cfg import TracksEnvCfg

BASE_SPEED_MPS = 0.36  # 1x cap in m/s, for the lap-time bonus


class TracksEnv(DirectRLEnv):
    cfg: TracksEnvCfg

    def __init__(self, cfg: TracksEnvCfg, render_mode: str | None = None, **kwargs):
        self.library = build_library(cfg.library)
        cfg.scene.ramp.spawn.usd_path = write_library_usd(self.library)
        expected = 1 + 2 + 2 * len(cfg.lookahead_distances) + 2 + 2 + 1 + 2 + 1 + 1
        if cfg.observation_space != expected:
            raise ValueError(f"observation_space must be {expected}")
        cfg.state_space = expected
        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene["robot"]
        ids, _ = self.robot.find_joints(self.cfg.wheel_joint_names, preserve_order=True)
        self._wheel_ids = torch.tensor(ids, device=self.device)
        self.lib = LibraryTensors(self.library, device=self.device)
        E, dev = self.num_envs, self.device
        self._max_wheel = self.cfg.base_wheel_speed * max(self.cfg.speed_levels)

        z = lambda *s, **k: torch.zeros(*s, device=dev, **k)  # noqa: E731
        self.track_id = z(E, dtype=torch.long)
        self.progress, self.prev_progress, self.start_progress = z(E), z(E), z(E)
        self.lateral, self.heading_err, self.yaw = z(E), z(E), z(E)
        self.pos_xy = z(E, 2)
        self.speed_scale = torch.ones(E, device=dev)
        self._fell = z(E, dtype=torch.bool)
        self._finished = z(E, dtype=torch.bool)
        self._just_reset = z(E, dtype=torch.bool)
        self.actions, self.prev_actions = z(E, 2), z(E, 2)
        self._action_hist = z(E, self.cfg.max_action_delay + 1, 2)
        self._delay = z(E, dtype=torch.long)
        self._wheel_targets = z(E, 2)
        self._cam_drop = z(E, dtype=torch.bool)
        self._lidar_drop = z(E, dtype=torch.bool)

        # speed curriculum (shared): index of the highest unlocked level
        self.speed_level = self.cfg.start_level
        self._level_ema = 0.0
        self._level_episodes = 0
        # per-track success EMA (for logging which shapes are hard)
        self._track_success = torch.zeros(self.lib.num_tracks, device=dev)

        # showcase bookkeeping: 0 running, 1 finished, 2 fell
        self.status = z(E, dtype=torch.long)
        self.finish_time = torch.full((E,), float("nan"), device=dev)
        self.best_progress = z(E)

        # analytic range sensors
        grid, origin, res = rasterize_library(self.library)
        self._grid = torch.tensor(grid, device=dev)
        self._grid_origin = torch.tensor(origin, device=dev)
        self._grid_res = res
        self._lidar_dirs, self._lidar_org = self._fan(self.cfg.lidar_pos, self.cfg.lidar_pitch_deg,
                                                      torch.arange(self.cfg.lidar_fov_deg[0], self.cfg.lidar_fov_deg[1] + 1e-6, self.cfg.lidar_res_deg))
        self._cliff_dirs, self._cliff_org = self._fan(self.cfg.cliff_pos, self.cfg.cliff_pitch_deg, torch.tensor(self.cfg.cliff_yaws_deg))
        self.lidar_range = torch.full((E, self._lidar_dirs.shape[0]), self.cfg.lidar_max_range, device=dev)
        self.lidar_kind = torch.zeros(E, self._lidar_dirs.shape[0], dtype=torch.long, device=dev)  # 1 ramp, 0 floor, -1 miss
        self.lidar_hits = torch.zeros(E, self._lidar_dirs.shape[0], 3, device=dev)  # env-local
        self.cliff_range = torch.full((E, 3), self.cfg.cliff_max_range, device=dev)
        self.cliff_kind = torch.zeros(E, 3, dtype=torch.long, device=dev)
        self.cliff_hits = torch.zeros(E, 3, 3, device=dev)
        self._lidar = self.cfg.use_lidar
        self._cliff = self.cfg.use_cliff
        self._camera = self.scene["front_cam"] if self.cfg.use_camera else None
        self._frames = None
        self._aug_gain = torch.ones(E, 1, 1, 1, device=dev)
        self._aug_contrast = torch.ones(E, 1, 1, 1, device=dev)
        self._aug_shift = z(E, 1, 1, 3)

    # ------------------------------------------------------------------ actions
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions[:] = self.actions
        self.actions[:] = actions.clamp(-1.0, 1.0)
        a = self.actions
        if self.cfg.domain_randomization:
            a = (a + self.cfg.action_noise_std * torch.randn_like(a)).clamp(-1.0, 1.0)
        if self.cfg.showcase:
            a = torch.where((self.status == 1)[:, None], torch.zeros_like(a), a)  # finished robots stop
        self._action_hist = torch.roll(self._action_hist, 1, dims=1)
        self._action_hist[:, 0] = a
        delayed = self._action_hist[torch.arange(self.num_envs, device=self.device), self._delay]
        self._wheel_targets[:] = delayed * self.cfg.base_wheel_speed * self.speed_scale[:, None]

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target_index(target=self._wheel_targets, joint_ids=self._wheel_ids)

    # ------------------------------------------------------------------ state, dones, rewards
    def _update_track_state(self) -> torch.Tensor:
        pos_w = self.robot.data.root_pos_w.torch
        o = self.scene.env_origins
        self.pos_xy[:] = pos_w[:, :2] - o[:, :2]
        self.yaw[:] = self.robot.data.heading_w.torch
        _, s, lat, tyaw = self.lib.project(self.track_id, self.pos_xy)
        self.progress[:] = s
        self.lateral[:] = lat
        self.heading_err[:] = wrap_to_pi(self.yaw - tyaw)
        return pos_w[:, 2] - o[:, 2]

    def _get_dones(self):
        self._just_reset[:] = False
        z = self._update_track_state()
        hw = self.lib.half_width[self.track_id]
        fell = (z < self.cfg.library.top_z - self.cfg.fall_margin)
        fell |= self.robot.data.projected_gravity_b.torch[:, 2] > -0.5
        fell |= self.lateral.abs() > (hw - self.cfg.edge_margin)
        finished = (self.progress >= self.lib.length[self.track_id] - self.cfg.finish_margin) & ~fell
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.showcase:
            running = self.status == 0
            newly_fell = running & fell
            newly_fin = running & finished
            self.best_progress[:] = torch.where(running, torch.maximum(self.best_progress, self.progress), self.best_progress)
            self.status[newly_fell] = 2
            self.status[newly_fin] = 1
            self.finish_time[newly_fin] = self.episode_length_buf[newly_fin].float() * self.step_dt
            self._fell[:] = newly_fell
            self._finished[:] = newly_fin
            return torch.zeros_like(fell), time_out
        self._fell[:] = fell
        self._finished[:] = finished
        return fell | finished, time_out

    def _get_rewards(self) -> torch.Tensor:
        dt = self.step_dt
        ds = (self.progress - self.prev_progress).clamp(-0.1, 0.1)
        self.prev_progress[:] = self.progress
        hw = self.lib.half_width[self.track_id]
        r_prog = self.cfg.rew_progress * ds
        r_lat = self.cfg.rew_lateral * (self.lateral / hw).square() * dt
        r_head = self.cfg.rew_heading * (1.0 - torch.cos(self.heading_err)) * dt
        r_act = self.cfg.rew_action_rate * (self.actions - self.prev_actions).square().sum(-1)
        t = (self.episode_length_buf.float() * dt).clamp(min=dt)
        avg_speed = (self.progress - self.start_progress) / t
        r_fin = self._finished.float() * (self.cfg.rew_finish + self.cfg.rew_fast * avg_speed / BASE_SPEED_MPS)
        r_fall = self._fell.float() * self.cfg.rew_fall
        if self.cfg.showcase:
            r_prog = r_prog * (self.status == 0).float()
        log = self.extras.setdefault("log", {})
        log["Episode_Reward/progress"] = r_prog.mean()
        log["Episode_Reward/lateral"] = r_lat.mean()
        log["Episode_Reward/finish"] = r_fin.mean()
        return r_prog + r_lat + r_head + r_act + r_fin + r_fall

    # ------------------------------------------------------------------ reset
    def _sample_speed(self, n: int) -> torch.Tensor:
        dev = self.device
        if self.cfg.fixed_speed_scale > 0:
            return torch.full((n,), float(self.cfg.fixed_speed_scale), device=dev)
        lv = self.cfg.speed_levels
        top = lv[self.speed_level]
        lo = lv[0]
        uni = lo + (top - lo) * torch.rand(n, device=dev)
        return torch.where(torch.rand(n, device=dev) < self.cfg.top_level_prob, torch.full_like(uni, top), uni)

    def _update_curriculum(self, env_ids: torch.Tensor):
        if self.cfg.fixed_speed_scale > 0 or self.common_step_counter == 0:
            return
        top = self.cfg.speed_levels[self.speed_level]
        at_top = (self.speed_scale[env_ids] - top).abs() < 1e-4
        ended = self._finished[env_ids] | self._fell[env_ids]
        sel = at_top & ended
        n = int(sel.sum())
        if n > 0:
            rate = float(self._finished[env_ids][sel].float().mean())
            w = min(1.0, n / 200.0)
            self._level_ema = (1 - w) * self._level_ema + w * rate
            self._level_episodes += n
        if (
            self.speed_level < len(self.cfg.speed_levels) - 1
            and self._level_episodes >= self.cfg.promote_min_episodes
            and self._level_ema >= self.cfg.promote_success
        ):
            self.speed_level += 1
            self._level_ema = 0.0
            self._level_episodes = 0
            print(f"[curriculum] speed cap unlocked: {self.cfg.speed_levels[self.speed_level]}x")

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        dev = self.device
        if self.common_step_counter > 0:
            log = self.extras.setdefault("log", {})
            ended = self._finished[env_ids] | self._fell[env_ids]
            log["Metrics/success_rate"] = self._finished[env_ids].float().mean()
            log["Metrics/fall_rate"] = self._fell[env_ids].float().mean()
            log["Metrics/progress_fraction"] = (self.progress[env_ids] / self.lib.length[self.track_id[env_ids]]).mean()
            # per-track success EMA
            tid = self.track_id[env_ids][ended]
            fin = self._finished[env_ids][ended].float()
            if len(tid):
                cnt = torch.bincount(tid, minlength=self.lib.num_tracks).float()
                hit = torch.bincount(tid, weights=fin, minlength=self.lib.num_tracks)
                upd = cnt > 0
                self._track_success[upd] = 0.95 * self._track_success[upd] + 0.05 * (hit[upd] / cnt[upd])
            for k in range(self.lib.num_tracks):
                log[f"Track/{self.library.shapes[k].name}"] = self._track_success[k]
            self._update_curriculum(env_ids)
            log["Curriculum/max_speed_scale"] = torch.tensor(float(self.cfg.speed_levels[self.speed_level]), device=dev)
            log["Curriculum/top_level_success_ema"] = torch.tensor(self._level_ema, device=dev)

        super()._reset_idx(env_ids)
        n = len(env_ids)

        if self.cfg.track_per_env:
            self.track_id[env_ids] = env_ids % self.lib.num_tracks
        elif self.cfg.fixed_track >= 0:
            self.track_id[env_ids] = self.cfg.fixed_track
        else:
            self.track_id[env_ids] = torch.randint(0, self.lib.num_tracks, (n,), device=dev)
        tid = self.track_id[env_ids]
        length = self.lib.length[tid]
        s0 = torch.full((n,), self.cfg.start_s, device=dev)
        rs = torch.rand(n, device=dev) < self.cfg.random_start_prob
        s0 = torch.where(rs, self.cfg.start_s + torch.rand(n, device=dev) * (self.cfg.random_start_max_frac * length - self.cfg.start_s), s0)
        xy, tyaw = self.lib.pose_at(tid, s0)
        lat0 = (torch.rand(n, device=dev) * 2 - 1) * self.cfg.reset_lateral_noise
        xy = xy + lat0[:, None] * torch.stack([-torch.sin(tyaw), torch.cos(tyaw)], -1)
        yaw0 = tyaw + (torch.rand(n, device=dev) * 2 - 1) * self.cfg.reset_yaw_noise

        pose = torch.zeros(n, 7, device=dev)
        pose[:, :2] = xy + self.scene.env_origins[env_ids, :2]
        pose[:, 2] = self.scene.env_origins[env_ids, 2] + self.cfg.library.top_z + self.cfg.spawn_height
        pose[:, 5], pose[:, 6] = torch.sin(yaw0 / 2), torch.cos(yaw0 / 2)
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=env_ids)
        jp = self.robot.data.default_joint_pos.torch[env_ids].clone()
        self.robot.write_joint_position_to_sim_index(position=jp, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(jp), env_ids=env_ids)

        self.pos_xy[env_ids], self.yaw[env_ids] = xy, yaw0
        self.progress[env_ids] = s0
        self.prev_progress[env_ids] = s0
        self.start_progress[env_ids] = s0
        self.lateral[env_ids] = lat0
        self.heading_err[env_ids] = wrap_to_pi(yaw0 - tyaw)
        self._fell[env_ids] = False
        self._finished[env_ids] = False
        self._just_reset[env_ids] = True
        self.status[env_ids] = 0
        self.finish_time[env_ids] = float("nan")
        self.best_progress[env_ids] = s0
        self.speed_scale[env_ids] = self._sample_speed(n)

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._action_hist[env_ids] = 0.0
        dr = self.cfg.domain_randomization
        self._delay[env_ids] = torch.randint(0, self.cfg.max_action_delay + 1, (n,), device=dev) if dr else 0
        self._cam_drop[env_ids] = torch.rand(n, device=dev) < self.cfg.camera_dropout_prob
        self._lidar_drop[env_ids] = (torch.rand(n, device=dev) < self.cfg.lidar_dropout_prob) & ~self._cam_drop[env_ids]
        if self._frames is not None:
            self._frames[env_ids] = 0.0
        if self.cfg.image_aug:
            u = lambda lo, hi, m: lo + (hi - lo) * torch.rand(m, device=dev)  # noqa: E731
            self._aug_gain[env_ids] = u(*self.cfg.aug_brightness, n).view(n, 1, 1, 1)
            self._aug_contrast[env_ids] = u(*self.cfg.aug_contrast, n).view(n, 1, 1, 1)
            self._aug_shift[env_ids] = u(-self.cfg.aug_color_shift, self.cfg.aug_color_shift, 3 * n).view(n, 1, 1, 3)

    # ------------------------------------------------------------------ analytic range sensors
    def _fan(self, pos, pitch_deg, yaws_deg):
        """Ray directions (body frame) of a planar fan pitched down by pitch_deg; returns (dirs[R,3], origin[3])."""
        a = torch.deg2rad(yaws_deg.float())
        th = torch.deg2rad(torch.tensor(float(pitch_deg)))
        d = torch.stack([torch.cos(a) * torch.cos(th), torch.sin(a), -torch.cos(a) * torch.sin(th)], -1)
        return d.to(self.device), torch.tensor(pos, dtype=torch.float32, device=self.device)

    def _cast(self, dirs_b, org_b, max_range):
        """Cast body-frame rays from every robot against the ramp tops (occupancy map) and the floor.

        Exact for ramp top surfaces and the floor; ignores the 3 cm ramp side faces and the thin pillars.
        Returns range [E,R], kind [E,R] (1 ramp, 0 floor, -1 none within range), hit points [E,R,3] env-local.
        """
        q, pos = self._sensor_pose()
        E, R = q.shape[0], dirs_b.shape[0]
        o = pos + quat_apply(q, org_b.expand(E, 3))  # [E,3] env-local
        d = quat_apply(q[:, None, :].expand(E, R, 4).reshape(-1, 4), dirs_b.expand(E, R, 3).reshape(-1, 3)).view(E, R, 3)
        oz = o[:, None, 2]
        dz = d[..., 2]
        down = dz < -1e-6
        top = self.cfg.library.top_z
        t_top = torch.where(down & (oz > top), (oz - top) / (-dz).clamp(min=1e-6), torch.full_like(dz, 1e9))
        p_top = o[:, None, :] + t_top[..., None] * d
        j = ((p_top[..., 0] - self._grid_origin[0]) / self._grid_res).floor().long()
        i = ((p_top[..., 1] - self._grid_origin[1]) / self._grid_res).floor().long()
        H, W = self._grid.shape
        inside = (i >= 0) & (i < H) & (j >= 0) & (j < W) & (t_top < 1e8)
        on_ramp = torch.zeros_like(inside)
        on_ramp[inside] = self._grid[i[inside], j[inside]]
        hit_ramp = on_ramp & (t_top <= max_range)
        t_floor = torch.where(down, oz / (-dz).clamp(min=1e-6), torch.full_like(dz, 1e9))
        hit_floor = ~hit_ramp & (t_floor <= max_range)
        t = torch.where(hit_ramp, t_top, torch.where(hit_floor, t_floor, torch.full_like(dz, max_range)))
        kind = torch.where(hit_ramp, 1, torch.where(hit_floor, 0, -1))
        return t, kind, o[:, None, :] + t[..., None] * d

    def _sensor_pose(self):
        """Robot root pose (env-local position, xyzw quaternion); respawned robots use their new spawn pose
        because the sim buffers only refresh on the next physics step."""
        q = self.robot.data.root_quat_w.torch.clone()
        pos = self.robot.data.root_pos_w.torch - self.scene.env_origins
        m = self._just_reset
        if m.any():
            pos = pos.clone()
            pos[m, :2] = self.pos_xy[m]
            pos[m, 2] = self.cfg.library.top_z + self.cfg.spawn_height
            y = self.yaw[m]
            q[m] = torch.stack([torch.zeros_like(y), torch.zeros_like(y), torch.sin(y / 2), torch.cos(y / 2)], -1)
        return q, pos

    def _update_range_sensors(self):
        if self._lidar and (self.common_step_counter % max(1, self.cfg.lidar_period_steps) == 0 or self._just_reset.any()):
            r, k, h = self._cast(self._lidar_dirs, self._lidar_org, self.cfg.lidar_max_range)
            if self.common_step_counter % max(1, self.cfg.lidar_period_steps) == 0:
                self.lidar_range[:], self.lidar_kind[:], self.lidar_hits[:] = r, k, h
            else:  # only refresh robots that just respawned
                m = self._just_reset
                self.lidar_range[m], self.lidar_kind[m], self.lidar_hits[m] = r[m], k[m], h[m]
        if self._cliff:
            self.cliff_range[:], self.cliff_kind[:], self.cliff_hits[:] = self._cast(self._cliff_dirs, self._cliff_org, self.cfg.cliff_max_range)

    # ------------------------------------------------------------------ observations
    def _ranges(self, rng: torch.Tensor, max_range: float, noise: float = 0.0, dropout: float = 0.0) -> torch.Tensor:
        d = rng
        if noise > 0:
            d = (d + noise * torch.randn_like(d)).clamp(0.0, max_range)
        x = d / max_range
        if dropout > 0:
            x = torch.where(torch.rand_like(x) < dropout, torch.zeros_like(x), x)
        return x

    def _privileged(self) -> torch.Tensor:
        hw = self.lib.half_width[self.track_id]
        la = self.lib.lookahead_body(self.track_id, self.progress, self.pos_xy, self.yaw, self.cfg.lookahead_distances)
        fresh = (~self._just_reset).float()[:, None]
        wv = self.robot.data.joint_vel.torch[:, self._wheel_ids] / self._max_wheel * fresh
        lin = self.robot.data.root_lin_vel_b.torch[:, :2] / 1.2 * fresh
        ang = self.robot.data.root_ang_vel_b.torch[:, 2:3] / 6.0 * fresh
        return torch.cat(
            [
                (self.lateral / hw)[:, None], torch.sin(self.heading_err)[:, None], torch.cos(self.heading_err)[:, None],
                la.reshape(self.num_envs, -1), wv, lin, ang, self.actions,
                (self.speed_scale / max(self.cfg.speed_levels))[:, None], (hw / 0.16)[:, None],
            ],
            dim=-1,
        )

    def _camera_obs(self) -> torch.Tensor:
        img = self._camera.data.output["rgb"].torch[..., :3].float() / 255.0
        k = self.cfg.camera_downsample
        if k > 1:
            img = F.avg_pool2d(img.permute(0, 3, 1, 2), k).permute(0, 2, 3, 1)
        if self.cfg.image_aug:
            m = img.mean(dim=(1, 2, 3), keepdim=True)
            img = ((img - m) * self._aug_contrast + m) * self._aug_gain + self._aug_shift
            img = (img + self.cfg.aug_noise_std * torch.randn_like(img)).clamp(0.0, 1.0)
        img = (img - img.mean(dim=(1, 2), keepdim=True)).permute(0, 3, 1, 2)
        img = img * (~self._cam_drop).float()[:, None, None, None]
        if self.cfg.frame_stack <= 1:
            return img
        C = img.shape[1]
        if self._frames is None:
            self._frames = img.repeat(1, self.cfg.frame_stack, 1, 1)
        else:
            self._frames = torch.roll(self._frames, C, dims=1)
            self._frames[:, :C] = img
        return self._frames.clone()

    def _get_observations(self) -> dict:
        state = self._privileged()
        obs = {"policy": state, "critic": state}
        fresh = (~self._just_reset).float()[:, None]
        wv = self.robot.data.joint_vel.torch[:, self._wheel_ids] / self._max_wheel * fresh
        obs["proprio"] = torch.cat([self.actions, wv, (self.speed_scale / max(self.cfg.speed_levels))[:, None]], -1)
        self._update_range_sensors()
        if self._lidar:
            x = self._ranges(self.lidar_range, self.cfg.lidar_max_range, self.cfg.lidar_noise_std,
                             self.cfg.lidar_ray_dropout if self.cfg.domain_randomization else 0.0)
            obs["lidar"] = x * (~self._lidar_drop).float()[:, None]
        if self._cliff:
            obs["cliff"] = self._ranges(self.cliff_range, self.cfg.cliff_max_range, self.cfg.lidar_noise_std)
        if self._camera is not None:
            obs["camera"] = self._camera_obs()
        return obs
