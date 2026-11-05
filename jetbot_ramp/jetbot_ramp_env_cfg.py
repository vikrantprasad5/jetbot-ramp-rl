"""Configs for the Jetbot S-ramp task.

* ``JetbotRampTeacherEnvCfg``: privileged state observations, no camera, many envs. Trained with PPO.
* ``JetbotRampCameraEnvCfg``: adds the front camera and student observation groups. Used for
  teacher-to-student distillation (and optional PPO fine-tuning of the student).

Observation groups returned by the env:

* ``policy``: teacher state (always present; the teacher reads this in both tasks)
* ``critic``: same as ``policy`` (PPO critic)
* ``camera``: (3 * frame_stack, H, W) image, camera task only
* ``proprio``: last action (+ optional wheel speeds), camera task only
"""

from __future__ import annotations

import math
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass, replace
from isaaclab.visualizers import VisualizerCfg

from .jetbot_cfg import JETBOT_CFG, JETBOT_MAX_WHEEL_SPEED, JETBOT_WHEEL_JOINTS
from .ramp import TrackSpec, footprint_size, write_ramp_usd

_TRACK = TrackSpec()
_FOOT = footprint_size(_TRACK)
ENV_SPACING = max(_FOOT) + 1.5
"""Distance between neighbouring envs: the larger side of both tracks plus a margin [m]."""

_RAMP_FRICTION = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
    static_friction=1.0,
    dynamic_friction=1.0,
    restitution=0.0,
)


def _cam_pitch_quat(pitch_down_deg: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) quaternion pitching a +x-forward camera down by ``pitch_down_deg``."""
    h = math.radians(pitch_down_deg) / 2
    return (0.0, math.sin(h), 0.0, math.cos(h))


##
# Scene
##


@configclass
class JetbotRampSceneCfg(InteractiveSceneCfg):
    """Ground, two ramps per env (S and mirrored S), Jetbot and light."""

    ground = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=_RAMP_FRICTION,
        debug_vis=False,
    )
    # usd_path is filled in by the env from cfg.track (see JetbotRampEnv.__init__)
    ramp = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Ramp",
        spawn=sim_utils.UsdFileCfg(usd_path=""),
    )
    robot: ArticulationCfg = replace(JETBOT_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.9, 0.9, 0.9)),
    )


@configclass
class JetbotRampCameraSceneCfg(JetbotRampSceneCfg):
    """Adds the forward camera on the chassis.

    Pose matches the stock Jetbot camera: about 7 cm forward and 6 cm up from the axle, pitched
    20 deg down. Horizontal FOV is about 115 deg (stock USD intrinsics). The real Jetbot camera
    (IMX219, 160 deg) is wider; match width/height/FOV to your real camera before sim-to-real.
    """

    front_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis/front_cam",
        update_period=0.0,
        width=160,  # rendered at 2x and average-pooled to 80x60 in the env (less RTX speckle/aliasing)
        height=120,
        data_types=["rgb"],
        renderer_cfg=IsaacRtxRendererCfg(),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=0.75, horizontal_aperture=2.35, clipping_range=(0.01, 20.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.069, 0.0, 0.059), rot=_cam_pitch_quat(20.0), convention="world"),
    )


##
# Domain randomization (sim-to-real)
##


@configclass
class JetbotRampEventCfg:
    """Physics randomization. Visual randomization is done on the image in the env (cheaper and
    works with replicate_physics=True)."""

    wheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 32,
            "make_consistent": True,
        },
    )
    chassis_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="chassis"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    motor_strength = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_wheel_joint"),
            "damping_distribution_params": (0.7, 1.3),
            "operation": "scale",
        },
    )


##
# Teacher (state-only) env
##


@configclass
class JetbotRampTeacherEnvCfg(DirectRLEnvCfg):
    # env timing: physics 120 Hz, policy 30 Hz (a realistic camera rate)
    decimation = 4
    episode_length_s = 30.0

    # spaces
    action_space = 2  # left/right wheel speed in [-1, 1]
    lookahead_distances: list[float] = [0.1, 0.2, 0.35, 0.5, 0.7]
    """Centreline points the teacher sees [m ahead]. 0.7 m is roughly what the camera can see."""
    observation_space = 1 + 2 + 2 * 5 + 2 + 3 + 2  # = 20, checked in the env
    state_space = 20

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics=PhysxCfg(),
        physics_material=_RAMP_FRICTION,
    )

    # scene
    scene: JetbotRampSceneCfg = JetbotRampSceneCfg(num_envs=1024, env_spacing=ENV_SPACING, replicate_physics=True)

    # track geometry (single source of truth for the USD and the reward maths)
    track: TrackSpec = TrackSpec()

    # robot
    wheel_joint_names: list[str] = JETBOT_WHEEL_JOINTS
    max_wheel_speed = JETBOT_MAX_WHEEL_SPEED  # [rad/s] at action = 1
    spawn_height = 0.035  # root above ramp top at reset [m] (wheel radius 0.03 + margin)

    # reset distribution
    randomize_track = True  # pick S or mirrored S per reset
    fixed_track = -1  # >= 0 forces that track (0 = S, 1 = mirrored S), e.g. for recordings
    start_s = 0.15  # [m] normal start position along the track
    random_start_prob = 0.3  # fraction of resets that start somewhere along the track (curriculum)
    random_start_max_frac = 0.8
    reset_lateral_noise = 0.03  # [m]
    reset_yaw_noise = 0.2  # [rad]

    # termination
    fall_margin = 0.05  # terminate when root z < top_z - fall_margin [m]
    edge_margin = 0.06  # terminate when |lateral| > width/2 - edge_margin (wheel contact ~0.06 m from centre) [m]
    finish_margin = 0.15  # finished when progress >= length - finish_margin [m]

    # rewards
    rew_progress = 10.0  # per metre of progress along the centreline
    rew_lateral = -20.0  # * lateral^2 * dt
    rew_heading = -1.0  # * (1 - cos(heading error)) * dt
    rew_action_rate = -0.01  # * |a_t - a_{t-1}|^2 per step
    rew_fall = -10.0
    rew_finish = 10.0

    # sim-to-real
    domain_randomization = True
    events: JetbotRampEventCfg = JetbotRampEventCfg()
    max_action_delay = 2  # policy steps of random actuation latency (0..max) per episode
    action_noise_std = 0.02

    # camera (unused by the teacher)
    use_camera = False

    def __post_init__(self):
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(3.0, 3.0, 2.5), lookat=(0.0, 0.0, 0.3))
        # generated ramp USD (cached by geometry hash); the env re-generates it if cfg.track is overridden
        self.scene.ramp.spawn.usd_path = write_ramp_usd(self.track)
        if not self.domain_randomization:
            self.events = None


@configclass
class JetbotRampTeacherEnvCfg_PLAY(JetbotRampTeacherEnvCfg):
    # the teacher ignores the camera; it is here so play videos can show the robot's view
    scene: JetbotRampCameraSceneCfg = JetbotRampCameraSceneCfg(num_envs=16, env_spacing=ENV_SPACING, replicate_physics=True)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.random_start_prob = 0.0


##
# Camera (student) env
##


@configclass
class JetbotRampCameraEnvCfg(JetbotRampTeacherEnvCfg):
    use_camera = True
    frame_stack = 2
    camera_downsample = 2  # average-pool factor from the rendered image to the policy input (160x120 -> 80x60)
    student_use_wheel_vel = False  # the stock Jetbot has no wheel encoders
    # image augmentation (per env, sampled at reset), applied before normalization
    image_aug = True
    aug_brightness = (0.6, 1.4)
    aug_contrast = (0.7, 1.3)
    aug_color_shift = 0.08  # per-channel additive shift, in [0, 1] image units
    aug_noise_std = 0.02
    write_image_to_file = False  # dump env 0's camera frames to ./jetbot_cam_*.png (debug)
    num_rerenders_on_reset = 1  # re-render after teleporting so the first image is not stale

    scene: JetbotRampCameraSceneCfg = JetbotRampCameraSceneCfg(
        num_envs=128, env_spacing=ENV_SPACING, replicate_physics=True
    )


@configclass
class JetbotRampCameraEnvCfg_PLAY(JetbotRampCameraEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.random_start_prob = 0.0
