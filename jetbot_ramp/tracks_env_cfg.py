"""Configs for the generalization task: procedural track library, LiDAR + cliff sensors, camera, speed curriculum.

Observation groups (all built by ``TracksEnv``):

* ``policy`` / ``critic``: privileged state (track geometry ahead, offsets, speeds). Teacher actor + every critic.
* ``lidar``: tilted 2D LiDAR, 61 normalized ranges (0 = invalid/dropped, 1 = no return within 3 m).
* ``cliff``: 3 forward-down range sensors at the front.
* ``proprio``: last action, wheel speeds, current speed cap.
* ``camera``: 2 stacked 80x60 RGB frames (only when ``use_camera``).

Model variants (agents/tracks_agents_cfg.py):
A = LiDAR-only PPO, B = camera+LiDAR PPO, C = teacher -> camera+LiDAR distillation -> PPO fine-tune.
"""

from __future__ import annotations

import math

from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass, replace
from isaaclab.visualizers import VisualizerCfg

from .jetbot_cfg import JETBOT_CFG, JETBOT_WHEEL_JOINTS
from .jetbot_ramp_env_cfg import JetbotRampEventCfg
from .tracks import TEST_LIBRARY, TRAIN_LIBRARY, LibrarySpec, build_library

BASE_WHEEL_SPEED = 12.0  # [rad/s] = 0.36 m/s, the original cap ("1x")

_FRICTION = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="multiply", restitution_combine_mode="multiply",
    static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
)


def _pitch_down(deg: float) -> tuple[float, float, float, float]:
    h = math.radians(deg) / 2
    return (0.0, math.sin(h), 0.0, math.cos(h))


def _spacing(spec: LibrarySpec) -> float:
    return max(build_library(spec).footprint()) + 2.0


# Jetbot with the speed limit unlocked to 3x (36 rad/s ~ 1.08 m/s) and stronger motors
JETBOT_FAST_CFG = replace(
    JETBOT_CFG,
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            stiffness=0.0,
            damping=0.8,
            joint_effort_limit=1.5,
            joint_velocity_limit=3 * BASE_WHEEL_SPEED + 4.0,
            velocity_limit_sim=3 * BASE_WHEEL_SPEED + 4.0,
        ),
    },
)

@configclass
class TracksSceneCfg(InteractiveSceneCfg):
    ground = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1, physics_material=_FRICTION, debug_vis=False
    )
    ramp = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Ramp", spawn=sim_utils.UsdFileCfg(usd_path=""))
    robot: ArticulationCfg = replace(JETBOT_FAST_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.9, 0.9, 0.9)))


@configclass
class TracksCameraSceneCfg(TracksSceneCfg):
    front_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis/front_cam",
        update_period=0.0,
        width=160,
        height=120,
        data_types=["rgb"],
        renderer_cfg=IsaacRtxRendererCfg(),
        spawn=sim_utils.PinholeCameraCfg(focal_length=0.75, horizontal_aperture=2.35, clipping_range=(0.01, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.069, 0.0, 0.059), rot=_pitch_down(20.0), convention="world"),
    )


@configclass
class TracksEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 40.0
    action_space = 2
    lookahead_distances: list[float] = [0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4]
    observation_space = 26  # privileged state; verified in the env
    state_space = 26

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=PhysxCfg(), physics_material=_FRICTION)
    scene: TracksSceneCfg = TracksSceneCfg(num_envs=2048, env_spacing=_spacing(TRAIN_LIBRARY), replicate_physics=True)
    library: LibrarySpec = TRAIN_LIBRARY

    wheel_joint_names: list[str] = JETBOT_WHEEL_JOINTS
    base_wheel_speed = BASE_WHEEL_SPEED
    spawn_height = 0.035

    # speed curriculum: the cap is unlocked level by level when the top level is mastered
    speed_levels: list[float] = [1.0, 2.0, 3.0]
    start_level = 0
    promote_success = 0.85
    promote_min_episodes = 3000
    top_level_prob = 0.6  # else a uniform speed in [1x, top]
    fixed_speed_scale = 0.0  # > 0 disables the curriculum (eval / video)

    # resets
    fixed_track = -1
    track_per_env = False  # env i always drives track i % num_tracks (mosaic videos)
    start_s = 0.15
    random_start_prob = 0.3
    random_start_max_frac = 0.8
    reset_lateral_noise = 0.03
    reset_yaw_noise = 0.2

    # termination
    fall_margin = 0.05
    edge_margin = 0.06
    finish_margin = 0.15

    # rewards
    rew_progress = 10.0
    rew_lateral = -1.0  # * (lateral / half_width)^2 * dt
    rew_heading = -1.0
    rew_action_rate = -0.01
    rew_fall = -10.0
    rew_finish = 10.0
    rew_fast = 5.0  # extra finish bonus * (average speed / 0.36 m/s)

    # sensors (analytic ray casting against the ramp-top occupancy map and the floor; see TracksEnv._cast)
    lidar_pos = (0.05, 0.0, 0.09)  # in the robot root frame [m]
    lidar_pitch_deg = 35.0
    lidar_fov_deg = (-60.0, 60.0)
    lidar_res_deg = 2.0
    lidar_max_range = 3.0
    lidar_period_steps = 3  # 10 Hz at 30 Hz policy
    cliff_pos = (0.07, 0.0, 0.03)
    cliff_pitch_deg = 50.0
    cliff_yaws_deg = (-25.0, 0.0, 25.0)
    cliff_max_range = 1.0
    use_lidar = True
    use_cliff = True
    use_camera = False
    lidar_noise_std = 0.01  # [m]
    lidar_ray_dropout = 0.02
    camera_dropout_prob = 0.0  # per episode, fusion models
    lidar_dropout_prob = 0.0
    frame_stack = 2
    camera_downsample = 2
    image_aug = True
    aug_brightness = (0.6, 1.4)
    aug_contrast = (0.7, 1.3)
    aug_color_shift = 0.08
    aug_noise_std = 0.02

    # randomization
    domain_randomization = True
    events: JetbotRampEventCfg = JetbotRampEventCfg()
    max_action_delay = 2
    action_noise_std = 0.02

    # showcase (swarm videos / evaluation): no resets except timeout; finished robots stop, fallen ones stay down
    showcase = False

    def __post_init__(self):
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(6.0, -6.0, 6.0), lookat=(0.0, 0.0, 0.3))
        if not self.domain_randomization:
            self.events = None


@configclass
class TracksCameraEnvCfg(TracksEnvCfg):
    use_camera = True
    promote_min_episodes = 800  # ~10x fewer robots than the state/LiDAR runs
    camera_dropout_prob = 0.1
    lidar_dropout_prob = 0.1
    num_rerenders_on_reset = 1
    scene: TracksCameraSceneCfg = TracksCameraSceneCfg(
        num_envs=192, env_spacing=_spacing(TRAIN_LIBRARY), replicate_physics=True
    )


def _as_eval(cfg, camera: bool):
    """Held-out tracks, fixed top speed unless overridden, no curriculum tricks."""
    cfg.library = TEST_LIBRARY
    cfg.scene.env_spacing = _spacing(TEST_LIBRARY)
    cfg.scene.num_envs = 64
    cfg.random_start_prob = 0.0
    cfg.camera_dropout_prob = 0.0
    cfg.lidar_dropout_prob = 0.0
    cfg.fixed_speed_scale = 3.0
    cfg.episode_length_s = 60.0
    return cfg


@configclass
class TracksEnvCfg_EVAL(TracksEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _as_eval(self, camera=False)


@configclass
class TracksCameraEnvCfg_EVAL(TracksCameraEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _as_eval(self, camera=True)
