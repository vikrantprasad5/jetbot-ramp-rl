"""Jetbot articulation configuration (Isaac Lab 3.0).

The asset is NVIDIA's Jetbot from the Isaac Sim 6.2 asset library. Facts taken from the USD:

* root prim ``jetbot`` with bodies ``chassis``, ``left_wheel``, ``right_wheel``
* revolute joints ``left_wheel_joint`` / ``right_wheel_joint`` (axis X), velocity drives
* PhysX max joint velocity 719.6 deg/s, about 12.5 rad/s
* wheel radius 0.03 m, wheel base 0.1125 m (values used by Isaac Sim's DifferentialController)
* root origin at wheel-axle height; forward is +x (the stock camera sits at +x)
"""

from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

JETBOT_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Robots_Multiphysics/NVIDIA/Jetbot/jetbot.usda"

JETBOT_WHEEL_RADIUS = 0.03  # [m]
JETBOT_WHEEL_BASE = 0.1125  # [m]
JETBOT_MAX_WHEEL_SPEED = 12.0  # [rad/s], just under the asset's PhysX joint velocity limit (~12.56)
JETBOT_WHEEL_JOINTS = ["left_wheel_joint", "right_wheel_joint"]

JETBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=JETBOT_USD_PATH,
        rigid_props=[
            sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
            PhysxRigidBodyCfg(
                max_linear_velocity=10.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
        ],
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
        ],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={".*_wheel_joint": 0.0},
        joint_vel={".*_wheel_joint": 0.0},
    ),
    actuators={
        # Velocity control: stiffness 0, damping acts as velocity gain. Effort limit stops the
        # wheels from producing unrealistic acceleration (real Jetbot TT motors are weak).
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            stiffness=0.0,
            damping=0.5,
            joint_effort_limit=0.6,
            joint_velocity_limit=JETBOT_MAX_WHEEL_SPEED + 0.5,
        ),
    },
)
"""Jetbot with velocity-controlled wheels. Values are first guesses; tune them in the smoke test."""
