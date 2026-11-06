"""Jetbot on an elevated S-ramp: teacher (state) PPO and camera student distillation.

Registration only uses string entry points, so importing this package does not pull in Isaac Lab.
"""

import gymnasium as gym

_AGENTS = f"{__name__}.agents"

gym.register(
    id="Isaac-JetbotRamp-Teacher-Direct",
    entry_point=f"{__name__}.jetbot_ramp_env:JetbotRampEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jetbot_ramp_env_cfg:JetbotRampTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_ppo_cfg:JetbotRampTeacherPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Isaac-JetbotRamp-Teacher-Direct-Play",
    entry_point=f"{__name__}.jetbot_ramp_env:JetbotRampEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jetbot_ramp_env_cfg:JetbotRampTeacherEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_ppo_cfg:JetbotRampTeacherPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Isaac-JetbotRamp-Distill-Direct",
    entry_point=f"{__name__}.jetbot_ramp_env:JetbotRampEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jetbot_ramp_env_cfg:JetbotRampCameraEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_distillation_cfg:JetbotRampDistillationRunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Isaac-JetbotRamp-Distill-Direct-Play",
    entry_point=f"{__name__}.jetbot_ramp_env:JetbotRampEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jetbot_ramp_env_cfg:JetbotRampCameraEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_distillation_cfg:JetbotRampDistillationRunnerCfg",
        "default_agent": "rsl_rl",
    },
)


# ---------------------------------------------------------------------------
# Generalization experiments: procedural tracks, LiDAR + camera, speed curriculum
# ---------------------------------------------------------------------------
_T = f"{__name__}.tracks_env:TracksEnv"
_TC = f"{__name__}.tracks_env_cfg"
_TA = f"{_AGENTS}.tracks_agents_cfg"
_TRACK_AGENTS = {
    "teacher_cfg_entry_point": f"{_TA}:TracksTeacherPPOCfg",
    "lidar_cfg_entry_point": f"{_TA}:TracksLidarPPOCfg",
    "fusion_cfg_entry_point": f"{_TA}:TracksFusionPPOCfg",
    "distill_cfg_entry_point": f"{_TA}:TracksFusionDistillCfg",
    "finetune_cfg_entry_point": f"{_TA}:TracksFusionFinetuneCfg",
}
for _id, _cfg, _agent in [
    ("Isaac-JetbotTracks-Teacher", "TracksEnvCfg", "TracksTeacherPPOCfg"),
    ("Isaac-JetbotTracks-Lidar", "TracksEnvCfg", "TracksLidarPPOCfg"),
    ("Isaac-JetbotTracks-Fusion", "TracksCameraEnvCfg", "TracksFusionPPOCfg"),
    ("Isaac-JetbotTracks-FusionDistill", "TracksCameraEnvCfg", "TracksFusionDistillCfg"),
    ("Isaac-JetbotTracks-FusionFinetune", "TracksCameraEnvCfg", "TracksFusionFinetuneCfg"),
    ("Isaac-JetbotTracks-Eval", "TracksEnvCfg_EVAL", "TracksTeacherPPOCfg"),
    ("Isaac-JetbotTracks-Camera-Eval", "TracksCameraEnvCfg_EVAL", "TracksFusionPPOCfg"),
    ("Isaac-JetbotTracks-Train-Show", "TracksEnvCfg", "TracksTeacherPPOCfg"),
    ("Isaac-JetbotTracks-Stress", "TracksEnvCfg_STRESS", "TracksTeacherPPOCfg"),
    ("Isaac-JetbotTracks-Camera-Stress", "TracksCameraEnvCfg_STRESS", "TracksFusionPPOCfg"),
    ("Isaac-JetbotTracks-Camera-Train-Show", "TracksCameraEnvCfg", "TracksFusionPPOCfg"),
]:
    gym.register(
        id=_id,
        entry_point=_T,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_TC}:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{_TA}:{_agent}",
            "default_agent": "rsl_rl",
            **_TRACK_AGENTS,
        },
    )
