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
