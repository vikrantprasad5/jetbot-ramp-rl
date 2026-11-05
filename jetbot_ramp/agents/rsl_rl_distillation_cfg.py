"""Teacher-to-student distillation (DAgger-style: the student drives, the teacher labels).

Train with ``--checkpoint <teacher model_XXXX.pt>``; RSL-RL loads the PPO ``actor_state_dict``
into the teacher network below, so it must match the teacher's actor exactly.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)

from .rsl_rl_ppo_cfg import TEACHER_ACTIVATION, TEACHER_HIDDEN_DIMS, TEACHER_OBS_NORMALIZATION


@configclass
class JetbotRampDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 600
    save_interval = 50
    experiment_name = "jetbot_ramp_student"
    obs_groups = {"student": ["camera", "proprio"], "teacher": ["policy"]}
    clip_actions = 1.0
    student = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 32],
            kernel_size=[5, 3, 3],
            stride=[2, 2, 2],
            activation="relu",
        ),
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.1),
    )
    teacher = RslRlMLPModelCfg(
        hidden_dims=TEACHER_HIDDEN_DIMS,
        activation=TEACHER_ACTIVATION,
        obs_normalization=TEACHER_OBS_NORMALIZATION,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.0),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=5.0e-4,
        gradient_length=16,
        max_grad_norm=1.0,
        loss_type="mse",
    )
