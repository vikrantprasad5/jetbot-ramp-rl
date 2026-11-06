"""RSL-RL configs for the generalization experiments.

Teacher: PPO on privileged state.
A: LiDAR-only PPO from scratch (lidar + cliff + proprio), privileged critic.
B: camera + LiDAR fusion PPO from scratch, privileged critic.
C: teacher -> fusion student (distillation) -> PPO fine-tune (same network as B).
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

TEACHER_ACTOR = dict(hidden_dims=[256, 128], activation="elu", obs_normalization=True)
CRITIC = RslRlMLPModelCfg(hidden_dims=[256, 128], activation="elu", obs_normalization=True)
SENSORS_1D = ["lidar", "cliff", "proprio"]
FUSION_GROUPS = ["camera", *SENSORS_1D]


def _ppo(lr=1.0e-3, entropy=0.005, epochs=5, minibatches=4, schedule="adaptive"):
    return RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=entropy,
        num_learning_epochs=epochs, num_mini_batches=minibatches, learning_rate=lr, schedule=schedule,
        gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
    )


def _fusion_actor(init_std: float) -> RslRlCNNModelCfg:
    return RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(output_channels=[16, 32, 32], kernel_size=[5, 3, 3], stride=[2, 2, 2], activation="relu"),
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=init_std),
    )


@configclass
class TracksTeacherPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 20
    experiment_name = "tracks_teacher"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0
    actor = RslRlMLPModelCfg(**TEACHER_ACTOR, distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.6))
    critic = CRITIC
    algorithm = _ppo()


@configclass
class TracksLidarPPOCfg(RslRlOnPolicyRunnerCfg):
    """A: LiDAR-only, PPO from scratch."""

    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 20
    experiment_name = "tracks_A_lidar"
    obs_groups = {"actor": SENSORS_1D, "critic": ["critic"]}
    clip_actions = 1.0
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.6),
    )
    critic = CRITIC
    algorithm = _ppo()


@configclass
class TracksFusionPPOCfg(RslRlOnPolicyRunnerCfg):
    """B: camera + LiDAR, PPO from scratch."""

    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 5  # dense checkpoints for the generations video
    experiment_name = "tracks_B_fusion_ppo"
    obs_groups = {"actor": FUSION_GROUPS, "critic": ["critic"]}
    clip_actions = 1.0
    actor = _fusion_actor(0.6)
    critic = CRITIC
    algorithm = _ppo(lr=3.0e-4, minibatches=8)


@configclass
class TracksFusionDistillCfg(RslRlDistillationRunnerCfg):
    """C step 1: distil the teacher into the fusion network (--checkpoint <teacher model>)."""

    num_steps_per_env = 32
    max_iterations = 400
    save_interval = 20
    experiment_name = "tracks_C_distill"
    obs_groups = {"student": FUSION_GROUPS, "teacher": ["policy"]}
    clip_actions = 1.0
    student = _fusion_actor(0.1)
    teacher = RslRlMLPModelCfg(**TEACHER_ACTOR, distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.0))
    algorithm = RslRlDistillationAlgorithmCfg(num_learning_epochs=2, learning_rate=5.0e-4, gradient_length=16, max_grad_norm=1.0)


@configclass
class TracksFusionFinetuneCfg(TracksFusionPPOCfg):
    """C step 2: PPO fine-tune of the distilled student (--checkpoint from tools/distill_to_ppo.py)."""

    max_iterations = 500
    save_interval = 5
    experiment_name = "tracks_C_finetune"
    actor = _fusion_actor(0.2)
    algorithm = _ppo(lr=1.0e-4, entropy=0.0, minibatches=8)
