# # Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# from isaaclab.utils import configclass

# from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


# @configclass
# class SharpaWavePPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 32
#     max_iterations = 1000000
#     save_interval = 100
#     experiment_name = "sharpa_wave_inhand_rotate"
#     empirical_normalization = True
#     policy = RslRlPpoActorCriticCfg(
#         init_noise_std=1.0,
#         actor_hidden_dims=[512, 256, 128],
#         critic_hidden_dims=[512, 256, 128],
#         activation="elu",
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.005,
#         num_learning_epochs=5,
#         num_mini_batches=4,
#         learning_rate=5.0e-4,
#         schedule="adaptive",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.016,
#         max_grad_norm=1.0,
#     )
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class DexHandImitationPPORunnerCfg(RslRlOnPolicyRunnerCfg):

    num_steps_per_env = 32                 # horizon_length
    max_iterations = 100000                # max_epochs
    save_interval = 100                   # save_frequency
    experiment_name = "DexHand"
    empirical_normalization = True         # normalize_input


    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,                # fixed_sigma + sigma_init=0
        actor_hidden_dims=[256, 512, 128, 64],
        critic_hidden_dims=[256, 512, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        # --- loss ---
        value_loss_coef=4.0,               # critic_coef
        use_clipped_value_loss=True,       # clip_value
        clip_param=0.2,                    # e_clip
        entropy_coef=0.0,

        # --- optimization ---
        learning_rate=5e-4,
        schedule="adaptive",               # lr_schedule
        num_learning_epochs=5,             # mini_epochs
        num_mini_batches=128,              # 4096*32 / 1024
        max_grad_norm=1.0,                 # grad_norm

        # --- RL ---
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,                  # kl_threshold
    )
