import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="imitation-rh",
    entry_point=f"dexx.tasks.hand_imitation.imitation_rh_env:ImitatorRHEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.imitation_rh_env_cfg:ImitatorRHEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:imitation_ppo_cfg.yaml",
    },
)

gym.register(
    id="imitation-lh",
    entry_point=f"dexx.tasks.hand_imitation.imitation_lh_env:ImitatorLHEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.imitation_lh_env_cfg:ImitatorLHEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:imitation_ppo_cfg.yaml",
    },
)


gym.register(
    id="dexmanip-rh",
    entry_point=f"dexx.tasks.hand_imitation.dexmanip_rh_env:DexManipRHEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexmanip_rh_env_cfg:DexManipRHEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
    },
)

gym.register(
    id="dexmanip-lh",
    entry_point=f"dexx.tasks.hand_imitation.dexmanip_lh_env:DexManipLHEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexmanip_lh_env_cfg:DexManipLHEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
    },
)
# franka sharpa rh env
gym.register(
    id="franka-sharpa-rh",
    entry_point=f"dexx.tasks.hand_imitation.franka_sharpa_rh_env:FrankaSharpaRHEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sharpa_rh_env_cfg:FrankaSharpaRHEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
    },
)

gym.register(
    id="franka-sharpa-lh",
    entry_point=f"dexx.tasks.hand_imitation.franka_sharpa_lh_env:FrankaSharpaLHEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sharpa_lh_env_cfg:FrankaSharpaLHEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
    },
)

gym.register(
    id="arm-lh",
    entry_point=f"dexx.tasks.hand_imitation.arm_lh_env:ArmLHEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.arm_lh_env_cfg:ArmLHEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
    },
)

# gym.register(
#     id="franka-sharpa",
#     entry_point=f"dexx.tasks.hand_imitation.franka_sharpa_env:FrankaSharpaEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.franka_sharpa_env_cfg:FrankaSharpaEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
#         "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
#     },
# )

# gym.register(
#     id="franka-sharpa-force",
#     entry_point=f"dexx.tasks.hand_imitation.franka_sharpa_force_env:FrankaSharpaForceEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.franka_sharpa_env_cfg:FrankaSharpaEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
#         "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
#     },
# )
# ---- deploy ----
gym.register(
    id="franka-sharpa-force-deploy",
    entry_point=f"dexx.tasks.hand_imitation.franka_sharpa_force_deploy_env:FrankaSharpaForceDeployEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sharpa_env_cfg:FrankaSharpaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
    },
)

# ----residual ----
gym.register(
    id="res-rh",
    entry_point=f"dexx.tasks.hand_imitation.resdexhand_rh_env:DexManipRHEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.resdexhand_rh_env_cfg:DexManipRHEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:residual_ppo_cfg.yaml",
    },
)
gym.register(
    id="res-lh",
    entry_point=f"dexx.tasks.hand_imitation.resdexhand_lh_env:DexManipLHEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.resdexhand_lh_env_cfg:DexManipLHEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:residual_ppo_lh_cfg.yaml",
    },
)

# ----recording----
gym.register(
    id="record-lh",
    entry_point=f"dexx.tasks.hand_imitation.rec_lh_env:RecDexManipLHEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexmanip_lh_env_cfg:DexManipLHEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
    },
)

gym.register(
    id="record-rh",
    entry_point=f"dexx.tasks.hand_imitation.rec_rh_env:RecDexManipRHEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexmanip_rh_env_cfg:DexManipRHEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
    },
)

gym.register(
    id="record-arm-lh",
    entry_point=f"dexx.tasks.hand_imitation.rec_arm_lh_env:RecFrankaSharpaLHEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sharpa_lh_env_cfg:FrankaSharpaLHEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
    },
)

gym.register(
    id="rec-franka-sharpa",
    entry_point=f"dexx.tasks.hand_imitation.rec_franka_sharpa_env:RecFrankaSharpaEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sharpa_env_cfg:FrankaSharpaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
    },
)

# gym.register(
#     id="franka-sharpa-visual",
#     entry_point=f"dexx.tasks.hand_imitation.franka_sharpa_visual_env:FrankaSharpaVisualEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.franka_sharpa_env_cfg:FrankaSharpaEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
#         "gym_style_cfg_entry_point": f"{agents.__name__}:franka_sharpa_visual_ppo_cfg.yaml",
#     },
# )


# ----debug-----
gym.register(
    id="track-rh",
    entry_point=f"dexx.tasks.hand_imitation.simple_tracking_env:SimpleTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simple_tracking_env_cfg:SimpleTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:imitation_ppo_cfg.yaml",
    },
)

# ----data preprocessing----
gym.register(
    id="franka-sharpa-data-prep",
    entry_point=f"dexx.tasks.hand_imitation.franka_sharpa_data_prep_env:FrankaSharpaDataPrepEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sharpa_env_cfg:FrankaSharpaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
    },
)

# ----test tracking----
gym.register(
    id="test-franka-sharpa-tracking",
    entry_point=f"dexx.tasks.hand_imitation.test_franka_sharpa_tracking_env:TestFrankaSharpaTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.test_franka_sharpa_tracking_env_cfg:TestFrankaSharpaTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:test_tracking_ppo_cfg.yaml",
    },
)

gym.register(
    id="test-franka-sharpa-tracking-lh",
    entry_point=f"dexx.tasks.hand_imitation.test_franka_sharpa_tracking_lh_env:TestFrankaSharpaTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.test_franka_sharpa_tracking_lh_env_cfg:TestFrankaSharpaTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:test_tracking_ppo_cfg.yaml",
    },
)

# ----dual-arm tracking test----
gym.register(
    id="test-franka-sharpa-dual-tracking",
    entry_point=f"dexx.tasks.hand_imitation.test_franka_sharpa_dual_tracking_env:TestFrankaSharpaDualTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.test_franka_sharpa_dual_tracking_env_cfg:TestFrankaSharpaDualTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:test_tracking_ppo_cfg.yaml",
    },
)

# ----dual-arm training----
gym.register(
    id="franka-sharpa-dual",
    entry_point=f"dexx.tasks.hand_imitation.franka_sharpa_dual_env:FrankaSharpaDualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sharpa_dual_env_cfg:FrankaSharpaDualEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
    },
)