import gymnasium as gym

from . import agents

##
# Register Gym environments (clean subset of franka-sharpa tasks).
##

# ---- Base env ----
gym.register(
    id="franka-sharpa",
    entry_point=f"dexx.tasks.franka_sharpa.franka_sharpa_env:FrankaSharpaEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sharpa_env_cfg:FrankaSharpaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
    },
)

# ---- Force env (main training task) ----
gym.register(
    id="franka-sharpa-force",
    entry_point=f"dexx.tasks.franka_sharpa.franka_sharpa_force_env:FrankaSharpaForceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sharpa_env_cfg:FrankaSharpaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
    },
)

# ---- Critic-horizon + observed object pose obs (FoundationPose-emulating). ----
gym.register(
    id="franka-sharpa-force-poseobs",
    entry_point=f"dexx.tasks.franka_sharpa.franka_sharpa_force_poseobs_env:FrankaSharpaForcePoseObsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sharpa_force_poseobs_cfg:FrankaSharpaPoseObsCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_critic_horizon_ppo_cfg.yaml",
    },
)

# ---- V3 deploy: pytorch_kinematics FK in lieu of sim.step. Aims for full 30Hz. ----
gym.register(
    id="franka-sharpa-force-critic-horizon-deploy-pkfk",
    entry_point=f"dexx.tasks.franka_sharpa.franka_sharpa_force_critic_horizon_deploy_env_v3:FrankaSharpaForceCriticHorizonDeployEnvV3",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sharpa_critic_horizon_cfg:FrankaSharpaCriticHorizonCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_critic_horizon_ppo_cfg.yaml",
    },
)

# ---- PointCloud env variants (POINTCLOUD_ENV_PLAN) ---------------------------
# Same env class for all three; the env exposes raw point clouds in the obs
# dict. The student / actor-critic chooses fusion strategy via cfg.
gym.register(
    id="franka-sharpa-pointcloud",
    entry_point="dexx.tasks.franka_sharpa.franka_sharpa_pointcloud_env:FrankaSharpaPointCloudEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sharpa_pointcloud_env_cfg:FrankaSharpaPointCloudEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
    },
)

# PointCloud deploy env: real-robot sibling, scene_pc from a depth source +
# hand_pc / tactile_pc from pk FK, tactile_force from Sharpa SDK F6.
# Same obs dict schema as sim — any PC ckpt (DAgger / PPO) loads as-is.
# NOTE: this base env class is kept because the Polymetis deploy env below inherits
# it, but the ROS2-only deploy task is NOT registered — this build is Polymetis-only.
# (arm driven by Polymetis (NUC) over ZMQ; depth over ZMQ from the camera host.)
gym.register(
    id="franka-sharpa-pointcloud-polymetis-deploy",
    entry_point="dexx.tasks.franka_sharpa.franka_sharpa_pointcloud_polymetis_deploy_env:FrankaSharpaPointCloudPolymetisDeployEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sharpa_pointcloud_env_cfg:FrankaSharpaPointCloudEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
    },
)

# scene-only ablation: same env, different fusion strategy at policy build time.
gym.register(
    id="franka-sharpa-pointcloud-sceneonly",
    entry_point="dexx.tasks.franka_sharpa.franka_sharpa_pointcloud_env:FrankaSharpaPointCloudEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sharpa_pointcloud_env_cfg:FrankaSharpaPointCloudEnvCfg_SceneOnly",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
    },
)

# separate-encode ablation.
gym.register(
    id="franka-sharpa-pointcloud-separate",
    entry_point="dexx.tasks.franka_sharpa.franka_sharpa_pointcloud_env:FrankaSharpaPointCloudEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sharpa_pointcloud_env_cfg:FrankaSharpaPointCloudEnvCfg_Separate",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
    },
)

# Offline recording variant: same env + a third-person RGB+depth Camera
# sensor. Requires `--enable_cameras` at AppLauncher and is GPU-heavy, so
# only meant for `--num_envs ≤ 16` viz runs, not training.
gym.register(
    id="franka-sharpa-pointcloud-record",
    entry_point="dexx.tasks.franka_sharpa.franka_sharpa_pointcloud_record_env:FrankaSharpaPointCloudRecordEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sharpa_pointcloud_record_env_cfg:FrankaSharpaPointCloudRecordEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
    },
)
