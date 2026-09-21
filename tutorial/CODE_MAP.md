# Code map

Where things are, and which lesson explains each one. The pipeline is not large —
78 modules, 22.7k lines — but it is unevenly distributed: one file holds a third
of it. This page is the index into that file so you do not have to scroll.

## The shape of the repository

```
src/dexx/
  deploy_config.py                  76   every sim2real constant          lesson 01
  robot_constants.py                67   hand gains, armature, friction   lessons 02, 07
  tasks/franka_sharpa/                   the environments
  tasks/hand_imitation/                  demo loading and transforms      lesson 03
  tasks/sharpa_VBTS/                     tactile sensor config
  algo/{ppo,dagger,models}/              training algorithms          lessons 04, 05
scripts/                                 entry points — one per stage
deploy/                                  the real-robot runtime           lesson 09
tutorial/                                these lessons
```

## Environments: a five-deep inheritance chain

Each layer adds one thing. Read them in this order; each is small except the base.

| class | file | lines | adds |
|---|---|---|---|
| `FrankaSharpaEnv` | `franka_sharpa_env.py` | 4358 | scene, actions, reward, reset — everything |
| `FrankaSharpaForceEnv` | `franka_sharpa_force_env.py` | 691 | contact sensing |
| `FrankaSharpaForceCriticHorizonEnv` | `..._force_critic_horizon_env.py` | 474 | the K-frame privileged block, the named slot map |
| `FrankaSharpaForcePoseObsEnv` | `..._force_poseobs_env.py` | 229 | the 7-d noisy object pose, with its noise model |
| `FrankaSharpaPointCloudEnv` | `..._pointcloud_env.py` | 414 | scene / hand / tactile point clouds |

The configs mirror it exactly: `FrankaSharpaEnvCfg` → `CriticHorizonCfg` →
`PoseObsCfg` → `PointCloudEnvCfg`. A field set on a parent is visible to every
child, which is why `deploy_config.py` only has to be imported once.

## Navigating `franka_sharpa_env.py`

4358 lines, and the only file where you need line numbers. Grouped by what you
would be looking for. Line numbers drift when the file is edited — if one is off,
`grep -n "def <name>" src/dexx/tasks/franka_sharpa/franka_sharpa_env.py` is
authoritative and this table is not.

| you want | method | line | lesson |
|---|---|---|---|
| how an action becomes a joint target | `_pre_physics_step` | 822 | 07 |
| the PD control itself | `_apply_action` | 1214 | 07 |
| what the actor observes | `compute_observations` | 2370 | 04 |
| the reward | `compute_imitation_reward` | 3557 | 04 |
| episode termination | `_get_dones` | 1863 | — |
| reset, curriculum, object placement | `_reset_idx` | 1888 | 01 |
| domain randomization draws | `_rand_pd_scales`, `set_friction`, `set_com`, `set_mass` | 1881, 2590, 2634, 2640 | 07 |
| actuator setup (gains, armature) | `_setup_actuators` | 3310 | 02, 07 |
| loading the demonstrations | `_build_data` | 2724 | 03 |
| which joints are the arm's | `_identify_arm_joints` | 3104 | 02 |
| the scene: table, curtain, camera | `_setup_scene` | 510 | 06 |

Below line 3534 the file is free functions — quaternion and frame helpers,
`scale`/`unscale`, rotation utilities. Nothing stateful lives there.

`_pre_physics_step` (822) and `compute_observations` (2370) are the two worth
reading in full. Between them they define the policy's entire interface to the
world.

## Where a number you care about lives

| number | file |
|---|---|
| table height, arm base, workspace crop, camera intrinsics, ports | `deploy_config.py` |
| hand stiffness / damping / armature / friction | `robot_constants.py` |
| **arm** stiffness / damping | `franka_sharpa_critic_horizon_cfg.py` `__post_init__` |
| action delay, EMA coefficients, delta scale | `franka_sharpa_env_cfg.py` |
| domain randomization ranges | `franka_sharpa_env_cfg.py` |
| object pose noise model | `franka_sharpa_force_poseobs_cfg.py` |
| point counts, depth noise, crop | `franka_sharpa_pointcloud_env_cfg.py` |
| reward weights | `franka_sharpa_env.py:4027` onward |
| camera extrinsic | `calib/camera_align/*.npy` — **not** a constant, see lesson 06 |

## Algorithms

| file | lines | what |
|---|---|---|
| `algo/models/models.py` | 1060 | `ActorCriticAsymmetric` and friends. The critic takes `cat([obs, priv_info])` |
| `algo/ppo/ppo.py` | 662 | PPO for the state expert |
| `algo/dagger/dagger_pointcloud.py` | 436 | the DAgger loop, the convex blend, the lean-student slicing |
| `algo/dagger/pointcloud_student.py` | 89 | proprio ⊕ PointNet feature → action |
| `tasks/franka_sharpa/pointcloud/pointcloud_encoder.py` | 227 | the shared PointNet |

## Verifying a change

```bash
bash tutorial/run_acceptance.sh
```

Six minutes: static checks, a three-iteration distillation, an evaluation with
domain randomization on, then assertions on the artefacts — the student's width,
its dropped channels, its slot map, and that the evaluation episodes came out
balanced across demonstrations.

"The imports still resolve" is not the same as "it still works". This is the
second one.
