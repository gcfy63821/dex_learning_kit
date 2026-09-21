# Pipeline

End-to-end walkthrough for Dex-X: retargeting a human demo onto the **Sharpa HA4**
hand (22 DOF) + **Franka FR3** arm (7 DOF), training a privileged PPO teacher,
distilling a deployable point-cloud student, fine-tuning with PPO, evaluating,
and (optionally) deploying to real hardware.

Simulation runs at `dt = 1/120` with `decimation = 4`, giving an effective
**~30 Hz** control rate that matches 30 fps demos and the real robot loop.

```
retarget           PPO teacher             DAgger (PointCloud)          PPO fine-tune          deploy (optional)
MANO → Sharpa   →  force-poseobs (557d) →  proprio + PointNet student → asymmetric PPO      →  ROS2 + Sharpa SDK
RETARGET.md        TRAINING.md             DISTILLATION.md              DISTILLATION.md        DEPLOY.md
```

All commands run from the `dexx_release/` root. Python is Isaac Lab's Python
(the interpreter of your Isaac Lab conda env).

## Install

Isaac Lab / Isaac Sim are installed **separately**, following their official pip
instructions (create a conda env, `pip install` Isaac Sim, then Isaac Lab). Once
that env is active, install this package into it:

```bash
conda activate <your_isaaclab_env>
cd dexx_release
pip install -e .
pip install -r requirements.txt
```

USD asset paths are resolved relative to the current working directory, so always
launch scripts from `dexx_release/`.

### Example data shipped with the release

- Source demo: `data/examples/robotool_batch/0416_grasp/cube_small_2/`
- Retargeted output: `data/examples/retargeting/robotool_batch/mano2sharpa_rh/0416_grasp/cube_small_2@0.pkl`
- Pretrained teacher: `checkpoints/teacher_poseobs.pth`

The demo is referenced everywhere by its data index `rt/0416_grasp/cube_small_2`.

> **Portability caveat:** the source demo `meta.json` stores **absolute**
> `obj_mesh_paths`. On a new machine you may need to fix these paths to point at
> the local mesh files before retargeting / training. See
> [RETARGET.md](RETARGET.md).

## Stages

### 1. Retarget (MANO → Sharpa)

Converts a MANO human-hand demo into a Sharpa joint trajectory + arm EE
trajectory, and gates it by arm reachability.

```bash
python scripts/retarget.py --side right --data_idx rt/0416_grasp/cube_small_2 --headless
```

- **Input:** source demo under `data/.../robotool_batch/<task>/<obj>/`.
- **Output:** `<obj>@0.pkl` with `opt_dof_pos` (Sharpa/cfg order), arm joints, and
  a `reachable` flag. Unreachable demos get a partial pkl that loaders skip.
- **Details:** [RETARGET.md](RETARGET.md).

### 2. PPO teacher (privileged)

Trains a privileged PPO policy on `franka-sharpa-force-poseobs` (obs = **557d**,
the last 7d being a noisy object pose). This is the source policy for distillation.

```bash
python scripts/train_teacher.py --task franka-sharpa-force-poseobs --side right \
    --data_idx '["rt/0416_grasp/cube_small_2"]' --num_envs 2048 --headless
```

- **Input:** retargeted pkl(s) selected via `--data_idx`.
- **Output:** teacher checkpoint (`.pth`). A pretrained one ships at
  `checkpoints/teacher_poseobs.pth`.
- **Details:** [TRAINING.md](TRAINING.md).

### 3. DAgger PointCloud student

Distills the poseobs teacher into a student that consumes raw point clouds
(`scene_pc`, `hand_pc`, `tactile_pc`, `tactile_force`) via a PointNet encoder in
the policy network. The env inherits poseobs, so obs still = 557d.

```bash
python scripts/train_dagger_pc.py --task franka-sharpa-pointcloud \
    --teacher_ckpt checkpoints/teacher_poseobs.pth --side right \
    --data_idx '["rt/0416_grasp/cube_small_2"]' --headless
```

- **Input:** teacher checkpoint + retargeted pkl(s).
- **Output:** `dagger_final.pth` under `--out_dir` (default `logs/dagger_pc`).
- **Details:** [DISTILLATION.md](DISTILLATION.md).

### 4. PPO fine-tune (optional)

Asymmetric PPO (critic uses privileged info) that warm-starts from the DAgger
student. `--init_logstd -4` is critical — see the doc.

```bash
python scripts/train_ppo_pc.py --task franka-sharpa-pointcloud \
    --dagger_ckpt <dagger_final.pth> --side right \
    --data_idx '["rt/0416_grasp/cube_small_2"]' \
    --init_logstd -4.0 --headless
```

- **Output:** `ppo_final.pth` under `--out_dir` (default `logs/ppo_pc`).
- **Note:** empirically PPO fine-tune does not exceed the DAgger student on this
  pipeline; the DAgger checkpoint is usually the deliverable. See
  [DISTILLATION.md](DISTILLATION.md).

### 5. Evaluate

`eval.py` auto-detects DAgger vs PPO architecture from the checkpoint and reports
success under the strict3 protocol.

```bash
python scripts/eval.py --task franka-sharpa-pointcloud \
    --load_path <dagger_final.pth> --side right \
    --data_idx '["rt/0416_grasp/cube_small_2"]' \
    --out_dir logs/eval_cube --headless
```

For an interactive/visual rollout use `scripts/play.py`; for expert-rollout
videos use `scripts/record_videos.py`. **Details:** [EVAL.md](EVAL.md).

### 6. Deploy (optional, hardware-specific)

Polymetis + ZMQ + Sharpa SDK deployment of a point-cloud student: arm via the NUC
Polymetis joint bridge (ZMQ), depth from the camera host (ZMQ), hand via the Sharpa
SDK. Requires the real robot + RealSense + a running Polymetis bridge. **Details:**
[DEPLOY.md](DEPLOY.md).

## Critical reference

Hand-joint ordering differs across the pipeline (cfg/Sharpa order vs USD/sorted
order). Getting this wrong maps commands to the wrong motor on the real hand. Read
[JOINT_ORDERING.md](JOINT_ORDERING.md) before any sim2real work.
