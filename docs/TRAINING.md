# PPO teacher training

`scripts/train_teacher.py` trains the **privileged PPO teacher** on the
`franka-sharpa-force-poseobs` task. The teacher is the source policy for DAgger
distillation ([DISTILLATION.md](DISTILLATION.md)).

## What "teacher" means here

- Task: **`franka-sharpa-force-poseobs`**, observation = **557d**. The observation
  is the critic-horizon force obs (550d) plus a **7d noisy object-pose tail**
  (position + quaternion). The tail acts as a privileged-but-noisy input that the
  actor is free to rely on during training.
- Because the pose is noisy, the DAgger student can later drop it and replace it
  with point clouds without losing much (see the distillation doc).

## Command

```bash
python scripts/train_teacher.py --task franka-sharpa-force-poseobs --side right \
    --data_idx '["rt/0416_grasp/cube_small_2"]' --num_envs 2048 --headless
```

- **Input:** retargeted pkl(s) named by `--data_idx`.
- **Output:** teacher checkpoint (`.pth`). A pretrained one ships at
  `checkpoints/teacher_poseobs.pth`.

## Key flags

| Flag | Default | Purpose |
|---|---|---|
| `--task` | `None` | Use `franka-sharpa-force-poseobs` for the teacher. |
| `--side` | `None` | Hand side (`left`/`right`). |
| `--data_idx` | `None` | JSON/Python list of demo indices, e.g. `'["rt/0416_grasp/cube_small_2"]'`. |
| `--num_envs` | `16384` | Parallel environments. Lower it (e.g. 2048) to fit memory. |
| `--seed` | `42` | Environment / training seed. |
| `--max_agent_steps` | `None` | Training iteration budget. |
| `--load_path` | `None` | Checkpoint to load. |
| `--resume` | off | Resume training from `--load_path`. |
| `--env_cfg` | `[]` | Override env_cfg fields, e.g. `--env_cfg force_reward_weight=0.0`. |
| `--no_contact_force` | off | Ablation: disable the 5d scalar contact force in the obs. |
| `--video`, `--video_length`, `--video_interval` | | Record training videos. |
| `--wandb-project-name`, `--wandb-entity`, `--wandb-name` | | wandb logging. |

`--data_idx` accepts JSON (double quotes) or a Python-literal list.

## Demo-ordering gotcha (14-key ordering)

The env's data builder (`env._build_data`) expects the per-demo metadata keys to
be consistent across the demos in `--data_idx`. A demo that **lacks the
`obj_scale` field** (older retargeted pkls) must appear **first** in the
`--data_idx` list. If such a demo is not first, `env._build_data` raises a
`KeyError` while assembling the batched data tensors, because the first demo seeds
the key schema for the rest.

Rule of thumb: put any legacy / minimal-metadata demo at index 0 of `--data_idx`,
or re-retarget it so it carries `obj_scale` (retarget writes `obj_scale` into new
pkls). See [RETARGET.md](RETARGET.md) for the `--obj_scale` flag.
