# DAgger PointCloud distillation

(plus an optional PPO fine-tune that is **not** part of the default pipeline — see §Step B)

Distills the privileged poseobs PPO teacher into a **point-cloud student** that
observes raw point clouds instead of ground-truth object pose, then optionally
fine-tunes it with asymmetric PPO.

```
poseobs PPO teacher (obs=557d)
    └─ DAgger → PointCloudStudent (proprio + PointNet → action)
         └─ PPO fine-tune via ActorCriticPointCloud (asymmetric: critic uses priv_info)
              └─ deploy: PointNetEncoder in the actor (PC tensors only)
```

The env (`franka-sharpa-pointcloud`) inherits the poseobs env, so its obs is still
**557d**. It additionally exposes raw point clouds in the obs dict — `scene_pc`,
`scene_mask`, `hand_pc`, `tactile_pc`, `tactile_force`. The **PointNet encoder
lives in the policy network, not the env**. Three fusion strategies are registered
as task variants: `franka-sharpa-pointcloud{,-sceneonly,-separate}`.

## Step A — DAgger

```bash
python scripts/train_dagger_pc.py --task franka-sharpa-pointcloud \
    --teacher_ckpt checkpoints/teacher_poseobs.pth --side right \
    --data_idx '["rt/0416_grasp/cube_small_2"]' \
    --num_envs 128 --dagger_iters 30 --rollout_steps 4096 \
    --beta_init 1.0 --beta_decay 0.85 \
    --train_epochs 8 --batch_size 512 --lr 5e-5 --max_buffer 200000 \
    --out_dir logs/dagger_pc --headless
```

- **Output:** `dagger_final.pth` under `--out_dir` (default `logs/dagger_pc`).

Key DAgger flags:

| Flag | Default | Purpose |
|---|---|---|
| `--teacher_ckpt` | required | Poseobs teacher checkpoint. |
| `--dagger_iters` | `20` | DAgger outer iterations. |
| `--rollout_steps` | `4096` | Steps collected per iteration. |
| `--beta_init` / `--beta_decay` | `1.0` / `0.8` | Teacher-mixing schedule. |
| `--train_epochs` / `--batch_size` / `--lr` | `5` / `512` / `5e-5` | BC update. |
| `--max_buffer` | `200000` | Replay buffer size. |
| `--hand_body_subset` | `None` | `minimal5` / `minimal6` / `default11` / `dense22` (see below). |
| `--student_ckpt` | `None` | Resume from a pretrained student. |

## Step B — PPO fine-tune (optional, and measured to hurt)

```bash
python scripts/train_ppo_pc.py --task franka-sharpa-pointcloud \
    --dagger_ckpt logs/dagger_pc/.../dagger_final.pth \
    --side right --data_idx '["rt/0416_grasp/cube_small_2"]' \
    --num_envs 64 --max_iters 100 \
    --mini_epochs 1 --minibatch_size 1024 \
    --lr 1e-5 --clip 0.05 \
    --init_logstd -4.0 \
    --save_every_n 20 --out_dir logs/ppo_pc --headless
```

- **Output:** `ppo_final.pth` under `--out_dir` (default `logs/ppo_pc`).

## Critical invariants

These are load-bearing — earlier configurations gave ~1% success or NaN'd:

- **Env inherits poseobs (557d), not the force env (583d).** The student's proprio
  obs dim must match the teacher's actor obs dim.
- **Workspace bbox crop** — `cfg.pc_workspace_min/max` (env-local frame) drops
  table + curtain hits before subsampling, so `scene_pc` concentrates on the
  manipulation region.
- **Hand-body subset = `minimal6` is the best found** — wrist + 5 fingertips only.
  The MCP knuckles barely contact objects, so including them adds PointNet noise.
  The cfg default is `default11` (wrist + 5 MCP + 5 fingertips); pass
  `--hand_body_subset minimal6` to reproduce the best result.
- **DAgger buffer is fp32** (not fp16): `scene_pc` coords > 1 m exceed fp16
  mantissa resolution.
- **PPO `--init_logstd -4`** (σ ≈ 0.018) is critical. The default `logstd = 0`
  (σ = 1) over 29 action dims produces a sampled-action norm ≈ 5.4 that fully
  overwhelms the DAgger `mu` signal, dropping success to ~1% before any learning.
  With `-4` the sampled action stays near `mu` and PPO preserves the warm start.
- **PPO NaN guards** (all needed together): `ratio.clamp(max=10)`,
  `logstd.clamp(-5, 2)`, Huber critic loss (`F.smooth_l1_loss`) instead of MSE, no
  bf16 autocast in the update, and conservative HP (`critic_coef=0.5`, `lr=1e-5`,
  `clip=0.05`). Without these, PPO NaN'd at iter 20–80.
- **`init_curriculum` must be OFF** for the pointcloud env (already the default in
  `franka_sharpa_pointcloud_env_cfg.py`). With it on, resets sample only from the
  last half of each demo, biasing both training and eval toward the easy
  demo-end subset. See [EVAL.md](EVAL.md).

### Hand-subset ↔ ckpt alignment

A DAgger ckpt saves `n_hand` (e.g. 6) but not the body names. `eval.py` and
`train_ppo_pc.py` pre-read the ckpt's `n_hand/n_scene/n_tactile` and override the
env_cfg + auto-pick the matching subset **before** building the env. Without this,
hand-subset ckpts crash with `Sizes of tensors must match` at PointNet forward
(env produces 11 hand points, encoder weights sized for 6).

## Empirical result

On the 20-demo / 9000-episode strict3 eval (deterministic `mu`, DR off,
init_curriculum off), **DAgger with `--hand_body_subset minimal6` reached 46.9%
strict3 / 71.4% strict5** — the best configuration found. PPO fine-tune from a
DAgger init did **not** exceed the DAgger student in this setup (best PPO config
regressed ~5.8pp), so the DAgger checkpoint is normally the deliverable. Tactile
force contributes ~+2.5pp; don't ablate it unless real-robot noise forces you to.
