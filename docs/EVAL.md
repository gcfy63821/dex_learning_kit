# Evaluation

Two entry points:

- `scripts/eval.py` — batched quantitative evaluation with strict success
  post-processing. Auto-detects DAgger `PointCloudStudent` vs PPO
  `ActorCriticPointCloud` from the checkpoint.
- `scripts/play.py` — interactive / visual rollout of a checkpoint.
- `scripts/record_videos.py` — render expert-rollout videos / snapshots.

## eval.py

```bash
python scripts/eval.py --task franka-sharpa-pointcloud \
    --load_path <dagger_final.pth or ppo_final.pth> --side right \
    --data_idx '["rt/0416_grasp/cube_small_2"]' \
    --num_envs 128 --out_dir logs/eval_cube --headless
```

| Flag | Default | Purpose |
|---|---|---|
| `--load_path` | required | DAgger or PPO ckpt (arch auto-detected). |
| `--out_dir` | required | Output dir for `records.json`. |
| `--data_idx` | `None` | Demo indices to evaluate. |
| `--num_envs` | `128` | Parallel envs. |
| `--max_episodes` | `5000` | Stop after this many episode terminations. |
| `--max_steps` | `12000` | Hard step ceiling (safety). |
| `--success_dist` | `0.05` | Closest-approach success threshold (`min_final_dist < this`). Post-hoc strict thresholds use `end_final_dist`, not this. |
| `--save_traj` | off | Save per-frame tracking arrays in `records.json`. |
| `--perturb_obj_xy` | `0.0` | Eval-time random XY perturbation of object init pos (m). |
| `--inject_jitter` / `--inject_dropout` / `--inject_hand_noise` | `None` | Override PC noise at eval time (sim2real preview); by default eval zeros all PC noise for a clean run. |

### strict3 protocol

`eval.py` computes strict2 / strict3 / strict5 itself and writes them to
`summary.json` under `strict`, plus a per-demo `strict3`. The separate
`success_rate_*` fields and `EpisodeRecord.succeeded` measure closest approach:
the object came within `--success_dist` of its target at some point during the
episode. They do not use the env's trajectory-completion success flag.
Endpoint distance and rotation come from the terminal step's reward diagnostics,
captured before the environment resets.

The headline metric, **strict3**, counts an episode as a success only if:

- `end_final_dist < 3 cm` at episode end (object reached the demo's final pose),
  **and**
- no object-position drift (stored as `obj_pos_drift` in `fail_causes`), and
- the episode was not a bad-init (`survival_len ≤ 5` are excluded).

This protocol does not require the env's success flag and deliberately does not
reject other failure causes. It is therefore not a subset of env-internal
reach-end success. Training's `Strict` is a separate conservative proxy: it
requires trajectory completion without failures and keeps bad inits in its
denominator as zeros (see [TRAINING.md](TRAINING.md)).

The gap between endpoint accuracy and closest-approach success is not constant.
For example, the previous evaluation breakdown was:

```
cube_small_1   closest 96.0%  |  strict3 86.0%
cube_small_2   closest 92.0%  |  strict3 89.8%
squeegee_1     closest 86.0%  |  strict3 84.0%
squeegee_2     closest 98.0%  |  strict3 62.0%     <-- 36 points
```

Reaching within 5 cm at some point does not guarantee finishing within 3 cm
without drift. Report strict3, and report it per demo. These historical numbers
illustrate the distinction; rerun evaluation after metric fixes for release results.

### Recommended eval settings (reproducibility)

Deterministic `mu` inference, DR off, **`init_curriculum` off**,
`random_state_init=True` (uniform over the full demo). The eval scripts set these
by default. Keeping `init_curriculum` on would sample resets only from the last
half of the demo, inflating success — see the gotcha in
[DISTILLATION.md](DISTILLATION.md).

### succeeded-vs-terminated caveat

Do not read a per-step success rate as the episode success rate. `env.success_buf`
is **cleared in `_reset_idx`** (which runs inside `step()`), so reading it right
after `step()` returns 0 for envs that just terminated — the per-step
`success_rate` looks tiny. To measure env trajectory completion, select terminated
environments from `extras["succeeded_per_env"]`, captured before reset, and average
those episode flags (as PPO does). `eval.py` instead computes the closest-approach
and strict endpoint metrics from episode records; its `succeeded` field is not
the env flag.

## play.py

Interactive rollout (visual, fewer envs):

```bash
python scripts/play.py --task franka-sharpa-pointcloud \
    --load_path <ckpt.pth> --side right \
    --data_idx '["rt/0416_grasp/cube_small_2"]' --num_envs 16
```

Key flags: `--num_envs` (default 16), `--max_episodes` (default 200),
`--max_steps`, `--label`. (Drop `--headless` to watch the GUI.)

⚠️ **`play.py`'s success rate is a different metric from `eval.py`'s.** Here a
success means the episode reached the end of the trajectory without a failure
termination; in `eval.py` it means the object finished within `--success_dist`
(5 cm) of its final target. A short `--max_steps` truncates episodes and drives
`play.py`'s number down without the policy being any worse — on one checkpoint
they read 6.2% and 95.5% respectively. Quote `eval.py`.

## record_videos.py

Renders one video per demo of an expert (teacher) rollout via the
`franka-sharpa-pointcloud-record` env (needs `--enable_cameras`; GPU-heavy, so use
few envs):

```bash
python scripts/record_videos.py --task franka-sharpa-pointcloud-record \
    --teacher_ckpt checkpoints/teacher_poseobs.pth --side right \
    --data_idx_list '["rt/0416_grasp/cube_small_2"]' \
    --out_dir logs/expert_rollout_videos
```

Notable flags: `--data_idx_list` (JSON list, one video per entry),
`--max_steps` (default 400), `--fps` (default 15), `--cam_yaw_deg`,
`--cam_z_offset`, `--snapshot_frames` (save PNG snapshots), `--snapshot_only`,
`--draw_force` (overlay per-fingertip contact-force arrows), `--save_per_env`.

---

## 评估 teacher（poseobs）用 `eval_teacher.py`

`eval.py` / `play.py` 吃的是 **PointCloud student** 的 ckpt（需要 `ckpt["cfg"]`）。
评估 **poseobs teacher** 请用 `scripts/eval_teacher.py`：

```bash
python scripts/eval_teacher.py --task franka-sharpa-force-poseobs \
    --load_path checkpoints/teacher_poseobs.pth \
    --side right --data_idx '["rt/0416_grasp/cube_small_2"]' \
    --modes random --success_dist 0.05 \
    --num_envs 64 --episodes_per_mode 256 --out_dir logs/eval_teacher --headless
```

输出 `eval_table.md` / `summary.json` / `records.json`。`records.json` 里每条含
`init_frame`、`survival_len`、`min_final_dist`、`fail_causes`，排查失败模式时很有用。

**换资产后请务必跑一次这个**作为回归基准 —— 资产层面的等价性（增益、限位、惯量…）
可以逐项相同而行为差几十个百分点，见 `assets/ASSETS.md` 的自碰撞过滤一节。
参考值：shipped teacher + `cube_small_2` @5cm ≈ **93–95%**。

## 参考轨迹工具（内部用）

- `scripts/collect_reference_rollouts.py` — 采集参考 rollout
- `scripts/build_rollout_references.py` — 由 rollout 构建参考轨迹

这两个是内部工具，不在主 pipeline 上，仅在需要重建参考轨迹时使用。
