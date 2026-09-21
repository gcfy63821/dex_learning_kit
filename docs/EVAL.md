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
| `--success_dist` | `0.05` | Env-internal success threshold (`min_final_dist < this`). Post-hoc strict thresholds use `end_final_dist`, not this. |
| `--save_traj` | off | Save per-frame tracking arrays in `records.json`. |
| `--perturb_obj_xy` | `0.0` | Eval-time random XY perturbation of object init pos (m). |
| `--inject_jitter` / `--inject_dropout` / `--inject_hand_noise` | `None` | Override PC noise at eval time (sim2real preview); by default eval zeros all PC noise for a clean run. |

### strict3 protocol

`eval.py` computes strict2 / strict3 / strict5 itself and writes them to
`summary.json` under `strict`, plus a per-demo `strict3`. It used to only record
the raw fields and leave the filtering to you — which meant the script printed
the env-internal rate this section tells you not to report, and never printed the
one it does.

The headline metric, **strict3**, counts an episode as a success only if:

- `end_final_dist < 3 cm` at episode end (object reached the demo's final pose),
  **and**
- no object-position drift (a `fail/obj_pos_drift` cause did not fire), and
- the episode was not a bad-init (`survival_len ≤ 5` are excluded).

This is stricter than the env-internal reach-end success, and the gap is not a
constant. On one checkpoint the two read 93.0% and 80.4% overall — but per demo:

```
cube_small_1   env 96.0%  |  strict3 86.0%
cube_small_2   env 92.0%  |  strict3 89.8%
squeegee_1     env 86.0%  |  strict3 84.0%
squeegee_2     env 98.0%  |  strict3 62.0%     <-- 36 points
```

`squeegee_2` survives its trajectory almost every time and still fails to put the
object down within 3 cm. Report strict3, and report it per demo.

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
`success_rate` looks tiny. The correct **episode** success rate is
`succeeded / terminated`, accumulated per episode termination (using the
`extras["succeeded"]` value captured in `_get_rewards` before reset). `eval.py`
does this accounting for you; the caveat matters if you post-process the raw
step-level stats yourself.

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
