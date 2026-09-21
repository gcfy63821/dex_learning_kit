# Retargeting (MANO → Sharpa)

`scripts/retarget.py` maps a MANO human-hand demo onto the Sharpa HA4 hand and
the Franka FR3 arm using a 2-stage IsaacLab optimization (stage 1: arm-only reach;
stage 2: hand + arm tracking). Output is a `.pkl` trajectory the training envs
consume.

Joint targets in the output (`opt_dof_pos`) are stored in **cfg / Sharpa order**
— the canonical hand order (see [JOINT_ORDERING.md](JOINT_ORDERING.md)).

## Command

```bash
python scripts/retarget.py --side right --data_idx rt/0416_grasp/cube_small_2 --headless
```

- **Input:** source demo directory (MANO joints + `meta.json`) under
  `data/.../robotool_batch/<task>/<obj>/`.
- **Output:** `<obj>@<n>.pkl` under the retargeting output root
  (default `data/retargeting/robotool_batch/mano2<dexhand>/`, override with
  `--dump_root`). For the example demo this is
  `data/examples/retargeting/robotool_batch/mano2sharpa_rh/0416_grasp/cube_small_2@0.pkl`.

> **Portability caveat:** the source `meta.json` stores **absolute**
> `obj_mesh_paths`. On a new machine, edit these to point at your local mesh files
> before running, or the object mesh will fail to load.

## Key flags

| Flag | Default | Purpose |
|---|---|---|
| `--data_idx` | `None` | Single demo index, e.g. `rt/0416_grasp/cube_small_2`. |
| `--task` | `None` | Retarget all sequences of a task (`--data_idx rt/{task}/*`). |
| `--side` | `right` | Hand side (`left`/`right`); selects hand + USD. |
| `--dexhand` | `sharpa` | Dexhand type. |
| `--iter` | `4000` | Max optimization iterations. |
| `--stage1_iter` | `None` | Stage-1 (arm-only) iterations; if unset uses side ratio. |
| `--z_offset` | `0.0` | Extra Z offset applied to the trajectory (m). |
| `--target_offset_xy` | `None` | Override the side default XY offset, e.g. `0.3 0.1`. |
| `--obj_scale` | `1.0` | Uniform object mesh scale, saved into the pkl for training. |
| `--reachability_th` | `0.08` | Reachability gate threshold (m); see below. |
| `--no_real_hand_clamp` | off | Disable clamping `opt_dof_pos` to the Sharpa reachable range. |
| `--num_envs` | `1` | Parallel envs (for batch / augmented retargeting). |
| `--dump_root` | `None` | Output root for the retargeted pkls. |
| `--skip_existing` | off | Incremental: skip variants whose pkl already exists. |

### Augmentation (object-pose perturbation)

Generate extra variants of a demo with the object translated / rotated:

| Flag | Default | Purpose |
|---|---|---|
| `--aug_num` | `0` | Number of augmented variants per demo (0 = original only). |
| `--aug_radius` | `0.05` | XY translation radius (m), uniform in `[-r, r]`. |
| `--aug_yaw_deg` | `10.0` | Yaw range (deg), uniform in `[-y, y]`. |
| `--aug_seed` | `42` | RNG seed (same seed → same variants). |

## Reachability gate

After stage 1 + 2, the script measures the mean arm end-effector error. If it
exceeds `--reachability_th` (default **0.08 m**), the variant is marked
`reachable=False` and only a **partial pkl** is written.

Training dataset loaders check this flag and **skip** unreachable demos
(`robotool_batch_dataset_dexhand.py`: `if opt_params.get("reachable", True) is
False`). So a demo that fails the gate will silently not appear in training —
lower `--reachability_th` for stricter tracking, or raise it to admit borderline
demos (the default was raised from 0.05 to 0.08 to admit ~5.8 cm cases).

---

## 重要：retarget 会原地写回 pkl

默认输出就是 `data/retargeting/robotool_batch/mano2sharpa_rh/<task>/<exp>@<res>.pkl`，
即**自带示例 demo 的位置**。相关参数：

| 参数 | 作用 |
|---|---|
| `--dump_root <dir>` | 换个输出根目录，不动示例数据 |
| `--skip_existing` | 目标 pkl 已存在则跳过（增量补 aug 用） |
| `--force_overwrite` | 允许 unreachable 的部分结果覆盖已有的 reachable 结果 |

**默认已有护栏**：unreachable 的部分结果**不会**覆盖已有的 reachable pkl（会打印 `[KEEP]` 并跳过）。
没有这道护栏时，照 quickstart 用小 `--iter` 跑一次就会把 93 kB 的示例 demo 写成 0.5 kB 的 stub。

## 迭代次数与耗时

`--iter` 少会收敛不到位，被可达性门禁判为 unreachable：

| iter | mean arm EE err | 结果 |
|---|---|---|
| 300 | ~0.114 m | unreachable |
| 6000（正式） | ~0.013 m | reachable |

6000 迭代单条 demo 在 RTX 4090 上约需 **45 分钟以上**，请据此安排批量任务。
