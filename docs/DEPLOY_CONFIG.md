# Deploy / Sim2Real constants — single source of truth

All the sim2real geometry and comm constants live in **`src/dexx/deploy_config.py`**.
Edit that one file; every consumer imports from it. No more hunting for the same
number copy-pasted across env cfgs, the deploy env, the depth subscribers and the
retarget scripts.

## What's centralized

| Constant | Meaning | Consumers (auto-pick-up) |
|---|---|---|
| `TABLE_SURFACE_Z = 0.415` | table top, env-local (object is placed relative to this) | `franka_sharpa_env.py` collision check, retarget |
| `ARM_BASE_Z = 0.415` / `ARM_BASE_POS` | fr3_link0 height. Was 0.432 while the base sat 1.7 cm above the table; the robot was remounted level on 2026-09-01. | `franka_sharpa_env_cfg.py`, `..._pointcloud_deploy_env.py`, `retarget_arm_2stage_aug.py`, `scripts/retarget.py` |
| `WRIST_POS_OFFSET = (-0.1,0,0.415)` | wrist retarget offset (intentionally table height) | `franka_sharpa_env_cfg.py` |
| `SIM_INTRINSICS {fx,fy,cx,cy}` | 320×240 camera intrinsics | `..._pointcloud_env_cfg.py`, `..._pointcloud_deploy_env.py`, `ros2_depth_subscriber.py` |
| `DEPTH_H, DEPTH_W = 240, 320` | depth resolution | same as above |
| `PC_WORKSPACE_MIN/MAX` | PC crop bbox, env-local | `..._pointcloud_env_cfg.py` (+ deploy env fallback) |
| `POLYMETIS_STATE_PORT/CMD_PORT = 5560/5561` | NUC bridge ZMQ ports | `deploy_pc.py`, polymetis deploy env, arm client |
| `CAMERA_ZMQ_ADDR_EXAMPLE` | example camera depth pub addr | `deploy_pc.py --depth_zmq_addr` help |

## Typical edits

- **Different table / arm mount height** → change `TABLE_SURFACE_Z` and/or `ARM_BASE_Z`.
  ⚠️ These change runtime geometry. Re-retarget
  + re-train if you change them meaningfully.
- **Different camera** → change `SIM_INTRINSICS` (+ recalibrate the extrinsic).
- **Different NUC / camera host ports** → change the port constants (or pass CLI flags).

## Not centralized (by design)

- **`deploy/polymetis_joint_bridge.py` port literals.** That file runs on the NUC
  in the `polymetis-local` environment, where `dexx` is not installed, so it
  cannot import this module. Its `5560` / `5561` must be kept in sync by hand;
  the file says so at the point of use.

- **Camera extrinsic** (camera-in-armbase 4×4) is calibrated per mount, not a
  global constant. `visual_raycaster.py:_build_camera_extrinsics()` holds only a
  stale 2026-05-20 fallback; pass `--camera_extrinsic <file.npy>` to **training,
  evaluation and deploy** alike. Shipped files live in `calib/camera_align/`.
  See `tutorial/06_camera_calibration/`.
- `franka_sharpa_env.py:157` computes `_table_surface_z = 0.4 + table_half_height`
  (derived from the table asset half-height) — left as a formula; its result equals
  `TABLE_SURFACE_Z`.
