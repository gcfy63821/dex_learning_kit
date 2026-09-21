# Debug & calibration tools (reference)

The release ships the Polymetis deploy runtime (`deploy/`) **and** the camera
calibration chain (`tools/calib/`). A few bring-up and profiling tools are not
bundled; they are listed at the end so you know they exist.

## Camera extrinsic calibration — **BUNDLED in `tools/calib/`**

The full chain ships with this release. **[tutorial/06](../tutorial/06_camera_calibration/)**
is the step-by-step procedure; this table is the reference.

| Tool | Purpose |
|---|---|
| `deploy/move_to_frame_polymetis.py` | drive the real arm to a demonstration frame so sim and real match |
| `tools/calib/capture_multiframe_zmq.py` | accumulate N ZMQ depth+colour frames into one dense camera-frame cloud |
| `tools/calib/gen_sim_frame_ply.py` | URDF FK at that frame → sim robot + table point cloud |
| `tools/calib/icp_align_extrinsic.py` | Kabsch/SVD ICP, real onto sim → refined 4×4 |
| `tools/calib/live_calibrate_extrinsic.py` | viser browser session: overlay + sliders, save to `.npy` |
| `tools/calib/viz_cropped_overlay.py` | cropped real vs cropped sim — what the policy actually sees |
| `tutorial/06_camera_calibration/inspect_extrinsic.py` | validate a 4×4 and report drift between two of them |

All of them read `dexx.deploy_config` for table height, arm-base height, crop box
and intrinsics, so they cannot silently disagree with the trainer.

The camera-host depth publisher (`realsense_depth_zmq_pub.py`) is the one piece
that lives on the camera host rather than in either repository.

## Dynamics / gain system-ID — **BUNDLED in `tools/sysid/`**

Replays a motion in sim and on the real arm and diffs them. **[tutorial/07](../tutorial/07_dynamics_alignment/)**
has the procedure and how to read the numbers.

| Tool | Purpose |
|---|---|
| `tools/sysid/motions/`, `motions_hand/` | pre-generated CSV+JSON motions (`chirp_sweep`, `step_per_joint`, `sin_j*`, `backlash_detection`, …) |
| `tools/sysid/generate_motions.py` / `generate_hand_motions.py` | regenerate or add motions, with an FR3 safety check |
| `tools/sysid/replay_motion_sim.py` | sim replay at the training gains; `--arm_kp/--arm_kd` to test a candidate |
| `tools/sysid/replay_motion_polymetis.py` | real replay via Polymetis; `--dry_run` first |
| `tools/sysid/replay_hand_motion_sim.py` / `_real.py` | the hand equivalents (Sharpa SDK, no ROS2) |
| `tools/sysid/step_response_sim.py`, `analyze_step_response.py` | rise time / overshoot per joint |
| `tools/sysid/analyze_motion.py` | sim-vs-real per-joint metrics + overlay plots → `metrics.csv`, two PNGs |

The two ROS2 recorders (`replay_motion_real.py`, `step_response_real.py`) are
deliberately not bundled: the live arm path here is Polymetis, and shipping both
invites recording through the wrong one.

## Polymetis arm backend — **BUNDLED in `deploy/`**

The polymetis deploy path is fully included in this release:

| Tool | Where it runs | Purpose |
|---|---|---|
| `deploy/polymetis_joint_bridge.py` | **NUC** (polymetis-local env) | bridges local Polymetis server ↔ ZMQ (state :5560 / cmd :5561). Required for the polymetis deploy path. |
| `deploy/move_to_frame_polymetis.py` | inference PC | move real arm to a pkl frame. |
| `deploy/test_polymetis_arm.py` | inference PC | sanity-check the bridge connection / arm state. |

Runtime env classes + client are under `src/dexx/`: `franka_sharpa_pointcloud_polymetis_deploy_env.py`,
`tasks/hand_imitation/deploy/polymetis_arm_client.py`, `scripts/deploy/realsense_depth_zmq_subscriber.py`.
See [DEPLOY.md](DEPLOY.md) for the full startup sequence.

## Arm / PC debugging

| Tool | Purpose |
|---|---|
| `tune_arm_gains.py` | sweep/set real arm joint-impedance Kq/Kqd. |
| `analyze_arm_jitter.py` | quantify arm tracking jitter. |
| `measure_pc_resample_jitter.py` | quantify PC subsample jitter frame-to-frame. |
| `apply_sharpa_load.py` / `calibrate_payload.py` | set / calibrate the Sharpa hand FCI payload. |
| `dump_real_pc_zmq.py` / `convert_real_depth_to_pc.py` | dump the real scene_pc as PLY for offline inspection. |
| `viz_sim_vs_real_depth.py` / `viz_sim_real_align_rgb.py` | side-by-side sim vs real depth / RGB alignment. |

## Notes

- These tools depend on the same env / cfg code the release now has, plus (for the
  ones that touch hardware) the `sharpa` hand SDK and Polymetis — only available on
  the deploy machines.
- The camera-host depth publisher (`realsense_depth_zmq_pub.py`) lives on the camera
  host, not in either repo's tree.
