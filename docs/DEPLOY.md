# Real-robot deployment (Polymetis + ZMQ + Sharpa SDK)

> **OPTIONAL / hardware-specific.** Requires the physical Franka FR3 + Sharpa HA4
> hand, a RealSense depth camera on a camera host, a NUC running Polymetis, and the
> Sharpa SDK. This build is **Polymetis-only** — the arm runs through a Polymetis
> joint bridge (ZMQ) and depth streams from the camera host over ZMQ. (The legacy
> ROS2 deploy path has been removed.)

Deploys a point-cloud student (DAgger or PPO checkpoint) via the
`franka-sharpa-pointcloud-polymetis-deploy` env, which mirrors the sim obs-dict
schema so any PC checkpoint loads as-is. Scene points come from the ZMQ depth
stream; `hand_pc` / `tactile_pc` from `pytorch_kinematics` FK; `tactile_force` from
the Sharpa SDK F6 sensors; the arm's joint/EE state comes from the Polymetis bridge.

## Architecture

```
Inference PC (Python, ~30Hz)
  ├─ arm: PolymetisArmClient  ──ZMQ cmd :5561──►  polymetis_joint_bridge (NUC)
  │        joint targets                            └► Polymetis server ► FR3
  │        arm/EE state       ◄──ZMQ state :5560──┘  (joint impedance)
  ├─ depth: RealSenseDepthZmqSubscriber ◄──ZMQ :5562── camera host depth publisher
  └─ hand: Sharpa SDK set_joint_position()  →  HA4 hand   (cfg/Sharpa joint order!)
```

All ports / the example camera addr live in `src/dexx/deploy_config.py`
(`POLYMETIS_STATE_PORT`, `POLYMETIS_CMD_PORT`, `CAMERA_ZMQ_ADDR_EXAMPLE`).

## Launch order

**1. NUC** (in the `polymetis-local` env that has the `polymetis` package):
```bash
# a. Polymetis server (connects the real Franka; FCI activated + joints unlocked)
launch_robot.py robot_client=franka_hardware      # your usual launcher
# b. joint bridge: Polymetis <-> ZMQ (state :5560, cmd :5561)
python deploy/polymetis_joint_bridge.py           # --robot_ip localhost by default
```

**2. Camera host**: start the RealSense depth(+color) ZMQ publisher (PUB :5562).

**3. Inference PC**:
```bash
# (optional) move the arm to a demo frame first
python deploy/move_to_frame_polymetis.py --ip <NUC_IP> \
    --pkl data/retargeting/robotool_batch/mano2sharpa_rh/0416_grasp/cube_small_2@0.pkl \
    --frame 0 --hold

# deploy the PC student
python deploy/deploy_pc.py \
    --task franka-sharpa-pointcloud-polymetis-deploy \
    --load_path <pc_student.pth> --side right \
    --polymetis_ip <NUC_IP> \
    --depth_zmq_addr tcp://<CAM_HOST>:5562 \
    --camera_extrinsic <cam_in_armbase_4x4.npy> \
    --data_idx '["rt/0416_grasp/cube_small_2"]'
```

`deploy/test_polymetis_arm.py --ip <NUC_IP>` sanity-checks the bridge connection
before you deploy.

## deploy_pc.py flags

| Flag | Default | Purpose |
|---|---|---|
| `--load_path` | required | PC student ckpt (DAgger or PPO; arch auto-detected). |
| `--polymetis_ip` | **required** | NUC bridge IP (this build is Polymetis-only). |
| `--depth_zmq_addr` | **required** | Camera-host depth publisher, e.g. `tcp://<CAM_HOST>:5562`. |
| `--task` | `franka-sharpa-pointcloud-polymetis-deploy` | Deploy env task id. |
| `--side` | `right` | Hand side. |
| `--camera_extrinsic` | `None` | Path to a 4×4 `cam_in_armbase` `.npy` (per-mount calibration). |
| `--polymetis_state_port` / `--polymetis_cmd_port` | `5560` / `5561` | Bridge ZMQ ports. |
| `--polymetis_kq` / `--polymetis_kqd` | `None` | 7-value joint-impedance Kq/Kqd (must give both). |
| `--depth_height` / `--depth_width` | `240` / `320` | Depth resolution. |
| `--data_idx` | `None` | Demo whose wrist/joint target refs the env still needs. |
| `--pc_workspace_min` / `--pc_workspace_max` | `None` | Override workspace crop bbox. |
| `--pc_no_crop` | off | Disable the workspace crop (RViz debug only). |
| `--action_ramp_steps` | `0` | Ramp actions in over N steps at start. |
| `--record_rgb`, `--record_fps`, `--record_dir` | | Optionally record an RGB video. |

## Sharpa joint order on the real hand

The real hand's `set_joint_position()` expects **cfg / Sharpa order**, which is not
the same as the sorted USD order the policy acts in. Sending sorted-order targets
to the hand maps joints to the wrong motors. See [JOINT_ORDERING.md](JOINT_ORDERING.md)
for the exact conversion — this is the single most safety-critical detail in deploy.
