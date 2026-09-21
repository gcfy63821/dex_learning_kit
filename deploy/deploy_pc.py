"""Deploy entry script for PointCloud students (DAgger / PPO). POLYMETIS-ONLY.

Auto-detects DAgger PointCloudStudent vs PPO ActorCriticPointCloud, launches the
real-robot deploy env `franka-sharpa-pointcloud-polymetis-deploy` (arm via the NUC
Polymetis bridge over ZMQ), and pipes ZMQ camera depth into the env via
`set_depth_source`. See docs/DEPLOY.md for the full startup sequence.

Pre-flight:
  1) NUC: Polymetis server + `python deploy/polymetis_joint_bridge.py`
  2) Camera host: RealSense depth(+color) ZMQ publisher (PUB :5562)
  3) (optional) `python deploy/move_to_frame_polymetis.py --ip <NUC_IP> --pkl ... --frame 0 --hold`

Usage:
    python deploy/deploy_pc.py \\
        --task franka-sharpa-pointcloud-polymetis-deploy \\
        --load_path <pc_student.pth> --side right \\
        --polymetis_ip <NUC_IP> --depth_zmq_addr tcp://<CAM_HOST>:5562 \\
        --camera_extrinsic <path/to/cam_in_armbase_4x4.npy> \\
        --data_idx '["rt/0416_grasp/cube_small_2"]'
"""
import argparse
import sys
import json
import ast
import os

from dexx import deploy_config as _dcfg  # single source of truth for ports/addrs

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deploy a PointCloud student on real robot.")
parser.add_argument("--task", type=str, default="franka-sharpa-pointcloud-polymetis-deploy")
parser.add_argument("--load_path", type=str, required=True)
parser.add_argument("--side", type=str, default="right")
parser.add_argument("--data_idx", type=str, default=None,
                    help="JSON list of demo idx — used by env to compute reference poses. "
                         "For PC deploy the env still needs demo data for target wrist/joint refs.")
parser.add_argument("--cache", type=str, default=None)
parser.add_argument("--max_steps", type=int, default=10_000_000)
parser.add_argument("--camera_extrinsic", type=str, default=None,
                    help="Path to a 4x4 npy file with cam-in-armbase transform "
                         "(per-mount calibration). If None, uses the hardcoded default.")
parser.add_argument("--depth_height", type=int, default=240)
parser.add_argument("--depth_width", type=int, default=320)

# Env-side PC transforms. All default to None = "take it from the ckpt's
# pc_env_meta" (see algo/dagger/pc_env_meta.py). Pass one only to deliberately
# deviate from training — e.g. --pc_ablate_tactile_force for a real-robot
# tactile ablation of an otherwise unchanged policy.
parser.add_argument("--pc_ablate_tactile_pc", action="store_true", default=None)
parser.add_argument("--pc_ablate_tactile_force", action="store_true", default=None)
parser.add_argument("--pc_force_repr", type=str, default=None, choices=["scalar", "binary"])
parser.add_argument("--pc_tactile_force_gate", type=float, default=None,
                    help="If >0, gate tactile points whose force < this.")
parser.add_argument("--pc_tactile_gate_mode", type=str, default=None,
                    choices=["zero", "mask"])
parser.add_argument("--pc_force_scale", type=float, default=None,
                    help="Divisor applied to tactile_force. Restored from ckpt if unset. "
                         "MUST match training or the real hand's forces land in a "
                         "range the policy never saw.")
parser.add_argument("--pc_tactile_use_vec3", action="store_true", default=None,
                    help="Enable 3D tactile force (feat_dim=3). Restored from ckpt if unset.")
parser.add_argument("--no_contact_force", action="store_true", default=None,
                    help="Zero the 5d proprio contact force. Restored from ckpt if unset.")
parser.add_argument("--no_tactile", action="store_true", default=None,
                    help="Zero the whole 20d proprio tactile tail. Restored from ckpt if unset.")

# --- RGB recording (optional, same scheme as deploy.py) ---
parser.add_argument("--record_rgb", action="store_true",
                    help="Subscribe to a RealSense RGB topic during deploy. On exit "
                         "(Ctrl+C or env quit), AUTO-saves the buffered frames as MP4 "
                         "to `<record_dir>/<record_label or ckpt-name>_<TS>.mp4`.")
parser.add_argument("--record_rgb_topic", type=str, default="/camera/color/image_raw")
parser.add_argument("--record_fps", type=float, default=30.0)
parser.add_argument("--record_max_seconds", type=float, default=120.0)
parser.add_argument("--record_dir", type=str, default="logs/deploy_videos")
parser.add_argument("--record_label", type=str, default=None,
                    help="Filename prefix. Default = ckpt basename.")
# --- Action safety ramp ---
parser.add_argument("--action_ramp_steps", type=int, default=0,
                    help="Linearly (smoothstep) scale action from 0 → 1 over first N "
                         "policy steps. Mitigates large initial joint velocity right "
                         "after reset. 30 ≈ 1s @ 30Hz recommended for first real-robot trial. "
                         "0 = off (default).")
# --- Workspace bbox override (for sim2real PC debug; default = env_cfg value) ---
parser.add_argument("--pc_workspace_min", type=str, default=None,
                    help='Override env_cfg.pc_workspace_min as "x,y,z" (env-local frame). '
                         'Example: "0.0,-0.45,0.40" lowers z_min so cube points near table '
                         'edge survive crop. WARNING: deviates from training distribution; '
                         'use only for debug PC visibility on real robot, not as final fix.')
parser.add_argument("--pc_workspace_max", type=str, default=None,
                    help='Override env_cfg.pc_workspace_max as "x,y,z" (env-local frame). '
                         'Example: "0.85,0.3,1.0".')
parser.add_argument("--pc_no_crop", action="store_true",
                    help="Disable workspace bbox crop entirely (sets bbox to ±10m). "
                         "Use to verify in RViz which regions the RealSense camera "
                         "actually sees; the /deploy/scene_pc will be ALL valid depth "
                         "back-projected to env-local frame. Heavily OOD for policy.")

# ---- Arm backend: Polymetis (NUC, ZMQ bridge) instead of ROS2 ----
parser.add_argument("--polymetis_ip", type=str, default=None,
                    help="If set, drive the arm through the NUC Polymetis bridge at this IP "
                         "(ZMQ). Also pass --task franka-sharpa-pointcloud-polymetis-deploy.")
parser.add_argument("--polymetis_kq", type=str, default=None,
                    help='7 comma-separated joint-impedance stiffnesses, e.g. '
                         '"200,200,200,200,100,100,50". Unset = Polymetis defaults.')
parser.add_argument("--polymetis_kqd", type=str, default=None,
                    help='7 comma-separated joint-impedance dampings; must be given with --polymetis_kq.')
parser.add_argument("--polymetis_state_port", type=int, default=_dcfg.POLYMETIS_STATE_PORT)
parser.add_argument("--polymetis_cmd_port", type=int, default=_dcfg.POLYMETIS_CMD_PORT)
# ---- Depth backend: ZMQ from the camera host instead of a local ROS2 topic ----
parser.add_argument("--depth_backend", type=str, default="zmq", choices=["zmq"],
                    help="zmq: subscribe realsense_depth_zmq_pub.py on the camera host. "
                         "(This build is Polymetis-only; the ROS2 depth path was removed.)")
parser.add_argument("--depth_zmq_addr", type=str, default=None,
                    help="ZMQ addr of the camera-host depth publisher, e.g. "
                         "tcp://101.6.90.122:5562 (required when --depth_backend zmq).")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# This build is Polymetis-only (the ROS2 arm/depth deploy path was removed).
if not args_cli.polymetis_ip:
    raise SystemExit(
        "deploy_pc.py is Polymetis-only: pass --polymetis_ip <NUC_IP> "
        "(and --depth_zmq_addr tcp://<CAM_HOST>:5562). See docs/DEPLOY.md.")
if "polymetis" not in args_cli.task:
    raise SystemExit(
        f"--task must be the polymetis variant "
        f"(franka-sharpa-pointcloud-polymetis-deploy); got '{args_cli.task}'.")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------------------- #
import time
import importlib
import threading

import numpy as np
import torch
import gymnasium as gym
from omegaconf import OmegaConf

import dexx.tasks.franka_sharpa  # noqa: F401

from dexx.algo.dagger.pc_env_meta import apply_pc_env_meta

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_entry_point(entry_point: str):
    module, target = entry_point.split(":")
    if target.endswith("Cfg") or target[0].isupper():
        mod = importlib.import_module(module)
        return getattr(mod, target)()
    if target.endswith(".yaml") or target.endswith(".yml"):
        mod = importlib.import_module(module)
        base_dir = os.path.dirname(mod.__file__)
        return OmegaConf.load(os.path.join(base_dir, target))
    raise ValueError(entry_point)


def _build_dagger_student(ckpt, device):
    from dexx.algo.dagger.pointcloud_student import PointCloudStudent
    # PointNet per-point input dim = 3 (xyz) + type_dim + tactile_feat_dim.
    # type_dim is 1 for "scalar" and 3 for "onehot" — must match training.
    # The trainer (dexx/algo/dagger/dagger_pointcloud.py:356) saves
    # `pc_type_repr` in the ckpt; default to "scalar" for older ckpts.
    pc_cfg = dict(
        fusion_strategy=ckpt["pc_fusion_strategy"],
        n_scene=ckpt["n_scene"], n_hand=ckpt["n_hand"], n_tactile=ckpt["n_tactile"],
        tactile_feat_dim=ckpt["tactile_feat_dim"], output_dim=ckpt["pc_output_dim"],
        ablate_tactile_pc=ckpt.get("ablate_tactile_pc", False),
        type_repr=ckpt.get("pc_type_repr", "scalar"),
    )
    model = PointCloudStudent(
        proprio_dim=ckpt["proprio_dim"], action_dim=ckpt["action_dim"],
        pc_encoder_cfg=pc_cfg,
        hidden=tuple(ckpt.get("student_hidden", (512, 256))),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[DeployPC] PointCloudStudent built: type_repr={pc_cfg['type_repr']!r} "
          f"tactile_feat_dim={pc_cfg['tactile_feat_dim']} → "
          f"per-point in_dim={3 + (1 if pc_cfg['type_repr']=='scalar' else 3) + pc_cfg['tactile_feat_dim']}",
          flush=True)
    return model, "DAgger PointCloudStudent"


def _build_ppo_student(ckpt, device):
    from dexx.algo.ppo.actor_critic_pointcloud import ActorCriticPointCloud
    cfg = ckpt["cfg"]
    pc_cfg = dict(
        fusion_strategy=cfg.pc_fusion_strategy,
        n_scene=cfg.n_scene, n_hand=cfg.n_hand, n_tactile=cfg.n_tactile,
        tactile_feat_dim=cfg.tactile_feat_dim, output_dim=cfg.pc_output_dim,
        ablate_tactile_pc=getattr(cfg, "ablate_tactile_pc", False),
    )
    model = ActorCriticPointCloud(dict(
        actions_num=cfg.action_dim, input_shape=(cfg.proprio_dim,),
        actor_units=list(cfg.actor_units),
        priv_mlp_units=[256, 128, cfg.priv_info_dim],
        priv_info_dim=cfg.priv_info_dim, critic_units=list(cfg.critic_units),
        pc_config=pc_cfg,
    )).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, "PPO ActorCriticPointCloud (mu only)"


def _hand_body_subset_for_n_hand(n_hand: int):
    """Return canonical hand_body_names list for a given n_hand. Mirrors
    train_dagger_pc.py + eval_dagger_pc.py logic so deploy env matches ckpt."""
    subsets = {
        6: ["hand_C_MC", "thumb_fingertip", "index_fingertip", "middle_fingertip",
            "ring_fingertip", "pinky_fingertip"],
        11: ["hand_C_MC", "thumb_CMC_VL", "index_MCP_VL", "middle_MCP_VL",
             "ring_MCP_VL", "pinky_MCP_VL", "thumb_fingertip", "index_fingertip",
             "middle_fingertip", "ring_fingertip", "pinky_fingertip"],
        22: ["hand_C_MC", "thumb_CMC_VL", "index_MCP_VL", "middle_MCP_VL",
             "ring_MCP_VL", "pinky_MCP_VL", "thumb_MCP_VL", "index_PP", "middle_PP",
             "ring_PP", "pinky_PP", "thumb_MC", "index_MP", "middle_MP", "ring_MP",
             "pinky_MP", "thumb_fingertip", "index_fingertip", "middle_fingertip",
             "ring_fingertip", "pinky_fingertip", "thumb_IP"],
    }
    return subsets.get(n_hand)


def main():
    # ---- Load env cfg + pre-align with DAgger ckpt metadata
    spec = gym.spec(args_cli.task)
    env_cfg = parse_entry_point(spec.kwargs["env_cfg_entry_point"])
    env_cfg.scene.num_envs = 1
    if args_cli.device:
        env_cfg.sim.device = args_cli.device
    env_cfg.hand_side = args_cli.side
    if args_cli.cache:
        env_cfg.grasp_cache_path = args_cli.cache
    if args_cli.data_idx:
        try:
            di = json.loads(args_cli.data_idx)
        except json.JSONDecodeError:
            di = ast.literal_eval(args_cli.data_idx)
        env_cfg.data_indices = di

    # Disable DR + augmentation
    for fl in ("randomize_pd_gains", "randomize_friction", "randomize_mass",
               "randomize_com", "enable_obj_pose_noise", "enable_depth_noise"):
        if hasattr(env_cfg, fl):
            setattr(env_cfg, fl, False)
    for fl in ("pc_jitter_std", "pc_dropout_ratio", "pc_hand_noise_std",
               "pc_force_noise_ratio", "pc_force_dropout_prob"):
        if hasattr(env_cfg, fl):
            setattr(env_cfg, fl, 0.0)
    if hasattr(env_cfg, "init_curriculum_enabled"):
        env_cfg.init_curriculum_enabled = False

    # NOTE: the env-side PC transforms are restored from the ckpt further down,
    # once it has been pre-loaded.

    # Workspace bbox override (sim2real debug flag).
    def _parse_xyz(s: str):
        a, b, c = (float(x) for x in s.split(","))
        return (a, b, c)
    if args_cli.pc_workspace_min is not None:
        old = tuple(getattr(env_cfg, "pc_workspace_min", (None,) * 3))
        env_cfg.pc_workspace_min = _parse_xyz(args_cli.pc_workspace_min)
        print(f"[DeployPC] OVERRIDE env_cfg.pc_workspace_min: {old} -> "
              f"{env_cfg.pc_workspace_min}  (deviates from training!)", flush=True)
    if args_cli.pc_workspace_max is not None:
        old = tuple(getattr(env_cfg, "pc_workspace_max", (None,) * 3))
        env_cfg.pc_workspace_max = _parse_xyz(args_cli.pc_workspace_max)
        print(f"[DeployPC] OVERRIDE env_cfg.pc_workspace_max: {old} -> "
              f"{env_cfg.pc_workspace_max}  (deviates from training!)", flush=True)
    if args_cli.pc_no_crop:
        env_cfg.pc_workspace_min = (-10.0, -10.0, -10.0)
        env_cfg.pc_workspace_max = ( 10.0,  10.0,  10.0)
        print(f"[DeployPC] !!! --pc_no_crop set: workspace bbox = ±10m. "
              f"Policy WILL be OOD; this is for RViz PC-coverage debug only !!!",
              flush=True)

    # Pre-peek DAgger ckpt to align env's n_hand / hand body names
    print(f"[DeployPC] pre-loading ckpt: {args_cli.load_path}", flush=True)
    ckpt = torch.load(args_cli.load_path, map_location="cpu", weights_only=False)
    for k_ckpt, k_env in [("n_scene", "pc_num_scene_points"),
                          ("n_hand", "pc_num_hand_points"),
                          ("n_tactile", "pc_num_tactile_points")]:
        if k_ckpt in ckpt and hasattr(env_cfg, k_env):
            v_ckpt = int(ckpt[k_ckpt])
            if v_ckpt != int(getattr(env_cfg, k_env)):
                print(f"[DeployPC] aligning env.{k_env}: "
                      f"{getattr(env_cfg, k_env)} -> {v_ckpt} (from ckpt)", flush=True)
                setattr(env_cfg, k_env, v_ckpt)
    n_hand = int(getattr(env_cfg, "pc_num_hand_points", 11))
    subset = _hand_body_subset_for_n_hand(n_hand)
    if subset is not None:
        env_cfg.pc_hand_body_names = subset
        print(f"[DeployPC] pc_hand_body_names set to {n_hand}-body subset", flush=True)

    # Restore the env-side PC transforms the student was TRAINED with. On the
    # real robot this matters most for `pc_force_scale`: the Sharpa SDK's F6
    # forces are fed through the same divisor as in sim, so getting it wrong
    # puts every contact reading in a range the policy never saw. CLI wins.
    apply_pc_env_meta(
        env_cfg,
        ckpt,
        overrides={
            "pc_force_scale": args_cli.pc_force_scale,
            "pc_tactile_force_gate": args_cli.pc_tactile_force_gate,
            "pc_tactile_gate_mode": args_cli.pc_tactile_gate_mode,
            "pc_force_repr": args_cli.pc_force_repr,
            "pc_ablate_tactile_pc": args_cli.pc_ablate_tactile_pc,
            "pc_ablate_tactile_force": args_cli.pc_ablate_tactile_force,
            "pc_tactile_use_vec3": args_cli.pc_tactile_use_vec3,
            "enable_contact_force": False if args_cli.no_contact_force else None,
            "enable_tactile": False if args_cli.no_tactile else None,
        },
        tag="DeployPC",
    )

    # Legacy fallback for pre-`pc_env_meta` ckpts: align env's tactile_force dim
    # with ckpt's `tactile_feat_dim`. If ckpt was trained with vec3 but env
    # defaults to scalar, the encoder forward hits a (N,25,3) vs (N,25,1)
    # shape mismatch on tactile_force.
    if "tactile_feat_dim" in ckpt:
        feat_dim_ckpt = int(ckpt["tactile_feat_dim"])
        use_vec3_needed = (feat_dim_ckpt == 3)
        cur_use_vec3 = bool(getattr(env_cfg, "pc_tactile_use_vec3", False))
        if use_vec3_needed != cur_use_vec3:
            env_cfg.pc_tactile_use_vec3 = use_vec3_needed
            print(f"[DeployPC] aligning env.pc_tactile_use_vec3: "
                  f"{cur_use_vec3} -> {use_vec3_needed} (ckpt tactile_feat_dim={feat_dim_ckpt})",
                  flush=True)

    # Polymetis arm backend: bind NUC bridge IP/ports onto the cfg (the polymetis
    # PC deploy env reads these in _setup_comm). Requires --task ...-polymetis-deploy.
    if args_cli.polymetis_ip:
        env_cfg.polymetis_server_ip = args_cli.polymetis_ip
        env_cfg.polymetis_state_port = args_cli.polymetis_state_port
        env_cfg.polymetis_cmd_port = args_cli.polymetis_cmd_port
        def _parse7(v, name):
            if v is None:
                return None
            xs = [float(x) for x in v.split(",")]
            if len(xs) != 7:
                raise ValueError(f"{name} needs 7 values, got {len(xs)}")
            return xs
        _kq = _parse7(args_cli.polymetis_kq, "--polymetis_kq")
        _kqd = _parse7(args_cli.polymetis_kqd, "--polymetis_kqd")
        if (_kq is None) != (_kqd is None):
            raise ValueError("--polymetis_kq and --polymetis_kqd must be given together.")
        if _kq is not None:
            env_cfg.polymetis_kq, env_cfg.polymetis_kqd = _kq, _kqd
            print(f"[DeployPC] polymetis joint impedance OVERRIDE Kq={_kq} Kqd={_kqd}", flush=True)
        print(f"[DeployPC] arm backend = Polymetis @ {args_cli.polymetis_ip}:"
              f"{args_cli.polymetis_state_port}/{args_cli.polymetis_cmd_port}", flush=True)
        if "polymetis" not in args_cli.task:
            print(f"[DeployPC] WARNING: --polymetis_ip set but --task='{args_cli.task}' "
                  f"is not the polymetis variant; use "
                  f"--task franka-sharpa-pointcloud-polymetis-deploy.", flush=True)

    # ---- Build env (this initialises ROS2/Polymetis, Sharpa SDK, pk FK chain, etc.)
    print(f"[DeployPC] creating env: {args_cli.task}", flush=True)
    env_raw = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    base_env = env_raw.unwrapped
    device = torch.device(str(base_env.device))

    # ---- Optional: load calibrated camera extrinsic
    if args_cli.camera_extrinsic and os.path.exists(args_cli.camera_extrinsic):
        T = np.load(args_cli.camera_extrinsic).astype(np.float32)
        assert T.shape == (4, 4)
        base_env.set_camera_extrinsic(T)
    else:
        print("[DeployPC] using DEFAULT camera extrinsic — calibrate before serious deploy!",
              flush=True)

    # ---- Build model
    sd_keys = list(ckpt.get("model", {}).keys())
    if any(k.startswith("actor_mlp.") for k in sd_keys):
        model, arch = _build_ppo_student(ckpt, device)
    elif any(k == "mlp.0.weight" or k.startswith("mlp.") for k in sd_keys):
        model, arch = _build_dagger_student(ckpt, device)
    else:
        raise RuntimeError(f"Unknown ckpt format; first 5 sd keys: {sd_keys[:5]}")
    print(f"[DeployPC] arch: {arch}", flush=True)

    # ---- Restore the student's trained observation layout --------------------
    # A lean student (--student_drop_slots) was trained on a sliced proprio
    # vector; the env still publishes the full one. Without this the first
    # forward fails on a shape mismatch — which is the good case. Masking runs
    # before slicing, because mask indices refer to the env's original layout.
    _mask_idx = list(ckpt.get("student_obs_mask_idx", []) or [])
    _keep_idx = list(ckpt.get("student_keep_idx", []) or [])
    _mask_t = (torch.as_tensor(_mask_idx, dtype=torch.long, device=device)
               if _mask_idx else None)
    _keep_t = (torch.as_tensor(_keep_idx, dtype=torch.long, device=device)
               if _keep_idx else None)
    if _keep_t is not None:
        print(f"[DeployPC] LEAN STUDENT ckpt: proprio sliced to {len(_keep_idx)}d, "
              f"dropped {list(ckpt.get('student_drop_slots', []) or [])}", flush=True)
    if _mask_t is not None:
        print(f"[DeployPC] SPARSE-REF ckpt: zeroing {len(_mask_idx)} proprio dims",
              flush=True)
    # Fail with a sentence rather than a matmul shape error three frames deep.
    _expect = int(ckpt["proprio_dim"])
    _have = len(_keep_idx) if _keep_idx else None
    if _have is not None and _have != _expect:
        raise RuntimeError(
            f"checkpoint says proprio_dim={_expect} but its keep-index selects "
            f"{_have} dims; the ckpt is inconsistent and would fail at the first "
            f"forward.")

    # ---- Depth subscriber: ZMQ from the camera host (Polymetis-only build) ----
    ros_exec = None  # kept for the shared cleanup path below (always None here)
    if not args_cli.depth_zmq_addr:
        raise SystemExit("--depth_zmq_addr is required (e.g. tcp://<CAM_HOST>:5562)")
    from dexx.scripts.deploy.realsense_depth_zmq_subscriber import (
        RealSenseDepthZmqSubscriber,
    )
    depth_sub = RealSenseDepthZmqSubscriber(
        addr=args_cli.depth_zmq_addr,
        height=args_cli.depth_height, width=args_cli.depth_width,
        device=str(device),
    )
    base_env.set_depth_source(depth_sub.get_latest)
    print(f"[DeployPC] depth backend = ZMQ @ {args_cli.depth_zmq_addr}", flush=True)

    # ---- Make obs adapter for ckpt input
    def _make_inp(d):
        obs_in = d["policy"]
        if _mask_t is not None:
            obs_in = obs_in.clone()
            obs_in[:, _mask_t] = 0.0
        if _keep_t is not None:
            obs_in = obs_in[:, _keep_t]
        return {
            "obs": obs_in,
            "scene_pc": d["scene_pc"], "scene_mask": d["scene_mask"],
            "hand_pc": d["hand_pc"],
            "tactile_pc": d["tactile_pc"], "tactile_force": d["tactile_force"],
        }

    # ---- Wait for first depth frame
    t0 = time.time()
    while depth_sub.n_received == 0 and time.time() - t0 < 5.0:
        time.sleep(0.1)
    if depth_sub.n_received == 0:
        print("[DeployPC] WARNING: no depth frame after 5s — running anyway (scene_pc=zeros)",
              flush=True)

    # ---- Optional RGB recorder ----
    rgb_buf = None
    if args_cli.record_rgb:
        try:
            from dexx.scripts.deploy.rgb_video_buffer import (
                RGBVideoBuffer, interactive_prompt_and_save,
            )
            rgb_buf = RGBVideoBuffer(
                topic=args_cli.record_rgb_topic,
                max_seconds=float(args_cli.record_max_seconds),
                fps_hint=float(args_cli.record_fps),
            )
            rgb_buf.start_spin()
            print(f"[DeployPC] RGB recorder armed on {args_cli.record_rgb_topic} "
                  f"(max {args_cli.record_max_seconds}s).", flush=True)
        except Exception as _e:
            print(f"[DeployPC] WARNING: could not start RGB recorder: {_e}", flush=True)
            rgb_buf = None

    # ---- Per-rollout video save: monkey-patch env's _flush_debug_recording
    # so every time the env saves a deploy_debug_*.npz, we also flush the
    # current RGB buffer to a sibling .mp4 (same basename, .mp4 extension).
    # This pairs each rollout with its video file 1:1.
    _last_saved_npz = {"path": None}
    if rgb_buf is not None and hasattr(base_env, "_flush_debug_recording"):
        _orig_flush = base_env._flush_debug_recording

        def _wrapped_flush(reason):
            # Snapshot the (next) saved npz filename BEFORE flush by reading
            # the env's run idx + frame; reproduce the path format from
            # franka_sharpa_force_deploy_env_v2.py:_flush_debug_recording.
            try:
                _orig_flush(reason)
            finally:
                # Find the most recently written .npz in env.debug_record_dir
                try:
                    rec_dir = getattr(base_env, "debug_record_dir", None)
                    if rec_dir and os.path.isdir(rec_dir):
                        npzs = sorted(
                            (os.path.join(rec_dir, f) for f in os.listdir(rec_dir)
                             if f.endswith(".npz")),
                            key=os.path.getmtime,
                            reverse=True,
                        )
                        if npzs:
                            newest = npzs[0]
                            if newest != _last_saved_npz["path"]:
                                _last_saved_npz["path"] = newest
                                _flush_rgb_buf_paired(newest, reason)
                except Exception as _e:
                    print(f"[DeployPC] paired video save: discover err: {_e}", flush=True)

        def _flush_rgb_buf_paired(npz_path, reason):
            if rgb_buf is None or rgb_buf.n_frames == 0:
                print(f"[DeployPC] paired video: npz at {os.path.basename(npz_path)} "
                      f"but RGB buffer empty (received={rgb_buf.n_received if rgb_buf else 0}). "
                      f"Skipping mp4 save.", flush=True)
                return
            mp4_path = os.path.splitext(npz_path)[0] + ".mp4"
            try:
                rgb_buf.save_mp4(mp4_path, fps=float(args_cli.record_fps))
                print(f"[DeployPC] ✓ paired video saved: {mp4_path} "
                      f"({rgb_buf.n_frames} frames, reason={reason})", flush=True)
            except Exception as _e:
                print(f"[DeployPC] paired video save err: {_e}", flush=True)
            finally:
                # Clear buffer so next rollout starts fresh.
                rgb_buf.discard()

        base_env._flush_debug_recording = _wrapped_flush
        print(f"[DeployPC] paired-video hook installed: every npz → matching mp4 "
              f"(in env.debug_record_dir={getattr(base_env, 'debug_record_dir', 'N/A')})",
              flush=True)

    # ---- Reset + run
    obs, _ = env_raw.reset()
    print("[DeployPC] entering control loop @ 30Hz (Ctrl+C to stop)", flush=True)
    step = 0
    _ramp_N = int(args_cli.action_ramp_steps)
    if _ramp_N > 0:
        print(f"[DeployPC] action ramp-up enabled: first {_ramp_N} steps scaled "
              f"smoothstep(0→1) (~{_ramp_N/30.0:.1f}s @ 30Hz)", flush=True)
    try:
        while simulation_app.is_running() and step < args_cli.max_steps:
            with torch.no_grad():
                action = model.act_inference(_make_inp(obs))
                action = torch.clamp(action, -1.0, 1.0)
            # Action ramp-up: smoothstep(0→1) over first N steps. Mitigates
            # large initial joint velocity on real-robot start.
            if _ramp_N > 0 and step < _ramp_N:
                s = min(step / float(_ramp_N), 1.0)
                scale = float(s * s * (3.0 - 2.0 * s))   # smoothstep
                action = action * scale
                if step % 5 == 0:
                    print(f"  [DeployPC-RAMP] step {step}/{_ramp_N} action_scale={scale:.3f}",
                          flush=True)
            obs, _r, done, _trunc, _info = env_raw.step(action)
            step += 1
            if step % 30 == 0:
                rx = f" rgb_rx={rgb_buf.n_received}" if rgb_buf else ""
                print(f"  [DeployPC] step={step} depth_rx={depth_sub.n_received}{rx}",
                      flush=True)
    except KeyboardInterrupt:
        print("\n[DeployPC] KeyboardInterrupt — leaving rollout loop.", flush=True)

    print("[DeployPC] exiting", flush=True)

    # ---- Final flush: if any frames are buffered (e.g. Ctrl+C mid-rollout
    # before env got to save its npz), dump them as an "orphan" mp4 in
    # --record_dir. If buffer is empty (paired-hook already saved everything
    # cleanly per-rollout), this is a no-op.
    if rgb_buf is not None:
        try:
            rgb_buf.stop_spin()
            if rgb_buf.n_frames > 0:
                ckpt_name = os.path.splitext(os.path.basename(args_cli.load_path))[0]
                label = args_cli.record_label or ckpt_name
                from datetime import datetime
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(args_cli.record_dir, f"{label}_orphan_{stamp}.mp4")
                rgb_buf.save_mp4(out_path, fps=float(args_cli.record_fps))
                print(f"[DeployPC] ✓ orphan video (buffered after last npz save) → {out_path}",
                      flush=True)
            else:
                print(f"[DeployPC] all rollouts saved as paired mp4s alongside npz "
                      f"(received {rgb_buf.n_received} total RGB frames).", flush=True)
        except Exception as _e:
            print(f"[DeployPC] final flush error: {_e}")

    try:
        ros_exec.shutdown()
        depth_sub.destroy_node()
    except Exception:
        pass
    env_raw.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
