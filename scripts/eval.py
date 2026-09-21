"""Per-episode eval for a DAgger / PPO PointCloud student.

Like `play_dagger_pc.py` (auto-detects DAgger PointCloudStudent vs PPO
ActorCriticPointCloud from the ckpt) but records `end_final_dist`,
`fail_causes`, `demo_idx` per episode and dumps `records.json` — the same
schema as `eval_policy.py` — so strict3/strict2/strict5 funnel post-processing
just works.

Usage:
    python dexx/scripts/gym_style/eval_dagger_pc.py \
        --task franka-sharpa-pointcloud \
        --load_path logs/.../dagger_final.pth \
        --side right --data_idx '[...]' \
        --num_envs 128 --max_episodes 5000 --out_dir logs/eval_dagger_pc/<tag>
"""
import argparse
import sys
import json
import ast

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="franka-sharpa-pointcloud")
parser.add_argument("--load_path", type=str, required=True)
parser.add_argument("--side", type=str, default="right")
parser.add_argument("--data_idx", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--cache", type=str, default=None)
parser.add_argument("--max_episodes", type=int, default=5000)
parser.add_argument("--max_steps", type=int, default=12000)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--success_dist", type=float, default=0.05,
                    help="Env-internal success threshold (min_final_dist<this). "
                         "Post-hoc strict thresholds use end_final_dist, not this.")
parser.add_argument("--label", type=str, default=None)
parser.add_argument("--save_traj", action="store_true", default=False,
                    help="Save per-frame trajectory arrays (traj_pos_dist, traj_rot_deg) "
                         "in records.json. Max/mean summary stats are always saved.")
# Env-side PC transforms. All default to None = "take it from the ckpt's
# pc_env_meta" (see algo/dagger/pc_env_meta.py). Pass one only to deliberately
# deviate from training — e.g. --pc_ablate_tactile_force for a modality ablation
# of an otherwise unchanged policy.
parser.add_argument("--pc_ablate_tactile_pc", action="store_true", default=None)
parser.add_argument("--pc_ablate_tactile_force", action="store_true", default=None)
parser.add_argument("--pc_force_repr", type=str, default=None,
                    choices=["scalar", "binary"])
parser.add_argument("--pc_tactile_force_gate", type=float, default=None,
                    help="If >0, gate tactile points whose force < this at inference.")
parser.add_argument("--pc_tactile_gate_mode", type=str, default=None,
                    choices=["zero", "mask"])
parser.add_argument("--pc_force_scale", type=float, default=None,
                    help="Divisor applied to tactile_force. Restored from ckpt if unset.")
parser.add_argument("--pc_tactile_use_vec3", action="store_true", default=None,
                    help="Enable 3D tactile force (sets feat_dim=3). Restored from ckpt if unset.")
parser.add_argument("--no_contact_force", action="store_true", default=None,
                    help="Zero the 5d proprio contact force. Restored from ckpt if unset.")
parser.add_argument("--no_tactile", action="store_true", default=None,
                    help="Zero the whole 20d proprio tactile tail. Restored from ckpt if unset.")
# Optional PC noise INJECTION at eval time (sim2real preview).
# By default the eval script forcibly zeros all PC noise for clean eval — use
# these flags to override and inject noise back in to gauge sim2real robustness
# without retraining.
parser.add_argument("--inject_jitter", type=float, default=None,
                    help="Override pc_jitter_std at eval time (e.g. 0.005 = 5mm).")
parser.add_argument("--inject_dropout", type=float, default=None,
                    help="Override pc_dropout_ratio at eval time (e.g. 0.10 = 10%).")
parser.add_argument("--inject_hand_noise", type=float, default=None,
                    help="Override pc_hand_noise_std at eval time (e.g. 0.003 = 3mm).")
# Stage-1 estimator drop-in (optional)
parser.add_argument("--stage1_ckpt", type=str, default=None,
                    help="Estimator+planner ckpt (best.pt). When set, replaces 19d obs "
                         "the target_obj_pose, tips_distance and obj_pose_tail slots with estimator outputs before "
                         "feeding the student. Goal-quat for hemisphere alignment is "
                         "extracted from demo final frame.")
parser.add_argument("--estimator_inject_pose_noise", action="store_true", default=False,
                    help="Add training-time PoseObs noise (8mm/0.06rad) to the [550:557] "
                         "slot when using estimator drop-in.")
parser.add_argument("--use_planner", action="store_true", default=False,
                    help="ALSO use the planner (from --stage1_ckpt) to replace K=1 wrist/"
                         "joints demo target slots (the ref_tracking block) via env.set_planner_targets(). "
                         "Combined with estimator drop-in this is fully demo-free obs.")
parser.add_argument("--flow_planner_ckpt", type=str, default=None,
                    help="If set, use FlowMatchingPlanner from this ckpt instead of the "
                         "MLP planner inside --stage1_ckpt. Implies --use_planner.")
parser.add_argument("--perturb_obj_xy", type=float, default=0.0,
                    help="Eval-time random xy perturbation of obj init pos (meters). "
                         "0=off, 0.05=±5cm uniform. Hooks env._eval_perturb_obj_xy.")
parser.add_argument("--per_demo_quota", type=int, default=None,
                    help="Episodes to collect PER DEMO. Default None = balanced, "
                         "ceil(max_episodes / n_demos). Pass 0 for the old "
                         "first-to-finish collection, which is biased: a successful "
                         "episode ends sooner than a failing one, so the easy demos "
                         "fill the quota first and the aggregate is weighted toward "
                         "them. Per-demo rates are reported either way.")
parser.add_argument("--keep_physics_dr", action="store_true", default=False,
                    help="Leave the physical domain randomisation ON during eval: object "
                         "mass 0.01-0.15 kg, friction x1.0-2.5, COM +-2 cm, PD gains x0.5-2. "
                         "It is force-disabled by default for a clean deterministic eval, "
                         "but that also removes the only variation contact force could help "
                         "with — a policy cannot show a benefit from sensing grip force when "
                         "every object weighs its nominal mass. Turn this on to test whether "
                         "tactile pays off under the physical uncertainty it is meant for.")
parser.add_argument("--camera_extrinsic", type=str, default=None,
                    help="Path to a 4x4 .npy camera-in-armbase extrinsic (ROS optical). "
                         "Must match what the student was TRAINED with, or its scene cloud "
                         "arrives from a different viewpoint than it ever saw.")
parser.add_argument("--ref_root", type=str, default=None,
                    help="Override robotool_batch retarget reference root.")
parser.add_argument("--pc_ablate_scene_pc", action="store_true", default=False,
                    help="Modality ablation: drop the camera-derived scene points.")
parser.add_argument("--mask_obs_slots", type=str, default=None,
                    help="Comma-separated obs-slot ranges to zero before feeding student. "
                         "Format: 'lo:hi,lo:hi,...' e.g. '390:397,550:557' — take the bounds from the [obs-slots] line the env prints at startup, never from a remembered number; the block order has changed before. "
                         "Useful for K=1 demo-target field ablations (eval-only).")
parser.add_argument("--obj_shift_correction", type=str, default="none",
                    choices=["none", "oracle", "estimator"],
                    help="Geometric demo correction: shift K=1 wrist/joint target by "
                         "δ = current_obj_pos - demo_K1_obj_pos. Compensates for "
                         "object position perturbation. 'oracle' uses env GT obj_pos, "
                         "'estimator' uses --stage1_ckpt estimator's prediction.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------------------- #
import os
import time
import collections
import importlib
from dataclasses import dataclass, asdict

import torch
import gymnasium as gym
from omegaconf import OmegaConf

import dexx.tasks.franka_sharpa  # noqa: F401

from dexx.algo.dagger.pc_env_meta import align_pc_dims_to_ckpt, apply_pc_env_meta


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
    return model, "DAgger PointCloudStudent"


def _build_ppo_student(ckpt, device):
    from dexx.algo.ppo.actor_critic_pointcloud import ActorCriticPointCloud
    cfg = ckpt["cfg"]
    pc_cfg = dict(
        fusion_strategy=cfg.pc_fusion_strategy,
        n_scene=cfg.n_scene, n_hand=cfg.n_hand, n_tactile=cfg.n_tactile,
        tactile_feat_dim=cfg.tactile_feat_dim, output_dim=cfg.pc_output_dim,
        ablate_tactile_pc=getattr(cfg, "ablate_tactile_pc", False),
        type_repr=getattr(cfg, "pc_type_repr", "scalar"),
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


@dataclass
class EpisodeRecord:
    env_id: int
    demo_idx: str
    init_frame: int
    survival_len: int
    succeeded: bool
    min_final_dist: float
    end_final_dist: float
    min_final_rot_deg: float
    end_final_rot_deg: float
    max_traj_pos_dist: float
    mean_traj_pos_dist: float
    max_traj_rot_deg: float
    mean_traj_rot_deg: float
    traj_pos_dist: list
    traj_rot_deg: list
    fail_causes: list
    obj_start: list
    obj_end: list


def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)

    spec = gym.spec(args_cli.task)
    env_cfg = parse_entry_point(spec.kwargs["env_cfg_entry_point"])
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device:
        env_cfg.sim.device = args_cli.device
    if args_cli.side:
        env_cfg.hand_side = args_cli.side
    if args_cli.cache:
        env_cfg.grasp_cache_path = args_cli.cache
    if args_cli.data_idx:
        try:
            data_indices = json.loads(args_cli.data_idx)
        except json.JSONDecodeError:
            data_indices = ast.literal_eval(args_cli.data_idx)
        env_cfg.data_indices = data_indices

    # Clean eval: DR + aug off, init_curriculum off (uniform), random_state_init on.
    _off = ["enable_obj_pose_noise", "enable_depth_noise"]
    if not args_cli.keep_physics_dr:
        _off += ["randomize_pd_gains", "randomize_friction", "randomize_mass",
                 "randomize_com"]
    else:
        print("[EvalPC] PHYSICS DR KEPT ON: mass / friction / COM / PD gains stay "
              "randomised for this eval", flush=True)
    for fl in _off:
        if hasattr(env_cfg, fl):
            setattr(env_cfg, fl, False)
    for fl in ("pc_jitter_std", "pc_dropout_ratio", "pc_hand_noise_std",
               "pc_force_noise_ratio", "pc_force_dropout_prob"):
        if hasattr(env_cfg, fl):
            setattr(env_cfg, fl, 0.0)
    if hasattr(env_cfg, "init_curriculum_enabled"):
        env_cfg.init_curriculum_enabled = False
    if args_cli.pc_ablate_scene_pc:
        env_cfg.pc_ablate_scene_pc = True
        print("[EvalPC] pc_ablate_scene_pc = True (scene points dropped)", flush=True)
    if args_cli.camera_extrinsic:
        assert os.path.exists(args_cli.camera_extrinsic), \
            f"--camera_extrinsic not found: {args_cli.camera_extrinsic}"
        env_cfg.camera_extrinsic_path = os.path.abspath(args_cli.camera_extrinsic)
        print(f"[EvalPC] sim camera extrinsic <- {env_cfg.camera_extrinsic_path}", flush=True)
    if args_cli.ref_root:
        env_cfg.robotool_batch_retarget_root = args_cli.ref_root
        print(f"[EvalPC] robotool_batch_retarget_root <- {args_cli.ref_root}", flush=True)
    # NOTE: the env-side PC transforms (force scale / gate / repr / ablations)
    # are restored from the ckpt further down, once `_ckpt_peek` is loaded.
    # Sim2real preview: inject PC noise back AFTER the clean-eval zeroing above.
    if args_cli.inject_jitter is not None:
        env_cfg.pc_jitter_std = float(args_cli.inject_jitter)
        print(f"[EvalPC][noise] pc_jitter_std = {env_cfg.pc_jitter_std}", flush=True)
    if args_cli.inject_dropout is not None:
        env_cfg.pc_dropout_ratio = float(args_cli.inject_dropout)
        print(f"[EvalPC][noise] pc_dropout_ratio = {env_cfg.pc_dropout_ratio}", flush=True)
    if args_cli.inject_hand_noise is not None:
        env_cfg.pc_hand_noise_std = float(args_cli.inject_hand_noise)
        print(f"[EvalPC][noise] pc_hand_noise_std = {env_cfg.pc_hand_noise_std}", flush=True)

    # ----- pre-load ckpt so the env emits exactly the point counts / tactile
    # width the student was trained with. Shared with play.py; see
    # dexx.algo.dagger.pc_env_meta.
    _ckpt_peek = torch.load(args_cli.load_path, map_location="cpu", weights_only=False)
    align_pc_dims_to_ckpt(env_cfg, _ckpt_peek, tag="EvalPC")
    # Restore the env-side PC transforms the student was TRAINED with. Without
    # this a policy trained on e.g. force/10 with 0.05 gating is evaluated on
    # raw force with every tactile point active — a silent input-distribution
    # mismatch that moves success rate. CLI flags (if passed) still win.
    apply_pc_env_meta(
        env_cfg,
        _ckpt_peek,
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
        tag="EvalPC",
    )

    # (the legacy tactile_feat_dim -> vec3 fallback now lives in
    # align_pc_dims_to_ckpt above, so an explicit pc_env_meta block wins over it)
    print(f"[EvalPC] env: {args_cli.task} num_envs={args_cli.num_envs}", flush=True)
    env_raw = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    base_env = env_raw.unwrapped
    # Eval-time object perturbation
    if args_cli.perturb_obj_xy > 0:
        base_env._eval_perturb_obj_xy = float(args_cli.perturb_obj_xy)
        print(f"[EvalPC] OBJ PERTURBATION ENABLED: ±{args_cli.perturb_obj_xy*100:.1f}cm "
              f"random uniform in xy at every reset", flush=True)
    device = torch.device(str(base_env.device))

    expanded_indices = [str(x) for x in getattr(base_env, "data_indices", [])]
    print(f"[EvalPC] expanded data_indices: {len(expanded_indices)} variants; "
          f"env e -> data_indices[e % {len(expanded_indices)}]", flush=True)
    with open(os.path.join(args_cli.out_dir, "expanded_data_indices.json"), "w") as f:
        json.dump({"data_indices": expanded_indices}, f, indent=2)

    def _env_demo(env_id: int) -> str:
        if expanded_indices:
            return expanded_indices[env_id % len(expanded_indices)]
        return "unknown"

    print(f"[EvalPC] loading ckpt: {args_cli.load_path}", flush=True)
    ckpt = torch.load(args_cli.load_path, map_location=device, weights_only=False)
    sd_keys = list(ckpt.get("model", {}).keys())
    if any(k.startswith("actor_mlp.") for k in sd_keys):
        model, arch_label = _build_ppo_student(ckpt, device)
    elif any(k == "mlp.0.weight" or k.startswith("mlp.") for k in sd_keys):
        model, arch_label = _build_dagger_student(ckpt, device)
    else:
        raise RuntimeError(f"Unknown ckpt format; first 5 sd keys: {sd_keys[:5]}")
    print(f"[EvalPC] architecture: {arch_label}", flush=True)

    # Optional Stage-1 estimator drop-in (--stage1_ckpt). Replaces 19d obs slots
    # the target_obj_pose / tips_distance / obj_pose_tail slots with estimator
    # outputs before feeding the student.
    _est = None
    _pln = None
    if args_cli.stage1_ckpt is not None:
        from dexx.algo.goal_planner import (
            Estimator, EstimatorConfig, Planner, PlannerConfig,
        )
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from _planner_helpers import extract_proprio_subset, extract_goal  # noqa
        _s1 = torch.load(args_cli.stage1_ckpt, map_location=device, weights_only=False)
        # Back-compat: old quat-only ckpts have no rot_repr field
        _est_cfg_dict = dict(_s1["estimator_cfg"])
        _est_cfg_dict.setdefault("rot_repr", "quat")
        _est = Estimator(EstimatorConfig(**_est_cfg_dict)).to(device)
        _sd = dict(_s1["estimator_state_dict"])
        # Back-compat: old quat-only ckpts saved as head_quat, new code uses head_rot
        if "head_quat.weight" in _sd and "head_rot.weight" not in _sd:
            _sd["head_rot.weight"] = _sd.pop("head_quat.weight")
            _sd["head_rot.bias"] = _sd.pop("head_quat.bias")
        _est.load_state_dict(_sd)
        _est.eval()
        if args_cli.flow_planner_ckpt is not None:
            args_cli.use_planner = True
            from dexx.algo.goal_planner import FlowMatchingPlanner, FlowPlannerConfig
            _fck = torch.load(args_cli.flow_planner_ckpt, map_location=device, weights_only=False)
            _pln_flow = FlowMatchingPlanner(FlowPlannerConfig(**_fck["flow_planner_cfg"])).to(device)
            _pln_flow.load_state_dict(_fck["flow_planner_state_dict"])
            _pln_flow.eval()
            _target_mean = _fck["target_mean"].to(device)
            _target_std = _fck["target_std"].to(device)
            _pln = _pln_flow   # signals downstream block to call flow.sample()
            _is_flow_planner = True
            print(f"[EvalPC] FLOW PLANNER enabled from {args_cli.flow_planner_ckpt}", flush=True)
        elif args_cli.use_planner:
            _pln = Planner(PlannerConfig(**_s1["planner_cfg"])).to(device)
            _pln.load_state_dict(_s1["planner_state_dict"])
            _pln.eval()
            _is_flow_planner = False
            print(f"[EvalPC] MLP PLANNER also enabled — full demo-free obs", flush=True)
        else:
            _is_flow_planner = False
        # PC encoder from student model (PointCloudStudent or ActorCriticPointCloud)
        _encoder = getattr(model, "pc_encoder", None)
        if _encoder is None:
            raise RuntimeError("model has no pc_encoder; cannot do estimator drop-in")
        _expected_fdim = _encoder.tactile_feat_dim
        _POS_SIGMA = 0.008
        _ROT_SIGMA = 0.06
        print(f"[EvalPC] estimator drop-in ENABLED from {args_cli.stage1_ckpt}",
              flush=True)
        print(f"  → replacing the target_obj_pose + tips_distance and obj_pose_tail slots "
              "with estimator output",
              flush=True)

    # Parse mask_obs_slots once
    # ---- Restore the student's trained proprio layout ----------------------
    # A lean student was trained on a sliced obs; feeding it the env's full
    # vector would be a silent distribution shift (or a shape error).
    _student_mask_idx = list(_ckpt_peek.get("student_obs_mask_idx", []) or [])
    if _student_mask_idx:
        print(f"[EvalPC] SPARSE-REF ckpt: zeroing {len(_student_mask_idx)} proprio dims",
              flush=True)
    _student_keep_idx = list(_ckpt_peek.get("student_keep_idx", []) or [])
    _keep_t = None
    if _student_keep_idx:
        print(f"[EvalPC] LEAN STUDENT ckpt: slicing proprio to "
              f"{len(_student_keep_idx)}d, dropped "
              f"{list(_ckpt_peek.get('student_drop_slots', []) or [])}", flush=True)
        # The slot map is recorded during the first obs assembly; nothing has
        # forced one yet at this point in the script.
        _live = getattr(base_env, "actor_obs_slots", None)
        if _live is None:
            base_env._get_observations()
            _live = getattr(base_env, "actor_obs_slots", None)
        _saved = _ckpt_peek.get("student_obs_slots") or {}
        if _live and _saved and any(_live.get(k) != tuple(v) for k, v in _saved.items()):
            raise RuntimeError(
                f"actor obs layout changed since training: ckpt {_saved} vs env "
                f"{_live}. The saved keep-indices would select the wrong channels.")
        _keep_t = torch.as_tensor(_student_keep_idx, dtype=torch.long, device=device)

    # ---- Resolve observation offsets from the env, never by hand ----------
    # These used to be literals (410:413 for target_obj_pos, 417:422 for
    # tips_distance). They were written for a layout in which the tactile block
    # preceded target_obj_pose; the block order has since changed, so the
    # literals pointed 20 dims high — into obj_bps. Anything writing through
    # them corrupted the shape encoding and left the pose untouched, silently.
    _slots_live = getattr(base_env, "actor_obs_slots", None)
    if _slots_live is None:
        base_env._get_observations()
        _slots_live = getattr(base_env, "actor_obs_slots", None)
    if _slots_live:
        _OBJ_POSE_LO = _slots_live["target_obj_pose"][0]          # pos 3 + quat 4
        _TIPS_LO, _TIPS_HI = _slots_live["tips_distance"]
        _TAIL_LO, _TAIL_HI = _slots_live.get("obj_pose_tail", (None, None))
        _REF_LO = _slots_live["ref_tracking"][0]
    else:
        _OBJ_POSE_LO = _TIPS_LO = _TIPS_HI = _TAIL_LO = _TAIL_HI = _REF_LO = None

    _mask_ranges = []
    if args_cli.mask_obs_slots:
        for part in args_cli.mask_obs_slots.split(","):
            lo, hi = part.strip().split(":")
            _mask_ranges.append((int(lo), int(hi)))
        print(f"[EvalPC] OBS MASK ACTIVE: zeroing slots {_mask_ranges}", flush=True)

    # Obj-shift correction setup
    _obj_shift = args_cli.obj_shift_correction
    if _obj_shift != "none":
        print(f"[EvalPC] OBJ SHIFT CORRECTION: '{_obj_shift}' "
              f"(shift K=1 wrist+joints by δ = curr_obj - demo_K1_obj)", flush=True)
        if _obj_shift == "estimator" and _est is None:
            raise RuntimeError("--obj_shift_correction estimator requires --stage1_ckpt")

    # Initial-frame δ for shift correction (constant per episode):
    # δ_init = obj_at_reset - demo_obj_at_init_frame.
    # We track per-env init frame and re-compute δ_init whenever an env resets.
    _shift_delta_init = torch.zeros((base_env.num_envs, 3), device=device)
    _shift_init_frame = torch.zeros(base_env.num_envs, dtype=torch.long, device=device)
    _shift_prev_progress = torch.full((base_env.num_envs,), -1, dtype=torch.long,
                                       device=device)
    _shift_initialized = torch.zeros(base_env.num_envs, dtype=torch.bool, device=device)

    @torch.no_grad()
    def _refresh_init_delta(reset_mask, pred_est_pos=None):
        """Compute and cache δ_init for envs that just reset (reset_mask is bool)."""
        if not reset_mask.any():
            return
        nE = base_env.num_envs
        # init_frame for this episode = current progress_buf (env stores frame index here)
        # demo_data["obj_trajectory"] indexed by (env_id, init_frame)
        ids = reset_mask.nonzero(as_tuple=False).flatten()
        init_idx = base_env.progress_buf[ids].clamp(min=0)
        demo_obj_T = base_env.demo_data["obj_trajectory"][ids, init_idx]
        demo_obj_pos = demo_obj_T[:, :3, 3]
        if _obj_shift == "oracle":
            cur_obj_pos = (base_env.object.data.root_pos_w[ids]
                           - base_env.scene.env_origins[ids])
        else:
            cur_obj_pos = pred_est_pos[ids] if pred_est_pos is not None else torch.zeros_like(demo_obj_pos)
        _shift_delta_init[ids] = cur_obj_pos - demo_obj_pos
        _shift_init_frame[ids] = init_idx

    def _apply_obj_shift(obs_t, delta):
        """In-place shift of the K=1 target slots by delta_xyz.

        Shifts delta_wrist_pos, the 32 per-finger delta_joints_pos, the target
        object position and the pose-observation tail. Rotation is not
        corrected. Offsets come from the env's slot map, not from literals.
        """
        obs_t[:, _REF_LO:_REF_LO + 3] += delta                  # delta_wrist_pos
        # delta_joints_pos sits 23 dims into ref_tracking (7 wrist blocks) and
        # is (B, 96) = 32 fingers x 3.
        _jp_lo = _REF_LO + 23
        b = obs_t.shape[0]
        jp = obs_t[:, _jp_lo:_jp_lo + 96].view(b, 32, 3)
        jp += delta.unsqueeze(1)
        obs_t[:, _jp_lo:_jp_lo + 96] = jp.view(b, 96)
        obs_t[:, _OBJ_POSE_LO:_OBJ_POSE_LO + 3] += delta        # target_obj_pos
        if _TAIL_LO is not None:
            obs_t[:, _TAIL_LO:_TAIL_LO + 3] += delta            # pose-obs tail
        return obs_t

    def _make_inp(d):
        obs_in = d["policy"]
        if _mask_ranges:
            obs_in = obs_in.clone()
            for lo, hi in _mask_ranges:
                obs_in[:, lo:hi] = 0.0
        if _est is not None:
            with torch.no_grad():
                tf = d["tactile_force"]
                if tf.shape[-1] != _expected_fdim:
                    if tf.shape[-1] < _expected_fdim:
                        pad = torch.zeros(tf.shape[0], tf.shape[1],
                                          _expected_fdim - tf.shape[-1],
                                          device=tf.device, dtype=tf.dtype)
                        tf = torch.cat([tf, pad], dim=-1)
                    else:
                        tf = tf[..., :_expected_fdim]
                pc_feat = _encoder(d["scene_pc"], d["scene_mask"], d["hand_pc"],
                                   d["tactile_pc"], tf, d.get("tactile_mask"))
                proprio = extract_proprio_subset(base_env)
                goal = extract_goal(base_env, 32)
                pred = _est.predict_aligned(pc_feat, proprio, goal[..., 3:7])
                # ---- Planner-driven K=1 target replacement (ref_tracking) ----
                if _pln is not None:
                    if _is_flow_planner:
                        # Flow planner samples (B, K, per_frame_dim); de-normalize.
                        s = _pln.sample(goal, pc_feat, proprio, num_steps=1)
                        s_flat = s.reshape(s.shape[0], -1)
                        s_flat = s_flat * _target_std + _target_mean
                        plan_out = s_flat
                    else:
                        plan_out = _pln(goal, pc_feat, proprio)   # (B, 1055)
                    # First 211d = K=1 frame: wp(3)+wv(3)+wq(3)+wang(3)+jp(96)+jv(96)+op(3)+oq(4)
                    pf = plan_out[:, :211]
                    base_env.set_planner_targets({
                        "wrist_pos":     pf[:, 0:3],
                        "wrist_vel":     pf[:, 3:6],
                        "wrist_rot":     pf[:, 6:9],
                        "wrist_ang_vel": pf[:, 9:12],
                        "joints_pos":    pf[:, 12:108],
                        "joints_vel":    pf[:, 108:204],
                    })
                    # Note: obs_in (d["policy"]) was computed BEFORE we set planner_targets.
                    # The next compute_observations() (after env.step) will see them.
                    # For the CURRENT step, replace target_obj_pos+quat manually:
                    obs_in = obs_in.clone()
                    obs_in[:, _OBJ_POSE_LO:_OBJ_POSE_LO + 3] = pf[:, 204:207]
                    obs_in[:, _OBJ_POSE_LO + 3:_OBJ_POSE_LO + 7] = pf[:, 207:211]
                else:
                    obs_in = obs_in.clone()
                    obs_in[:, _OBJ_POSE_LO:_OBJ_POSE_LO + 3] = pred["obj_pos"]
                    obs_in[:, _OBJ_POSE_LO + 3:_OBJ_POSE_LO + 7] = pred["obj_quat"]
                # Estimator fills tips + PoseObs tail regardless of planner
                obs_in[:, _TIPS_LO:_TIPS_HI] = pred["tips_distance"]
                pp = pred["obj_pos"]; pq = pred["obj_quat"]
                if args_cli.estimator_inject_pose_noise:
                    pp = pp + torch.randn_like(pp) * _POS_SIGMA
                    pq = pq + torch.randn_like(pq) * _ROT_SIGMA
                    pq = pq / (pq.norm(dim=-1, keepdim=True) + 1e-8)
                obs_in[:, _TAIL_LO:_TAIL_LO + 3] = pp
                obs_in[:, _TAIL_LO + 3:_TAIL_HI] = pq
        # Obj-shift correction (use INIT-FRAME δ, constant per episode).
        if _obj_shift != "none":
            with torch.no_grad():
                cur_prog = base_env.progress_buf
                # Reset detection: progress_buf went DOWN compared to prev step
                # (env reset and re-sampled an init_frame), OR first call (prev=-1).
                need_refresh = (cur_prog < _shift_prev_progress) | (~_shift_initialized)
                if need_refresh.any():
                    if _obj_shift == "estimator":
                        tf2 = d["tactile_force"]
                        if tf2.shape[-1] != _expected_fdim:
                            if tf2.shape[-1] < _expected_fdim:
                                pad = torch.zeros(tf2.shape[0], tf2.shape[1],
                                                  _expected_fdim - tf2.shape[-1],
                                                  device=tf2.device, dtype=tf2.dtype)
                                tf2 = torch.cat([tf2, pad], dim=-1)
                            else:
                                tf2 = tf2[..., :_expected_fdim]
                        pc_feat2 = _encoder(d["scene_pc"], d["scene_mask"],
                                            d["hand_pc"], d["tactile_pc"], tf2,
                                            d.get("tactile_mask"))
                        proprio2 = extract_proprio_subset(base_env)
                        goal2 = extract_goal(base_env, 32)
                        pred2 = _est.predict_aligned(pc_feat2, proprio2, goal2[..., 3:7])
                        _refresh_init_delta(need_refresh, pred_est_pos=pred2["obj_pos"])
                    else:
                        _refresh_init_delta(need_refresh)
                # Mark all envs as initialized (refresh either just updated their δ
                # or they keep using the prev one for in-episode steps).
                _shift_initialized[:] = True
                _shift_prev_progress[:] = cur_prog
                # Apply cached δ_init (constant for whole episode) to obs
                if obs_in is d["policy"]:
                    obs_in = obs_in.clone()
                _apply_obj_shift(obs_in, _shift_delta_init)
        # Reduce to the student's trained layout LAST: mask indices refer to the
        # env's original obs, slicing changes the width.
        if _student_mask_idx:
            if obs_in is d["policy"]:
                obs_in = obs_in.clone()
            obs_in[:, _student_mask_idx] = 0.0
        if _keep_t is not None:
            obs_in = obs_in[:, _keep_t]
        return {"obs": obs_in, "scene_pc": d["scene_pc"],
                "scene_mask": d["scene_mask"], "hand_pc": d["hand_pc"],
                "tactile_pc": d["tactile_pc"], "tactile_force": d["tactile_force"]}

    def _obj_pos_now():
        op = getattr(base_env, "object_pos", None)
        if op is None:
            return torch.zeros((args_cli.num_envs, 3), device=device)
        return op.detach().clone()

    obs, _ = env_raw.reset()
    per_env_init = base_env.progress_buf.clone()
    per_env_t0 = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
    per_env_min_final = torch.full((args_cli.num_envs,), float("inf"), device=device)
    per_env_min_final_rot = torch.full((args_cli.num_envs,), float("inf"), device=device)
    per_env_fail_causes = [set() for _ in range(args_cli.num_envs)]
    per_env_obj_start = _obj_pos_now()
    per_env_last_final_dist = torch.full((args_cli.num_envs,), -1.0, device=device)
    prev_final_dist = torch.full((args_cli.num_envs,), -1.0, device=device)
    prev_final_rot = torch.full((args_cli.num_envs,), -1.0, device=device)
    per_env_last_final_rot = torch.full((args_cli.num_envs,), -1.0, device=device)
    per_env_traj_pos = [[] for _ in range(args_cli.num_envs)]
    per_env_traj_rot = [[] for _ in range(args_cli.num_envs)]
    prev_obj_pos = _obj_pos_now()
    records: list = []
    fail_keys = ["fail/hand_tracking", "fail/obj_pos_drift", "fail/obj_rot_drift",
                 "fail/premature_contact", "fail/arm_below_table",
                 "fail/error_buf_velocity_explosion"]
    step_counter = 0
    t_start = time.time()

    # ---- Per-demo quota ----------------------------------------------------
    # Collecting the first N episodes to finish biases the aggregate: success
    # terminates an episode earlier than failure, so whichever demo the policy
    # handles best contributes the most episodes. Collect a fixed share each.
    _demo_universe = sorted({_env_demo(i) for i in range(args_cli.num_envs)})
    if args_cli.per_demo_quota is None:
        _demo_quota = -(-args_cli.max_episodes // max(1, len(_demo_universe)))
    else:
        _demo_quota = int(args_cli.per_demo_quota)
    _per_demo_counts = collections.Counter()
    if _demo_quota > 0:
        print(f"[EvalPC] per-demo quota = {_demo_quota} x {len(_demo_universe)} demos "
              f"= {_demo_quota * len(_demo_universe)} episodes", flush=True)
    else:
        print("[EvalPC] per-demo quota DISABLED: first-to-finish collection "
              "(aggregate is biased toward the easier demos)", flush=True)

    print(f"[EvalPC] running until {args_cli.max_episodes} eps or "
          f"{args_cli.max_steps} steps", flush=True)

    def _collection_done() -> bool:
        if _demo_quota > 0:
            return all(_per_demo_counts[d] >= _demo_quota for d in _demo_universe)
        return len(records) >= args_cli.max_episodes

    while (step_counter < args_cli.max_steps
           and not _collection_done()
           and simulation_app.is_running()):
        with torch.no_grad():
            action = model.act_inference(_make_inp(obs))
            action = torch.clamp(action, -1.0, 1.0)
        obs, _r, done, _trunc, _info = env_raw.step(action)
        step_counter += 1
        cur_obj_pos = _obj_pos_now()

        rd = getattr(base_env, "reward_dict", None)
        if rd is not None:
            if "diag/final_pos_dist" in rd:
                cur_dist = rd["diag/final_pos_dist"]
                per_env_min_final = torch.minimum(per_env_min_final, cur_dist)
                per_env_last_final_dist = cur_dist
            if "diag/final_rot_angle" in rd:
                cur_rot = rd["diag/final_rot_angle"]  # radians
                per_env_min_final_rot = torch.minimum(per_env_min_final_rot, cur_rot)
                per_env_last_final_rot = cur_rot
            # Trajectory tracking: per-frame obj-vs-demo-current distance.
            if "diag/obj_pos_dist" in rd:
                _pd = rd["diag/obj_pos_dist"].detach().cpu().tolist()
                for _i in range(args_cli.num_envs):
                    per_env_traj_pos[_i].append(float(_pd[_i]))
            if "diag/obj_rot_angle" in rd:
                import math as _math
                _rd_arr = rd["diag/obj_rot_angle"].detach().cpu().tolist()
                for _i in range(args_cli.num_envs):
                    per_env_traj_rot[_i].append(_math.degrees(float(_rd_arr[_i])))
            for k in fail_keys:
                if k in rd:
                    fired = (rd[k] > 0.5).nonzero(as_tuple=False).flatten()
                    for env_id in fired.tolist():
                        per_env_fail_causes[env_id].add(k.split("/")[-1])

        done_t = done if torch.is_tensor(done) else torch.tensor(done, device=device)
        if done_t.any():
            done_ids = done_t.nonzero(as_tuple=False).flatten().tolist()
            for env_id in done_ids:
                survival_len = step_counter - int(per_env_t0[env_id].item())
                if survival_len < 5:
                    per_env_init[env_id] = base_env.progress_buf[env_id]
                    per_env_t0[env_id] = step_counter
                    per_env_min_final[env_id] = float("inf")
                    per_env_min_final_rot[env_id] = float("inf")
                    per_env_traj_pos[env_id] = []
                    per_env_traj_rot[env_id] = []
                    per_env_fail_causes[env_id] = set()
                    per_env_obj_start[env_id] = cur_obj_pos[env_id]
                    continue
                min_dist = float(per_env_min_final[env_id].item())
                success = (min_dist < args_cli.success_dist) and \
                          (not torch.isinf(per_env_min_final[env_id]).item())
                end_dist = float(prev_final_dist[env_id].item())
                import math
                min_rot = float(per_env_min_final_rot[env_id].item())
                end_rot = float(prev_final_rot[env_id].item())
                _tp = per_env_traj_pos[env_id]
                _tr = per_env_traj_rot[env_id]
                _max_p = max(_tp) if _tp else -1.0
                _mean_p = sum(_tp)/len(_tp) if _tp else -1.0
                _max_r = max(_tr) if _tr else -1.0
                _mean_r = sum(_tr)/len(_tr) if _tr else -1.0
                rec = EpisodeRecord(
                    env_id=env_id,
                    demo_idx=_env_demo(env_id),
                    init_frame=int(per_env_init[env_id].item()),
                    survival_len=survival_len,
                    succeeded=bool(success),
                    min_final_dist=min_dist if min_dist != float("inf") else -1.0,
                    end_final_dist=end_dist if end_dist >= 0 else -1.0,
                    min_final_rot_deg=math.degrees(min_rot) if min_rot != float("inf") else -1.0,
                    end_final_rot_deg=math.degrees(end_rot) if end_rot >= 0 else -1.0,
                    max_traj_pos_dist=_max_p,
                    mean_traj_pos_dist=_mean_p,
                    max_traj_rot_deg=_max_r,
                    mean_traj_rot_deg=_mean_r,
                    traj_pos_dist=([round(x, 4) for x in _tp] if args_cli.save_traj else []),
                    traj_rot_deg=([round(x, 2) for x in _tr] if args_cli.save_traj else []),
                    fail_causes=sorted(list(per_env_fail_causes[env_id])),
                    obj_start=per_env_obj_start[env_id].tolist(),
                    obj_end=prev_obj_pos[env_id].tolist(),
                )
                # Drop episodes from a demo that already filled its share, so
                # the aggregate weights every demo equally.
                if _demo_quota > 0 and _per_demo_counts[rec.demo_idx] >= _demo_quota:
                    pass
                else:
                    records.append(rec)
                    _per_demo_counts[rec.demo_idx] += 1
                per_env_init[env_id] = base_env.progress_buf[env_id]
                per_env_t0[env_id] = step_counter
                per_env_min_final[env_id] = float("inf")
                per_env_min_final_rot[env_id] = float("inf")
                per_env_traj_pos[env_id] = []
                per_env_traj_rot[env_id] = []
                per_env_fail_causes[env_id] = set()
                per_env_obj_start[env_id] = cur_obj_pos[env_id]
                if _demo_quota > 0:
                    if all(_per_demo_counts[d] >= _demo_quota for d in _demo_universe):
                        break
                elif len(records) >= args_cli.max_episodes:
                    break

        if step_counter % 100 == 0:
            ok = sum(1 for r in records if r.succeeded)
            pct = 100.0 * ok / max(1, len(records))
            _lag = ""
            if _demo_quota > 0 and _demo_universe:
                _d = min(_demo_universe, key=lambda d: _per_demo_counts[d])
                _lag = (f" slowest={_d.split('/')[-1]}"
                        f" {_per_demo_counts[_d]}/{_demo_quota}")
            print(f"  step={step_counter:5d} eps={len(records):5d} "
                  f"env_succ(5cm)={ok}/{len(records)} ({pct:.1f}%){_lag} "
                  f"elapsed={time.time()-t_start:.0f}s", flush=True)

        prev_obj_pos = cur_obj_pos
        prev_final_dist = per_env_last_final_dist.clone()
        prev_final_rot = per_env_last_final_rot.clone()

    elapsed = time.time() - t_start

    out_records = os.path.join(args_cli.out_dir, "records.json")
    with open(out_records, "w") as f:
        json.dump({"single": [asdict(r) for r in records]}, f, indent=2)
    print(f"[EvalPC] saved {len(records)} records → {out_records}", flush=True)

    # ---- strictN: the metric the protocol says to report -------------------
    # An episode counts only if the object finished within N cm of the demo's
    # final pose, AND object-position drift never fired, AND the episode was not
    # a bad init. Deliberately excludes ONLY obj_pos_drift, not the other failure
    # causes — ORing them all in would silently change what the number means.
    # See docs/EVAL.md §strict3.
    _BAD_INIT_SURVIVAL = 5

    def _strict(recs, cm):
        kept = [r for r in recs if r.survival_len > _BAD_INIT_SURVIVAL]
        if not kept:
            return 0.0, 0, 0
        ok = sum(1 for r in kept
                 if 0.0 <= r.end_final_dist < cm / 100.0
                 and "fail/obj_pos_drift" not in r.fail_causes)
        return ok / len(kept), ok, len(kept)

    _strict_rates = {}
    for _cm in (2, 3, 5):
        rate, ok, n = _strict(records, _cm)
        _strict_rates[f"strict{_cm}"] = {"rate": rate, "successes": ok, "episodes": n}
    _n_bad_init = sum(1 for r in records if r.survival_len <= _BAD_INIT_SURVIVAL)

    print("[EvalPC] strict success (end_final_dist < N cm, no obj_pos_drift, "
          f"bad inits excluded: {_n_bad_init}/{len(records)}):", flush=True)
    for _k, _v in _strict_rates.items():
        print(f"    {_k:8s} {_v['successes']:4d}/{_v['episodes']:<4d} "
              f"({100 * _v['rate']:5.1f}%)", flush=True)

    _by_demo = {}
    for d in sorted({r.demo_idx for r in records}):
        _rs = [r for r in records if r.demo_idx == d]
        _ok = sum(1 for r in _rs if r.succeeded)
        _s3, _s3ok, _s3n = _strict(_rs, 3)
        _by_demo[d] = {"episodes": len(_rs), "succeeded": _ok,
                       "success_rate": _ok / max(1, len(_rs)),
                       "strict3": _s3, "strict3_successes": _s3ok,
                       "strict3_episodes": _s3n}
    _rates = [v["success_rate"] for v in _by_demo.values()]
    _macro = sum(_rates) / len(_rates) if _rates else 0.0
    _micro = sum(1 for r in records if r.succeeded) / max(1, len(records))
    print("[EvalPC] per-demo (env-internal | strict3):", flush=True)
    for d, v in _by_demo.items():
        print(f"    {d.split('/')[-1]:24s} "
              f"{v['succeeded']:4d}/{v['episodes']:<4d} ({100*v['success_rate']:5.1f}%)  |  "
              f"{v['strict3_successes']:4d}/{v['strict3_episodes']:<4d} "
              f"({100*v['strict3']:5.1f}%)", flush=True)
    print(f"[EvalPC] success  macro (demo-averaged) = {100*_macro:.1f}%   "
          f"micro (episode-weighted) = {100*_micro:.1f}%", flush=True)

    summary = {
        "ckpt": args_cli.load_path, "label": args_cli.label,
        "per_demo_quota": _demo_quota,
        # The headline number. docs/EVAL.md says to report strict3, not the
        # env-internal reach-end rate below.
        "strict": _strict_rates,
        "bad_init_excluded": _n_bad_init,
        "success_rate_per_demo": _by_demo,
        "success_rate_macro": _macro,
        "success_rate_micro": _micro,
        "arch": arch_label, "task": args_cli.task,
        "num_envs": args_cli.num_envs, "max_episodes": args_cli.max_episodes,
        "actual_episodes": len(records), "steps": step_counter,
        "elapsed_sec": elapsed,
    }
    with open(os.path.join(args_cli.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[EvalPC] done: {len(records)} eps in {elapsed:.1f}s", flush=True)
    env_raw.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
