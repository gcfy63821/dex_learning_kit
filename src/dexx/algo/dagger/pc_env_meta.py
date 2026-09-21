"""Env-side point-cloud transform metadata — persisted with the checkpoint.

The PointCloud student's *input distribution* is shaped by a handful of
transforms that live in `env_cfg`, not in the model weights:

    pc_force_scale          tactile_force is divided by this before the encoder
    pc_tactile_force_gate   below-threshold tactile points are gated out
    pc_tactile_gate_mode    how they are gated ("zero" | "mask")
    pc_force_repr           "scalar" | "binary" numeric form of the force
    pc_ablate_tactile_pc    drop the tactile block from the unified cloud
    pc_ablate_tactile_force zero the tactile force channel
    pc_tactile_use_vec3     3D force vector instead of scalar magnitude

Before this module these were **train-time CLI args only**: they were never
written into the checkpoint and never restored at eval / play / deploy, which
default them to `1.0 / 0.0 / "zero" / "scalar" / False`. A student trained with
`--pc_force_scale 10 --pc_tactile_force_gate 0.05 --pc_tactile_gate_mode mask`
was therefore evaluated on forces **10x larger than training** with **every**
tactile point un-gated, and a `--pc_force_repr binary` student was fed
continuous magnitudes. Silent, and it moves success rate.

Keeping the collect/apply pair here means trainer, eval, play and deploy read
one definition and cannot drift apart again.

Checkpoint layout: a single nested dict under `"pc_env_meta"` (DAgger, top
level) or `cfg.pc_env_meta` (PPO). Checkpoints written before this module have
neither — `read_pc_env_meta` returns `{}` and the caller falls back to the
documented defaults, i.e. exactly the legacy behaviour.
"""
from __future__ import annotations

from typing import Any

# key -> default. The default must equal the env_cfg default, so that a ckpt
# without metadata reproduces legacy behaviour exactly.
#
# `enable_contact_force` / `enable_tactile` gate the PROPRIO tactile tail
# (5d per-finger force + 15d contact positions), which is a second tactile
# pathway independent of the point cloud. They belong here for the same reason
# as the pc_* keys — they change the student's input distribution and used to
# be train-time-only. Worse, `eval_dagger_pc.py` never even defined
# `--no_contact_force`, and because it parses with `parse_known_args()` the
# flag was silently swallowed: the 2026-05-27 scene_force_ablation's R2/R4
# cells trained WITHOUT proprio force but were evaluated WITH it.
PC_ENV_META_DEFAULTS: dict[str, Any] = {
    "pc_force_scale": 1.0,
    "pc_tactile_force_gate": 0.0,
    "pc_tactile_gate_mode": "zero",
    "pc_force_repr": "scalar",
    "pc_ablate_tactile_pc": False,
    "pc_ablate_tactile_force": False,
    "pc_tactile_use_vec3": False,
    "enable_contact_force": True,
    "enable_tactile": True,
}


def collect_pc_env_meta(env_cfg) -> dict[str, Any]:
    """Snapshot the env-side PC transforms into a plain dict for the ckpt.

    Call AFTER every CLI override has been applied to `env_cfg`, so the
    snapshot reflects what the student actually trained on.
    """
    return {
        key: getattr(env_cfg, key, default)
        for key, default in PC_ENV_META_DEFAULTS.items()
    }


def read_pc_env_meta(ckpt: dict) -> dict[str, Any]:
    """Pull the metadata out of a DAgger or PPO checkpoint.

    Returns `{}` for legacy checkpoints that predate this module.
    """
    meta = ckpt.get("pc_env_meta", None)
    if meta is None and "cfg" in ckpt:
        meta = getattr(ckpt["cfg"], "pc_env_meta", None)
    return dict(meta) if meta else {}


def apply_pc_env_meta(
    env_cfg,
    ckpt: dict,
    overrides: dict[str, Any] | None = None,
    tag: str = "PC",
) -> dict[str, Any]:
    """Restore the ckpt's env-side PC transforms onto `env_cfg`.

    Precedence: explicit `overrides` (CLI, value not None) > ckpt metadata >
    `PC_ENV_META_DEFAULTS`. Returns the resolved dict.

    Also keeps `pc_tactile_feature_dim` consistent with `pc_tactile_use_vec3`,
    since the encoder's per-point input dim depends on it.
    """
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    meta = read_pc_env_meta(ckpt)

    if not meta:
        print(
            f"[{tag}] ckpt has no 'pc_env_meta' (trained before it was persisted) — "
            f"falling back to defaults {PC_ENV_META_DEFAULTS}. If this policy was "
            f"trained with non-default --pc_force_scale / --pc_tactile_force_gate / "
            f"--pc_force_repr, pass them explicitly or the input distribution will "
            f"NOT match training.",
            flush=True,
        )

    resolved: dict[str, Any] = {}
    for key, default in PC_ENV_META_DEFAULTS.items():
        if key in overrides:
            value, source = overrides[key], "cli"
        elif key in meta:
            value, source = meta[key], "ckpt"
        else:
            value, source = default, "default"
        resolved[key] = value
        if getattr(env_cfg, key, default) != value or source == "cli":
            print(f"[{tag}] env_cfg.{key} = {value!r} (from {source})", flush=True)
        setattr(env_cfg, key, value)

    # `enable_tactile=False` subsumes the force gate — keep them consistent so
    # a ckpt trained with all tactile off cannot be evaluated with force on.
    if not resolved["enable_tactile"] and resolved["enable_contact_force"]:
        print(f"[{tag}] enable_tactile=False → forcing enable_contact_force=False",
              flush=True)
        resolved["enable_contact_force"] = False
        env_cfg.enable_contact_force = False

    # The encoder's per-point input dim depends on the force feature width;
    # keep it in lockstep with the vec3 switch.
    feat_dim = 3 if resolved["pc_tactile_use_vec3"] else 1
    if getattr(env_cfg, "pc_tactile_feature_dim", 1) != feat_dim:
        print(
            f"[{tag}] env_cfg.pc_tactile_feature_dim = {feat_dim} "
            f"(pc_tactile_use_vec3={resolved['pc_tactile_use_vec3']})",
            flush=True,
        )
        env_cfg.pc_tactile_feature_dim = feat_dim

    return resolved


# Hand-body subsets by point count. MUST match train_dagger_pc.py's
# --hand_body_subset choices: the student's encoder width is fixed at train
# time, so producing a different number of hand points makes the env's point
# tensor un-concatenable with the model's (a "sizes must match" RuntimeError,
# or worse, a silent distribution shift).
HAND_BODY_SUBSETS = {
    5: ["thumb_fingertip", "index_fingertip", "middle_fingertip",
        "ring_fingertip", "pinky_fingertip"],
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


def align_pc_dims_to_ckpt(env_cfg, ckpt: dict, tag: str = "PC") -> None:
    """Make the env emit exactly the point counts / tactile width the student was trained with.

    Call this BEFORE `gym.make`, and before/alongside `apply_pc_env_meta`. Every
    entry point that loads a PointCloud student needs it -- `eval.py` and
    `play.py` each grew their own copy of this logic, drifted, and `play.py`
    ended up unable to run any checkpoint whose hand-point count differed from
    the env default.
    """
    # PPO ckpts nest the dims under .cfg; DAgger ckpts keep them top-level.
    dim_src = ckpt
    if "n_hand" not in ckpt and "cfg" in ckpt:
        ppo_cfg = ckpt["cfg"]
        dim_src = {k: getattr(ppo_cfg, k) for k in ("n_scene", "n_hand", "n_tactile")
                   if hasattr(ppo_cfg, k)}
        if dim_src:
            print(f"[{tag}] PPO ckpt: read dims from ckpt['cfg']: {dim_src}", flush=True)

    for k_ckpt, k_env in (("n_scene", "pc_num_scene_points"),
                          ("n_hand", "pc_num_hand_points"),
                          ("n_tactile", "pc_num_tactile_points")):
        if k_ckpt in dim_src and hasattr(env_cfg, k_env):
            v_ckpt, v_env = int(dim_src[k_ckpt]), int(getattr(env_cfg, k_env))
            if v_ckpt != v_env:
                print(f"[{tag}] aligning env.{k_env}: {v_env} -> {v_ckpt} (from ckpt)", flush=True)
                setattr(env_cfg, k_env, v_ckpt)

    # Legacy fallback: pre-`pc_env_meta` ckpts only record tactile_feat_dim.
    fdim = ckpt.get("tactile_feat_dim")
    if fdim is None and "cfg" in ckpt:
        fdim = getattr(ckpt["cfg"], "tactile_feat_dim", None)
    if fdim == 3 and not getattr(env_cfg, "pc_tactile_use_vec3", False):
        env_cfg.pc_tactile_use_vec3 = True
        env_cfg.pc_tactile_feature_dim = 3
        print(f"[{tag}] ckpt has tactile_feat_dim=3 -> auto-enable pc_tactile_use_vec3", flush=True)

    n_hand = int(getattr(env_cfg, "pc_num_hand_points", 11))
    if n_hand in HAND_BODY_SUBSETS:
        env_cfg.pc_hand_body_names = HAND_BODY_SUBSETS[n_hand]
        print(f"[{tag}] pc_hand_body_names set to {n_hand}-body subset", flush=True)
