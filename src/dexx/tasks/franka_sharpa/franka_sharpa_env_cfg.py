# Copyleft (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All lefts reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from dexx import deploy_config as _dcfg  # single source of truth for arm_base/table
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg, ImplicitActuatorCfg

from dexx.robot_constants import ARM_ARMATURE, ARM_FRICTION, hand_gain_dicts as _hand_gain_dicts
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from dexx.utils.modified_events import randomize_rigid_body_scale
from dexx.tasks.hand_imitation.dataset.oakink2_dataset_utils import oakink2_obj_scale, oakink2_obj_mass

from dexx.tasks.sharpa_VBTS.sensor_cfg.ray_caster_surface import SharpaVBTSCfg, SharpaVBTS
# dummy pattern (not used, but the base class needs it)
from isaaclab.sensors.ray_caster import patterns

# NOTE: dummy placeholder, never spawned — the real object is set at runtime from
# the demo dataset's `obj_urdf_path`. Paths point at the shipped example object so
# there is no dependency on any external/absolute path.
OBJECT_CFG_LIST = [
                sim_utils.UrdfFileCfg(
                    asset_path="data/robotool_batch/models/cube_small/cleaned_mesh_10000.urdf",
                    fix_base = False,
                    joint_drive=None,
                ),
                sim_utils.UrdfFileCfg(
                    asset_path="data/robotool_batch/models/cube_small/cleaned_mesh_10000.urdf",
                    fix_base = False,
                    joint_drive=None,
                ),
]

def get_workspace_root():
    """Get the workspace root directory (the dexx_release repo root).

    This function navigates up from the current file's directory to find
    the repo root (dexx_release). The file is located at:
    dexx_release/src/dexx/tasks/franka_sharpa/franka_sharpa_env_cfg.py
    so the repo root is 4 levels up (franka_sharpa -> tasks -> dexx -> src -> dexx_release).
    """
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up 4 levels: franka_sharpa -> tasks -> dexx -> src -> dexx_release
    workspace_root = os.path.join(current_file_dir, "..", "..", "..", "..")
    return os.path.normpath(workspace_root)


def franka_sharpa_urdf(side: str = "right") -> str:
    """Merged FR3 + Sharpa Wave articulation URDF for one hand side.

    Produced by ``scripts/build_merged_urdf.py`` from the two vendored public
    models (``assets/franka_fr3`` + ``assets/sharpa_wave``) and committed, so
    training needs no build step. Isaac Lab converts it to USD on first spawn.
    """
    path = os.path.join(get_workspace_root(), "assets", "generated",
                        f"fr3_with_{side}_sharpa_wave.urdf")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"merged URDF not found: {path}\n"
            f"Generate it with:  python scripts/build_merged_urdf.py --side {side}"
        )
    return path


# Self-collision pairs that MUST be filtered out, as undirected link-name suffix
# pairs. The palm sits directly against the proximal phalanges, and each of these
# pairs is in contact at rest; with self-collision on and no filter the fingers
# cannot close against the palm at all.
#
# This is not cosmetic. The legacy pre-converted robot USD authored exactly these
# (8 prims / 14 directed entries). URDF cannot express collision filtering and
# Isaac Lab's UrdfConverterCfg has no field for it, so they are re-applied to the
# converted USD here. Measured on the shipped teacher, cube_small @5cm:
# filtered 94.9% vs unfiltered 37.9% -- the failures are silent (no NaN, no
# warning), the hand simply never closes and the object is never moved.
SELF_COLLISION_FILTER_PAIRS = (
    ("hand_C_MC", "index_PP"),
    ("hand_C_MC", "middle_PP"),
    ("hand_C_MC", "ring_PP"),
    ("hand_C_MC", "pinky_PP"),
    ("hand_C_MC", "thumb_MC"),
    ("pinky_MC", "pinky_PP"),
    ("thumb_MC", "thumb_PP"),
)


def _apply_self_collision_filters(usd_path: str, side: str) -> int:
    """Author `physics:filteredPairs` on the converted robot USD. Idempotent."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()
    n = 0
    for a_suf, b_suf in SELF_COLLISION_FILTER_PAIRS:
        a = stage.GetPrimAtPath(root.GetPath().AppendChild(f"{side}_{a_suf}"))
        b = stage.GetPrimAtPath(root.GetPath().AppendChild(f"{side}_{b_suf}"))
        if not (a and a.IsValid() and b and b.IsValid()):
            raise RuntimeError(
                f"self-collision filter pair not found in {usd_path}: "
                f"{side}_{a_suf} <-> {side}_{b_suf}"
            )
        # Author both directions, as the legacy asset did.
        for src, dst in ((a, b), (b, a)):
            UsdPhysics.FilteredPairsAPI.Apply(src)
            rel = src.GetRelationship("physics:filteredPairs")
            targets = list(rel.GetTargets())
            if dst.GetPath() not in targets:
                rel.AddTarget(dst.GetPath())
                n += 1
    if n:
        stage.GetRootLayer().Save()
    return n


def urdf_source_hash(side: str = "right") -> str:
    """Content hash of the merged URDF, used to detect a stale shipped USD."""
    import hashlib

    with open(franka_sharpa_urdf(side), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def shipped_robot_usd(side: str = "right") -> str | None:
    """Path to the committed, pre-built robot USD for `side`, or None if absent.

    Warns (loudly, but does not fail) when the asset was built from a different
    URDF than the one currently in the tree -- a stale asset is exactly the kind
    of mismatch that produces a silently wrong simulation rather than an error.
    """
    d = os.path.join(get_workspace_root(), "assets", "robot", f"fr3_with_{side}_sharpa_wave")
    usd = os.path.join(d, f"fr3_with_{side}_sharpa_wave.usd")
    if not os.path.isfile(usd):
        return None

    stamp = os.path.join(d, ".source_hash")
    try:
        with open(stamp) as f:
            built_from = f.read().strip()
    except OSError:
        built_from = None
    if built_from != urdf_source_hash(side):
        print(
            f"[cfg] WARNING: {usd} was built from a different "
            f"{os.path.basename(franka_sharpa_urdf(side))} than the one on disk. "
            f"Re-run: python scripts/build_robot_usd.py --side {side}",
            flush=True,
        )
    return usd


def franka_sharpa_robot_usd(side: str = "right") -> str:
    """Resolve the robot USD to spawn: the committed asset, else convert on the fly.

    The committed asset already carries the self-collision filter pairs (which
    URDF cannot express and `UrdfConverterCfg` has no field for). The fallback
    conversion path reproduces them, so a checkout without the built asset still
    works -- it is just slower on first use and depends on the local Isaac Lab
    version producing the same output.
    """
    shipped = shipped_robot_usd(side)
    if shipped is not None:
        return shipped

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    print(f"[cfg] no built robot USD for '{side}'; converting the URDF "
          f"(run scripts/build_robot_usd.py to avoid this)", flush=True)
    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=franka_sharpa_urdf(side),
            usd_dir=usd_cache_dir(side),
            fix_base=True,
            # MUST stay False: merging would delete the `*_elastomer` (tactile
            # sensor prims) and `*_fingertip` (tracked bodies) links.
            merge_fixed_joints=False,
            collider_type="convex_hull",
            self_collision=True,
            # Zero here so the actuator cfgs are the single source of truth for
            # gains rather than silently inheriting URDF values.
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
        )
    )
    added = _apply_self_collision_filters(converter.usd_path, side)
    if added:
        print(f"[cfg] authored {added} self-collision filter targets -> {converter.usd_path}")
    return converter.usd_path


def usd_cache_dir(side: str = "right") -> str:
    """Where Isaac Lab caches the URDF -> USD conversion for this robot.

    Isaac Lab only reuses a converted USD when ``usd_dir`` is pinned: left at its
    default it picks a fresh ``/tmp/IsaacLab/usd_<timestamp>_<random>`` per
    process, so the ``.asset_hash`` check never hits and every run re-converts and
    re-writes ~25 MB. Override with ``DEXX_USD_CACHE``. The cache is shared, so
    warm it once (any single-env script) before launching parallel jobs.
    """
    root = os.environ.get(
        "DEXX_USD_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "dexx", "usd"),
    )
    return os.path.join(root, f"fr3_with_{side}_sharpa_wave")



@configclass
class FrankaSharpaEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    # Action space: arm control + hand joint control (22)
    # freeze_arm:              hand(22) = 22
    # use_pid_control:         pos_error(3) + rot_error_6d(6) + hand(22) = 31
    # use_joint_pos_control:   arm_joint_pos(7) + hand(22) = 29
    # use_joint_delta_control: arm_joint_delta(7) + hand(22) = 29
    # force/torque (default):  force(3) + torque(3) + hand(22) = 28
    action_space = 29  # Will be updated based on control mode
    # Observation space: proprioception + target states
    # Proprioception: q(22) + cos_q(22) + sin_q(22) + base_state(10) = 76
    # Target: depends on obs_future_length and number of joints
    # For obs_future_length=1: wrist states (3+3+3+4+4+3+3) + joint states (n_joints*3*3)
    # Actual dimension: 390 (calculated from runtime)
    observation_space = 583
    prop_hist_len = 30  # Required for ProprioAdapt: Conv1d needs at least 30 steps
    priv_info_dim = 40
    state_space = 0
    asymmetric_obs = True

    # Whether to include object BPS (128d static shape encoding) in actor obs.
    # Honored by FrankaSharpaForceEnv / FrankaSharpaForceCriticHorizonEnv
    # (and their deploy variants). When False, BPS is NOT appended to obs and
    # obs_dim is reduced by 128.
    # NOTE: under asymmetric_ac=True in the base force env, BPS is always
    # removed from actor regardless of this flag (treated as privileged).
    # In the critic-horizon variant, this flag solely controls actor BPS.
    enable_bps: bool = True

    hand_side: str = "right"
    robot_asset_override: str = None  # optional: path (abs or repo-relative) to a custom robot URDF (default is assets/generated/fr3_with_{side}_sharpa_wave.urdf)
    material_elastomer_ids: list = None  # collision-shape indices of the 5 fingertip elastomers for friction DR; None -> [27,28,30,32,33] (stock asset, 34 shapes). Re-calibrate after any asset change with tools/calibrate_elastomer_ids.py -- out-of-range ids are dropped silently.
    hand_file_name: str = "Right"  # Will be updated by update_cfg_for_hand_side()
    hand_name: str = "right"  # Will be updated by update_cfg_for_hand_side()
    
    # Keypoint tracking parameters
    obs_future_length: int = 1  # Number of future steps for target observations
    use_quat_rot: bool = False  # Whether to use quaternion rotation in action space
    use_pid_control: bool = False  # Whether to use Diff IK for wrist (action: pos_error(3)+rot_error(6)=9 dims)
    use_osc_control: bool = False  # Whether to use OSC (Operational Space Control) for arm (action: pos_error(3)+rot_error(6)=9 dims)
    actions_moving_average: float = 0.4  # Moving average coefficient for hand/wrist actions
    arm_actions_moving_average: float = 0.15  # Separate (more aggressive) EMA for arm joint_pos_des to match real Franka impedance ~50ms LP bandwidth; sim2real arm-shake mitigation
    use_joint_pos_control: bool = False  # Whether to use absolute joint position control for arm (action: joint_pos(7) dims)
    use_joint_delta_control: bool = True  # Whether to use joint delta control for arm (action: joint_delta(7) dims, recommended for sim2real)
    joint_delta_scale: float = 0.2  # was 0.1 @60Hz, doubled for 30Hz to maintain same arm speed
    freeze_arm: bool = False  # Freeze arm at reset position, action space = hand only (22)

    # Tracking reward mode: controls which reward terms are active
    # - wrist tracking: reward_eef_pos/rot/vel (world-frame wrist pose tracking)
    # - absolute hand tracking: reward_*_tip_pos (world-frame hand body tracking)
    # - relative hand tracking: reward_rel_*_tip (wrist-frame hand body tracking, decouples hand shape from wrist error)
    # All three can be independently enabled. For freeze_arm, use rel_hand only.
    use_wrist_tracking_reward: bool = True
    use_absolute_hand_tracking_reward: bool = True
    use_relative_hand_tracking_reward: bool = False

    # Tracking reference source. Controls which keys back `target_wrist_*` and
    # `target_joints_*` in `_build_data` (used by `_get_rewards` / obs).
    #   "auto"     — prefer retargeted `opt_*` when present in demo, fall back to MANO
    #   "retarget" — strict: require `opt_*`, raise if missing
    #   "mano"     — force raw MANO (`wrist_pos` / `mano_joints`), ignore `opt_*`
    # Default "auto" reproduces the collaborator's post-refactor behavior.
    reference_source: str = "auto"
    # Optional override for RoboToolBatch retarget pkl root. Empty keeps the
    # dataset default: data/retargeting/robotool_batch/mano2{dexhand}.
    robotool_batch_retarget_root: str = ""

    translation_scale: float = 0.02  # Scale for translation actions (±2cm, matching UWLab OSC scale)
    orientation_scale: float = 0.05  # Scale for orientation actions (±0.05 rad ≈ ±3°)

    # Gravity compensation in sim (real Franka has built-in gravity comp, sim PD does not)
    arm_gravity_compensation: bool = True  # Add gravity+coriolis compensation torques to arm PD

    # OSC controller parameters (only used when use_osc_control=True)
    osc_kp_xyz: float = 1500.0  # Position stiffness
    osc_kp_rot: float = 1500.0  # Rotation stiffness
    osc_damping_ratio: float = 1.0  # Damping ratio
    osc_partial_decoupling: bool = True  # Decouple translation/rotation inertia
    osc_nullspace_stiffness: float = 10.0  # Nullspace posture stiffness

    # Observation noise for sim2real (simulates sensor noise)
    obs_joint_pos_noise: float = 0.01  # Noise std for hand joint positions (rad, ~0.6°)
    obs_wrist_pos_noise: float = 0.002  # Noise std for wrist position (m, 2mm)
    obs_wrist_rot_noise: float = 0.017  # Noise std for wrist rotation (rad, ~1°)

    # Action delay for sim2real (simulates communication latency)
    action_delay_steps: int = 2  # Number of steps to delay actions (0=disabled, 2 steps @60Hz ≈ 33ms latency)
    # Action-delay Domain Randomization (sim2real). When randomize_action_delay=True,
    # each env gets its own delay sampled uniformly in [min, max] at reset time. The
    # FIFO buffer is always sized to `max` (upper bound). This covers the observation
    # that real-hardware lag varies 0-100ms across joints, so DR over the delay range
    # makes the policy robust to the actual per-run latency.
    randomize_action_delay: bool = True
    action_delay_min: int = 0          # inclusive, 0 = no delay for this env
    action_delay_max: int = 3          # inclusive, 3 steps @30Hz = 100ms

    # Tightening parameters (curriculum learning)
    tighten_method: str = "exp_decay"  # "None", "const", "linear_decay", "exp_decay", "cos"
    tighten_factor: float = 0.7  # Tightening factor (was 0.7, relaxed to let arm learn first)
    tighten_steps: int = 3000  # Number of steps for tightening (was 1000, slower ramp)
    
    # Reset parameters 
    random_state_init: bool = True  # Whether to randomly initialize state
    # When random_state_init is False, default is demo frame 0. Set this to start every reset at a fixed demo index (clamped per-env to seq_len-1). Used by dump_obs_sim / debug deploy start frame.
    fixed_reset_demo_frame: int | None = None
    rollout_state_init: bool = True  # Whether to initialize from rollout
    loop_trajectory: bool = False  # Whether to loop reference trajectory (reset only on max_episode_length or terminate)

    # Reverse curriculum on initialization: start from near-grasp frames, gradually expand to full trajectory.
    # Defaults tuned 2026-05-19 from V1 ablation (RECENT_CHANGES.md Week 13 §6):
    #   `start=0.3 + steps=15000` is the winning combo when combined with V1 reward boost.
    #   Together with `success_pos_weight=30, alpha_pos=15, window=10` (also retuned 2026-05-19),
    #   gives frame-0 reach success 56% (vs 34% at default curriculum).
    # Anti-patterns (from V2/V3 ablation):
    #   - `enabled=False` (uniform full demo) alone: no-op, reach success unchanged.
    #   - `enabled=False` + reward boost together: WORSE than baseline (reach 26.6% vs 34.4% F).
    init_curriculum_enabled: bool = True
    init_curriculum_method: str = "linear"  # "linear", "exp", "cos"
    init_curriculum_start: float = 0.3   # 2026-05-19: 0.0 → 0.3 (V1 ablation; sample from 30%+ initially)
    init_curriculum_end: float = 0.0     # Late training: sample from 0%+ (full approach)
    init_curriculum_steps: int = 15000   # 2026-05-19: 2000 → 15000 (V1 ablation; gentler ramp)

    # Adaptive initialization from rollout state buffer
    adaptive_init_enabled: bool = False  # Master switch
    adaptive_init_prob: float = 0.3  # Probability of sampling from buffer vs demo
    adaptive_init_buffer_size: int = 8192  # Max entries in ring buffer
    adaptive_init_warmup: int = 500  # Training steps before buffer is used
    adaptive_init_capture_top_k: float = 0.1  # Top fraction of per-step rewards to capture
    adaptive_init_min_progress: int = 10  # Minimum running_progress_buf before capture

    # Adaptive trajectory-fraction sampling — biases reset frames toward bins
    # where the policy currently fails most. Inspired by whole_body_tracking
    # commands.py adaptive bin sampler. Operates on normalized [0, 1) trajectory
    # fraction so it works with multi-data_idx envs (different per-env seq_len).
    #
    # Behavior:
    #   - During the first `warmup_steps` env steps, the sampler is FORCED uniform
    #     (equivalent to random_state_init baseline). Without this, the very
    #     first failures on a random policy collapse the pmf onto a few hard
    #     bins, the policy never sees easy frames, and reward tanks.
    #   - When `compose_with_curriculum=True` AND `init_curriculum_enabled=True`,
    #     bins below the curriculum's `earliest_frac` are masked out — adaptive
    #     samples by failure bias *inside* the curriculum window so you keep
    #     the easy-first warmup while still focusing on the hardest frames in
    #     that window.
    #   - When adaptive_sampling_enabled=True, REPLACES the plain curriculum /
    #     uniform random branches inside _reset_idx (still gated by random_state_init).
    adaptive_sampling_enabled: bool = False
    adaptive_sampling_bins: int = 50            # number of fractional bins along [0,1)
    adaptive_sampling_kernel_size: int = 3      # smoothing kernel size (non-causal)
    adaptive_sampling_lambda: float = 0.8       # geometric kernel decay
    # uniform_ratio: total uniform mass mixed into the pmf (split across bins).
    # 0.5 means a meaningful 50/50 floor against failure spikes — much more
    # forgiving than the reference's 0.1 which lets a single hot bin take ~100%
    # of probability mass once EMA stabilizes.
    adaptive_sampling_uniform_ratio: float = 0.5
    # alpha: EMA mixing factor for bin_failed_count.
    # 0.0005 → effective window ~2000 resets, slow enough that single bad
    # batches don't permanently bias the sampler.
    adaptive_sampling_alpha: float = 0.0005
    # warmup_steps: number of training env steps during which sampler stays
    # uniform (or follows curriculum window if enabled). After this, full
    # adaptive sampling kicks in. Should be at least 1× the typical episode
    # length × num_envs / batch_size to let policy collect signal.
    adaptive_sampling_warmup_steps: int = 1000
    # compose_with_curriculum: when True and init_curriculum_enabled=True, the
    # adaptive sampler restricts bin sampling to the curriculum's expanding
    # window. Best practice for early training stability.
    adaptive_sampling_compose_with_curriculum: bool = True

    # Approach reward shaping: guide hand toward object when far away
    approach_reward_weight: float = 2.0  # Weight for approach reward (active when hand far from object)
    approach_reward_scale: float = 5.0  # Exponential decay scale for distance

    # ------------------------------------------------------------------
    # Raycaster-based depth (FrankaSharpaVisualEnv)
    # ------------------------------------------------------------------
    # FrankaSharpaVisualEnv uses simple_raycaster.MultiMeshRaycaster to produce
    # its [N, H, W] depth tensor (z-depth, matching real-world depth cameras).
    # Hit clamps fed to MultiMeshRaycaster.raycast_fused.
    raycaster_min_dist: float = 0.01
    raycaster_max_dist: float = 5.0
    # Quadric decimation factors. 0.0 disables. Robot link meshes are already
    # simple, so leave at 0.0; object/scene meshes (cleaned_mesh_10000.obj) get
    # a small amount of simplification by default to keep BVH build/query cheap.
    raycaster_link_simplify_factor: float = 0.0
    raycaster_simplify_factor: float = 0.1
    # Pinhole intrinsics for the depth image.
    camera_focal_length: float = 21.77
    camera_horizontal_aperture: float = 36.0

    # Asymmetric Actor-Critic: actor sees only deployable obs, critic sees obs + priv_info
    asymmetric_ac: bool = True  # When True, removes privileged obs (tips_distance, BPS) from actor
    
    # Dexhand configuration
    dexhand: str = "sharpa"  # Hand type (used by DexHandFactory)
    # control
    decimation = 4  # was 2. Policy freq = 120/4 = 30Hz, matching demo data 30fps and deploy
    clip_obs = 5.0
    clip_actions = 1.0
    action_scale = 1
    torque_control = False
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=18388608, # 2**23
            gpu_max_rigid_patch_count=5*2**18,
            enable_ccd=True
        ),
    )
     # Place base at table edge: x=0.0 (left side), y=0.0 (center). z raised +1.7cm
     # from 0.415 -> 0.432 on 2026-07-16, then back to 0.415 on 2026-09-01 when the
     # robot was remounted level with the table. Matches the real arm-base height above
     # table. Keep in sync with retarget_arm_2stage_aug.py and pointcloud_deploy_env.py.
    arm_base_pos = _dcfg.ARM_BASE_POS   # (-0.1, 0.0, 0.415) — edit in dexx/deploy_config.py
    arm_base_rot = _dcfg.ARM_BASE_ROT   # identity quaternion
    arm_init_joint_pos: list = [0.24435, 0.17453, -0.13963, -2.14675, -1.78024, 1.83260, -0.05236]  # Will be updated by update_cfg_for_hand_side()

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            # Converted from the merged URDF and patched with the self-collision
            # filters; resolved per-side by update_cfg_for_hand_side().
            usd_path="",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.1,
                angular_damping=0.1,
                max_linear_velocity=1000.0,
                max_angular_velocity=64 / math.pi * 180.0,
                max_depenetration_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=arm_base_pos,
            rot=arm_base_rot,
            joint_pos={
                "fr3_joint1": arm_init_joint_pos[0],
                "fr3_joint2": arm_init_joint_pos[1],
                "fr3_joint3": arm_init_joint_pos[2],
                "fr3_joint4": arm_init_joint_pos[3],
                "fr3_joint5": arm_init_joint_pos[4],
                "fr3_joint6": arm_init_joint_pos[5],
                "fr3_joint7": arm_init_joint_pos[6],
                # Hand joints will be updated by update_cfg_for_hand_side()
            },
        ),
        actuators={
            # Arm joints: use ImplicitActuator
            # With arm_gravity_compensation=True, PhysX PD drives are zeroed at runtime
            # and replaced by manual PD + gravity + coriolis torques (mimics real Franka).
            # Gains are higher than real (K*3~4x) to compensate for sim's lower control
            # bandwidth (120Hz PD vs real Franka's 1kHz torque loop).
            # Real DexhandJointImpedanceController: K=[200,200,200,200,100,100,50], D=[20,20,20,20,10,10,5]
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint.*"],
            #     # ==== ACTIVE: uniform K=400 D=80 test set (2026-05-06) ====
            #     # Switched from per-joint tuned set (see BACKUP below) to a flat
            #     # K=400 D=80 baseline to evaluate training effect of a softer arm.
            
                # stiffness={
                #     "fr3_joint1": 400.0,
                #     "fr3_joint2": 400.0,
                #     "fr3_joint3": 400.0,
                #     "fr3_joint4": 400.0,
                #     "fr3_joint5": 400.0,
                #     "fr3_joint6": 400.0,
                #     "fr3_joint7": 400.0,
                # },
                # damping={
                #     "fr3_joint1": 80.0,
                #     "fr3_joint2": 80.0,
                #     "fr3_joint3": 80.0,
                #     "fr3_joint4": 80.0,
                #     "fr3_joint5": 80.0,
                #     "fr3_joint6": 80.0,
                #     "fr3_joint7": 80.0,
                # },
                # ==== BACKUP: per-joint tuned set (was active before 2026-05-06) ====
                # Tuned via step response comparison (system_id, 2026-03-31)
                # With gravity compensation ON. All joints @200ms within ±10% of real.
                
                # 2026-04-23 update: single-joint sin_j1/j2/j3 (motion-file) tests showed
                # sim LAGS real by ~40-50ms on proximal joints — symptom of sim being
                # over-damped (ζ≈1.34) relative to real (ζ≈0.3). Reduced j1-3 damping by
                # ~40% (targeting ζ≈0.8) to close the gap. j4-7 were already matching well
                # (lag 0-10ms, corr >0.998) and are left untouched.
                
                stiffness={
                    "fr3_joint1": 1600.0,
                    "fr3_joint2": 1600.0,
                    "fr3_joint3": 1200.0,
                    "fr3_joint4": 800.0,
                    "fr3_joint5": 500.0,
                    "fr3_joint6": 300.0,
                    "fr3_joint7": 150.0,
                },
                damping={
                    "fr3_joint1": 145.0,    # was 240  (ζ 1.34→0.81)
                    "fr3_joint2": 135.0,    # was 220  (ζ 1.37→0.84)
                    "fr3_joint3": 110.0,    # was 180  (ζ 1.50→0.92)
                    "fr3_joint4": 100.0,    # unchanged — j4 already matches
                    "fr3_joint5": 50.0,     # unchanged
                    "fr3_joint6": 30.0,     # unchanged
                    "fr3_joint7": 15.0,     # unchanged
                },
                # Rotor inertia / joint friction. Previously inherited from the
                # pre-converted USD; the URDF cannot express them, so they are
                # restored explicitly here. See dexx.robot_constants.
                armature=ARM_ARMATURE,
                friction=ARM_FRICTION,
            ),
            # Hand joints: use IdealPDActuator (will be updated by update_cfg_for_hand_side())
            # Hand gains/armature/friction come from HAND_GAINS, not from the
            # asset: a URDF-imported hand has armature=0, which lets the finger
            # joints tunnel through their PhysX limits in a single step.
            # update_cfg_for_hand_side() re-keys these for the chosen side.
            "hand_joints": IdealPDActuatorCfg(
                joint_names_expr=["right_.*"],  # Will be updated by update_cfg_for_hand_side()
                **_hand_gain_dicts("right"),
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    # contact_sensor, actuated_joint_names, and fingertip_body_names will be updated by update_cfg_for_hand_side()
    contact_sensor: list = []
    actuated_joint_names: list = []
    fingertip_body_names: list = []

    # table
    # Table dimensions:
    # - size: x=1.5, y=2.4, z=0.03  (was 1.0 x 1.6, enlarged 1.5x for more workspace)
    # - position: x=0.1, y=0, z=0.4 (unchanged; top surface still at z=0.415)
    # - fix_base_link = True -> kinematic_enabled=True
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 2.4, 0.03),  # x, y, z dimensions
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Fixed table (equivalent to fix_base_link=True)
                disable_gravity=True,
                enable_gyroscopic_forces=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),  # Mass doesn't matter for kinematic objects
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.1, 0.0, 0.4),  # x=-0.1, y=0, z=0.4
            rot=(1.0, 0.0, 0.0, 0.0)  # Identity quaternion
        ),
    )


    # POINTS_NPY_4F=os.path.join(get_workspace_root(), "dexx", "tasks", "sharpa_VBTS", "sensor_cfg", "ray_caster_surface", "ray_caster_surface_npy", "tactileSensor_map_4F_point.npy")
    # NORMALS_NPY_4F=os.path.join(get_workspace_root(), "dexx", "tasks", "sharpa_VBTS", "sensor_cfg", "ray_caster_surface", "ray_caster_surface_npy", "tactileSensor_map_4F_normal.npy")
    # POINTS_NPY_TH=os.path.join(get_workspace_root(), "dexx", "tasks", "sharpa_VBTS", "sensor_cfg", "ray_caster_surface", "ray_caster_surface_npy", "tactileSensor_map_TH_point.npy")
    # NORMALS_NPY_TH=os.path.join(get_workspace_root(), "dexx", "tasks", "sharpa_VBTS", "sensor_cfg", "ray_caster_surface", "ray_caster_surface_npy", "tactileSensor_map_TH_normal.npy")
    # # VBTS
    # vbts_sensor = [
    #     SharpaVBTSCfg(
    #         prim_path=f"/World/envs/env_.*/Robot/{hand_side}_thumb_elastomer",
    #         # mesh_prim_paths=["/World/envs/env_.*/object"], currently not support
    #         # get the mesh from env_0, and corresponding position in usd stage
    #         # mesh_prim_paths=["/World/envs/env_0/object"],
    #         # target_rigid_expr = "/World/envs/env_.*/object",
    #         mesh_prim_paths=["/World/envs/env_0/object/scan"],  # explicit ok for single env
    #         target_rigid_expr="/World/envs/env_.*/object/scan",
    #         # ------------------------------------------------------------
    #         update_period=0.0,  # set 0.0 if want every sim step
    #         # dummy pattern (not used by surface sensor, but base class requires it)
    #         pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.1, 0.1)),
    #         offset=SharpaVBTSCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    #         data_types=["distance_along_normal"],
    #         points_npy=POINTS_NPY_TH,
    #         normals_npy=NORMALS_NPY_TH,
    #         max_distance=0.02,
    #         # debug_vis=True,
    #         debug_vis=False,
    #         correction_scale=1e-3,
    #     ),
    # ]


    # maniptrans object:
    obj_id = "O02@0015@00019"
    obj_scale = oakink2_obj_scale.get(obj_id, 1.0)
    obj_mass = oakink2_obj_mass.get(obj_id, 0.02)
    # obj_scale = 1.0

    if obj_id in oakink2_obj_mass:
        obj_mass = oakink2_obj_mass[obj_id]
    else:
        obj_mass = 0.05

    object_cfg: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/object",
            # spawn=sim_utils.UsdFileCfg(
            spawn=sim_utils.UrdfFileCfg(
                asset_path=os.path.join(get_workspace_root(), "data", "OakInk-v2", "coacd_object_preview", "align_ds", obj_id, "scan.urdf"),
                fix_base = False,
                joint_drive=None,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    disable_gravity=False,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.0025,
                    max_depenetration_velocity=1000.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.002, 
                    rest_offset=0.0
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=obj_mass),
                scale=(obj_scale, obj_scale, obj_scale),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            # spawn=sim_utils.MultiAssetSpawnerCfg(
            #     assets_cfg=OBJECT_CFG_LIST,
            #     random_choice=False,
            #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
            #         kinematic_enabled=False,
            #         disable_gravity=False,
            #         enable_gyroscopic_forces=True,
            #     ),
            #     mass_props=sim_utils.MassPropertiesCfg(mass=obj_mass),
            #     collision_props=sim_utils.CollisionPropertiesCfg(
            #         collision_enabled=True,
            #         contact_offset=0.002, 
            #         rest_offset=0.0
            #     ),
            # ),
            # init_state=RigidObjectCfg.InitialStateCfg(
            #         pos=(0.0, 0.0, 0.0),
            #         rot=(1.0, 0.0, 0.0, 0.0),
            #     ),
        )
    
    


    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=1.2, replicate_physics=False)
    # event
    # events: EventCfg = EventCfg()
    # dataset_path
    # dataset_path = os.path.join(get_workspace_root(), "data", "imitation_left_sharpa.pkl")
    data_indices = ["925aa@1"]
    # reset
    reset_height_lower = 0.63
    reset_height_upper = 0.67
    reset_angle_diff = 45 / 180 * math.pi
    reset_random_quat = True
    # reward
    # primary reward
    rot_axis = (0, 0, 1)
    angvel_clip_min = -0.5
    angvel_clip_max = 0.5
    rotate_reward_scale = 2.5
    object_linvel_penalty_scale = -0.3
    pos_diff_penalty_scale = -0.4
    torque_penalty_scale = -0.1
    work_penalty_scale = -0.5
    # auxiliary reward
    rot_diff_clip_min = -0.025
    rot_diff_clip_max = 0.025
    object_pos_reward_scale = 0.001
    fingertip_mimic_penalty_scale = -0.0
    # fingertip_mimic_traj = 'cache/recorded_traj_50hz.npy'
    mimic_traj_step = 1 # to match env control freq (20Hz)
    mimic_traj_start_scope = [0, 200]
    contact_reward_scale = 0.2
    # grasp cache
    grasp_cache_path = 'cache/sharpa_grasp_linspace'
    # noise
    joint_noise_scale = 0.02
    # contact
    enable_tactile = True
    enable_contact_force = True   # 5d scalar contact force in obs
    binary_contact = False        # was True (deploy-only); False = continuous force for training
    enable_contact_pos = False
    disable_tactile_ids = []
    contact_smooth = 0.5
    contact_threshold = 0.2
    contact_latency = 0.005
    contact_sensor_noise = 0.01
    # Axis 2 ablation: force representation (see ablation_force/franka_sharpa_force_repr_env.py)
    contact_force_repr: str = "continuous"  # continuous|binary|log|normalized|binned|3dvec
    contact_force_max: float = 5.0          # divisor for 'normalized' repr
    contact_force_bins: tuple = (0.2, 1.0, 3.0)  # edges for 'binned' (→ 4 bins / finger)
    # Axis 3+4 ablation: temporal + spatial force encodings
    # (see ablation_force/franka_sharpa_force_encoding_env.py)
    force_encoding: str = "none"            # none|fft_mag|fft_mag_lowk|derivative|stacked_hist|
                                            # ema_multiscale|onset_event|force_at_tip|
                                            # weighted_tip_pos|force_vec_handframe|wrench_at_tip|
                                            # bilinear_fuse
    force_hist_len: int = 32                # FFT / history window (must be power of 2 ideally)
    fft_lowk: int = 4                       # # FFT bins kept for 'fft_mag_lowk'
    stack_k: int = 8                        # # frames for 'stacked_hist'
    ema_alphas: tuple = (0.1, 0.5, 0.9)     # EMA time-constants for 'ema_multiscale'
    onset_threshold: float = 0.2            # force threshold for 'onset_event'
    force_ref: float = 2.0                  # saturation point for 'weighted_tip_pos' sigmoid
    bilinear_proj_dim: int = 16             # output dim for 'bilinear_fuse'
    bilinear_seed: int = 0                  # seed for fixed random projection
    # Fingertip tactile ablations
    # (see ablation_force/franka_sharpa_fingertip_ablation_env.py)
    fingertip_tactile_mode: str = "baseline"       # baseline|binary|force5d|force3d
    fingertip_include_contact_center: bool = False # append 15d contact center position
    # Default False: center ablations expose the raw contact_center signal.
    # Set True via --env_cfg to run the masked-center control where 1b/2b/3b
    # all apply the same force-threshold mask to contact_center.
    fingertip_mask_contact_center_by_threshold: bool = False
    # Negative means reuse contact_threshold (default 0.2). Set to e.g. 0.05,
    # 0.5 from --env_cfg to sweep center-mask threshold independently later.
    fingertip_contact_center_mask_threshold: float = -1.0
    # Taxel-level tactile ablations
    # (see ablation_force/franka_sharpa_taxel_ablation_env.py)
    taxel_source: str = "vbts"               # vbts only
    taxel_tactile_mode: str = "abs"          # abs|force3d
    taxel_include_position: bool = True      # append taxel positions in hand frame
    taxel_count_per_finger: int = 64         # sampled taxels per fingertip mesh
    taxel_position_scale: float = 1.0e-3     # tactile OBJ vertices are stored in mm
    taxel_vbts_update_interval: int = 0      # VBTS stride in policy steps; 0 = legacy every sim substep
    taxel_thumb_obj_path: str = os.path.join(get_workspace_root(), "tactile_ha4_map", "tactileSensor_TH.obj")
    taxel_4f_obj_path: str = os.path.join(get_workspace_root(), "tactile_ha4_map", "tactileSensor_4F.obj")
    taxel_vbts_cache_dir: str = os.path.join(get_workspace_root(), "tactile_ha4_map", "vbts_cache")
    taxel_vbts_mesh_prim_path: str = "/World/envs/env_0/object/base"
    taxel_vbts_target_rigid_expr: str = "/World/envs/env_.*/object/base"
    taxel_vbts_max_distance: float = 0.02
    taxel_vbts_debug_vis: bool = False
    bind_multiasset_object_root_link: bool = False
    multiasset_object_root_link_name: str = "base"
    # contact domain randomization for sim2real
    contact_force_noise: float = 0.2        # Multiplicative noise std (±20% of force value)
    contact_pos_noise: float = 0.003        # Additive noise std (3mm)
    contact_dropout_prob: float = 0.05      # Per-finger probability of dropping tactile signal
    # align real
    dof_limits_scale = 0.9
    # Tighten hand joint limits to Sharpa HA4 real-hand measured reachable
    # range (cfg/Sharpa order). Needed for sim-real alignment of ring/pinky
    # MCP_AA etc. that have inter-finger mechanical coupling.
    use_real_hand_limits: bool = True
    current_coef = 0.7
    # randomize
    scale_range = [0.6, 0.9, 16]
    # events.rand_params(scale_range)
    randomize_pd_gains = True
    randomize_p_gain_scale_lower = 0.5
    randomize_p_gain_scale_upper = 2
    randomize_d_gain_scale_lower = 0.5
    randomize_d_gain_scale_upper = 2
    # Arm PD gain Domain Randomization (separate from hand's randomize_pd_gains,
    # which only touches hand joints). Single-joint sin_j1-j3 tests showed PD is
    # already well-matched after 2026-04-23 damping tuning; this DR handles
    # residual multi-joint coupling gap (~±5pp of chirp correlation) and real
    # hardware variability. ±20% is tight because per-joint PD is already close.
    randomize_arm_pd_gains: bool = True
    randomize_arm_p_scale_lower: float = 0.80   # widened ±10% → ±20% for sim2real shake mitigation
    randomize_arm_p_scale_upper: float = 1.20
    randomize_arm_d_scale_lower: float = 0.80
    randomize_arm_d_scale_upper: float = 1.20
    randomize_friction = True
    randomize_friction_scale_lower = 1.0
    randomize_friction_scale_upper = 2.5
    elastomer_base_friction = 0.8
    metal_base_friction = 0.1
    object_base_friction = 0.5
    randomize_com = True  # was False
    randomize_com_lower = -0.02  # was -0.01
    randomize_com_upper = 0.02   # was 0.01
    # Training-time object xy displacement, metres (uniform +-). Eval can widen
    # it per-run via env.set_eval_perturb_obj_xy; the env takes the larger.
    # ---- Polymetis arm backend (deploy) ----------------------------------
    # The real arm runs a joint-impedance controller; these are the gains it is
    # started with. Leave kq/kqd None to use the Polymetis defaults. Sim's
    # counterpart is robot_cfg.actuators["arm_joints"].stiffness/damping — the
    # two must describe the same controller or the policy meets a different
    # plant on hardware than the one it was trained against.
    polymetis_server_ip: str = "localhost"
    polymetis_state_port: int = _dcfg.POLYMETIS_STATE_PORT
    polymetis_cmd_port: int = _dcfg.POLYMETIS_CMD_PORT
    polymetis_kq: tuple | None = None
    polymetis_kqd: tuple | None = None

    randomize_obj_xy: float = 0.0
    randomize_mass = True  # was False
    randomize_mass_lower = 0.01
    randomize_mass_upper = 0.15  # was 0.05, increased for robustness
    # random forces applied to the object
    force_scale = 2
    random_force_prob_scalar = 0.25
    force_decay = 0.9
    force_decay_interval = 0.08
    # curriculum
    gravity_curriculum = True
    # Gravity scheduler parameters
    gravity_scheduler_enabled: bool = False  # Whether to enable gravity scheduler
    gravity_scheduler_method: str = "linear"  # "None", "linear", "exp", "cos"
    gravity_initial: float = 0.1  # Initial gravity magnitude (positive value, will be negated for z-axis)
    gravity_final: float = 9.81  # Final gravity magnitude (positive value, will be negated for z-axis)
    gravity_scheduler_steps: int = 1000  # Number of steps to reach final gravity
    
    # Virtual object force curriculum parameters
    virtual_object_force_enabled: bool = False  # Whether to enable virtual object force curriculum
    virtual_object_force_decay_method: str = "linear"  # "None", "linear", "exp", "cos"
    virtual_object_kp_initial: float = 30.0  # Initial proportional gain for virtual force
    virtual_object_kp_final: float = 0.0  # Final proportional gain (decay to zero)
    virtual_object_kd_initial: float = 3.0  # Initial derivative gain for virtual force
    virtual_object_kd_final: float = 0.0  # Final derivative gain (decay to zero)
    virtual_object_force_decay_steps: int = 10000  # Number of steps to decay gains to final values
    virtual_force_scale = 0.4
    # debug visualize
    debug_draw = True

    # recorder
    recorder_enabled = True
    recorder_save_dir = './recorded_trajectories'

    # deploy params
    speed_coef = 0.5
    current_coef = 0.3
    control_freq = 30  # Deploy control frequency (Hz), matches sim: 120Hz / decimation=4 = 30Hz = demo 30fps

    # Emergency stop safety limits (deploy only)
    emergency_stop_enabled: bool = True
    # 2.5 rad/s leaves a buffer above the policy's natural peak (~2.0 rad/s
    # at proper 30Hz). At 2.0 the policy sometimes brushes the limit on
    # snappy motions and gets a false e-stop, especially with the V3 pkfk
    # path running at the trained 30Hz cadence. FR3 hardware limit is 2.62
    # rad/s; 2.5 still leaves a safety margin to that.
    arm_joint_vel_limit: float = 2.5       # rad/s, max arm joint velocity before e-stop
    arm_joint_delta_limit: float = 0.3     # rad, max single-step arm joint change before e-stop
    # 1.5 rad covers thumb_IP demo-target vs real-current gap right after reset
    # (target ~1.31 rad = SDK upper limit, real hand may not reach it during the
    # reset settle window before the first rollout step → false e-stop).
    # The check semantics is target_command - measured_angle, so this is
    # really a "policy is asking for something the hand can't physically
    # reach yet" detector, not an instantaneous step-jump cap. 1.5 leaves
    # the loud-mistake band (>2 rad) intact while tolerating reset slop.
    hand_joint_delta_limit: float = 1.5    # rad, max single-step hand joint change before e-stop
    # Deploy debug recording
    deploy_debug_record: bool = True
    deploy_debug_record_steps: int = 100
    deploy_debug_record_dir: str = "logs/deploy_debug"
    # Wrist calibration offsets (Exp 0)
    # Root cause: sim uses right_hand_C_MC (Sharpa base link) as EE.
    # ROS2 /franka_wrist_state publishes fr3_hand/fr3_EE (different body → different pos+quat).
    # Measure these once via compare_obs.py --frame 10, then set here.
    #   wrist_pos_offset: translation [x,y,z] to ADD to ros2_wrist_position to match sim EE
    #   wrist_quat_offset: quaternion [w,x,y,z] to PREMULTIPLY ros2_wrist_quat (R_offset * R_ros2)
    wrist_pos_offset: tuple = _dcfg.WRIST_POS_OFFSET  # (-0.1,0,0.415) — edit in dexx/deploy_config.py
    wrist_quat_offset: tuple = (1.0, 0.0, 0.0, 0.0)
    clip_obs = 5.0
    clip_actions = 1.0
    action_scale = 1 / 24

    # ============================================================
    # Final-frame "success" shaping
    # ============================================================
    # Three additive reward terms computed in compute_imitation_reward,
    # all keyed off obj_trajectory[:, seq_len-1] (and the last K frames):
    #
    #   reward_final_pos      = exp(-α_pos · ||cur_obj − final_target||)
    #   reward_final_rot      = exp(-α_rot · |Δrot|)
    #   reward_final_approach = exp(-α_pos · min_k ||cur_obj − last_K[k]||)
    #
    # Defaults below were retuned 2026-05-19 from V1 ablation (see
    # RECENT_CHANGES.md Week 13 §6). Old defaults (`pos_weight=10, alpha=30,
    # window=5`) gave frame-0 reach success 34% (F baseline); new defaults
    # (`pos_weight=30, alpha=15, window=10`) → 56% reach (+22pp), 75% random
    # (+16pp) on `franka-sharpa-force-poseobs`. Trade-off: −11pp on mid-grasp
    # stage 2 (84.4%→73.4%) — policy becomes more goal-fixated.
    # Larger weight + slower α-decay + wider window combine to give a
    # denser final-pose signal that propagates back through the trajectory.
    # If you observe regression on grasp/manipulation tasks, dial pos_weight
    # back to 20 first, or shrink window to 7.
    success_pos_weight: float = 30.0    # 2026-05-19: 10.0 → 30.0 (V1 ablation)
    success_rot_weight: float = 5.0
    success_approach_weight: float = 1.0
    success_alpha_pos: float = 15.0     # 2026-05-19: 30.0 → 15.0 (V1 ablation)
                                         # exp decay on position dist (m)
    success_alpha_rot: float = 3.0      # exp decay on rotation angle (rad)
    success_reward_window: int = 10     # 2026-05-19: 5 → 10 (V1 ablation)
                                         # K = how many trailing frames count for "approach"
    success_reward_ramp: bool = True    # ramp from 0→1 over progress ∈ [max_length−K, max_length]
                                         # prevents policy from "shortcutting" to endpoint early

    # ---- In-hand manipulation reward (2026-06-06 tweak) ----
    # Penalize fingertip-object relative velocity when in contact, so the policy
    # learns to hold the object STILL inside the grasp (matching demo trajectories
    # for rotate / spin / pour) rather than getting away with friction-only grip.
    # 0.0 = disabled (old behavior); 0.5 = modest, below fingertip-force weight 3.0.
    no_slip_weight: float = 0.5

    # ---- Premature-contact failure (gates frame-0 reaching) -----------
    # Original behavior: episode fails if any fingertip <0.005m from object
    # AND demo target says no-contact AND running_progress >= 50. This kills
    # frame-0 reaching episodes because the policy approaches faster than
    # demo and triggers fingertip-touch before demo's expected contact frame.
    # Knobs:
    #   premature_contact_enabled = False    → fully disable the check
    #   premature_contact_dist_threshold     → lower = stricter
    #   premature_contact_progress_threshold → delay activation
    premature_contact_enabled: bool = True
    premature_contact_dist_threshold: float = 0.005
    premature_contact_progress_threshold: int = 50

    # ---- Eval infra: disable ALL terminations for full-rollout eval ------
    # When True, every `fail/*` cause is computed and LOGGED as before but
    # the aggregated `failed_execute` is zeroed → episodes run to
    # `episode_length_buf >= max_length` (i.e., natural time-out at seq_len-1).
    # Useful for diagnostic eval: see what the policy actually does when not
    # killed early, and which fail/* fire diagnostically.
    # NEVER enable during training (policy would have no incentive to avoid
    # failures since they cost nothing).
    eval_no_terminate: bool = False




@configclass
class EventCfg:
    def rand_params(self, scale_range: list[float, float, int]):
        self.randomize_scale = EventTermCfg(
            func=randomize_rigid_body_scale,
            mode="prestartup",
            params={
                "scale_range": scale_range,
                "asset_cfg": SceneEntityCfg("object"),
            },
        )

def update_cfg_for_hand_side(cfg: "FrankaSharpaEnvCfg", hand_side: str):
    """Update all configuration parameters that depend on hand_side.
    
    Args:
        cfg: The configuration object to update
        hand_side: Either "left" or "right"
    """
    cfg.hand_side = hand_side
    cfg.hand_file_name = "Right" if hand_side == "right" else "Left"
    cfg.hand_name = hand_side
    
    # Update arm initial joint positions based on hand side
    cfg.arm_init_joint_pos = (
        [0.24435, 0.17453, -0.13963, -2.14675, -1.78024, 1.83260, -0.05236] 
        if hand_side == "right" 
        else [0.24435, 0.13963, -0.27925, -2.30383, 1.97222, 1.69297, -0.62832]
    )
    
    # Update robot_cfg spawn: the merged FR3 + Sharpa Wave URDF for this side.
    workspace_root = get_workspace_root()
    cfg.robot_cfg.spawn.usd_path = franka_sharpa_robot_usd(hand_side)
    # Optional override (a different robot asset). Default None keeps stock behavior.
    _asset_override = getattr(cfg, "robot_asset_override", None)
    if _asset_override:
        cfg.robot_cfg.spawn.usd_path = _asset_override if os.path.isabs(_asset_override) \
            else os.path.join(workspace_root, _asset_override)
        print(f"[cfg] robot asset overridden -> {cfg.robot_cfg.spawn.usd_path}")
    
    # Update robot_cfg init_state joint_pos with hand_side prefix
    cfg.robot_cfg.init_state.joint_pos = {
        "fr3_joint1": cfg.arm_init_joint_pos[0],
        "fr3_joint2": cfg.arm_init_joint_pos[1],
        "fr3_joint3": cfg.arm_init_joint_pos[2],
        "fr3_joint4": cfg.arm_init_joint_pos[3],
        "fr3_joint5": cfg.arm_init_joint_pos[4],
        "fr3_joint6": cfg.arm_init_joint_pos[5],
        "fr3_joint7": cfg.arm_init_joint_pos[6],
        # Hand joints
        f"{hand_side}_thumb_CMC_FE": math.pi/180 * 94.33,
        f"{hand_side}_thumb_CMC_AA": math.pi/180 * -14.90,
        f"{hand_side}_thumb_MCP_FE": math.pi/180 * 27.79,
        f"{hand_side}_thumb_MCP_AA": math.pi/180 * -0.14,
        f"{hand_side}_thumb_IP": math.pi/180 * 10.44,
        f"{hand_side}_index_MCP_FE": math.pi/180 * 63.32, 
        f"{hand_side}_index_MCP_AA": math.pi/180 * -4.95,
        f"{hand_side}_index_PIP": math.pi/180 * 28.80,
        f"{hand_side}_index_DIP": math.pi/180 * 19.52,
        f"{hand_side}_middle_MCP_FE": math.pi/180 * 26.46,
        f"{hand_side}_middle_MCP_AA": math.pi/180 * -10.42,
        f"{hand_side}_middle_PIP": math.pi/180 * 45.27,
        f"{hand_side}_middle_DIP": math.pi/180 * 14.61,
        f"{hand_side}_ring_MCP_FE": math.pi/180 * 24.44,
        f"{hand_side}_ring_MCP_AA": math.pi/180 * 7.01,
        f"{hand_side}_ring_PIP": math.pi/180 * 36.34,
        f"{hand_side}_ring_DIP": math.pi/180 * 22.85,
        f"{hand_side}_pinky_CMC": math.pi/180 * 5.25,
        f"{hand_side}_pinky_MCP_FE": math.pi/180 * 51.80,
        f"{hand_side}_pinky_MCP_AA": math.pi/180 * 8.72,
        f"{hand_side}_pinky_PIP": math.pi/180 * 35.61,
        f"{hand_side}_pinky_DIP": math.pi/180 * 29.33,
    }
    
    # Update robot_cfg actuators hand_joints joint_names_expr
    cfg.robot_cfg.actuators["hand_joints"].joint_names_expr = [f"{hand_side}_.*"]
    # Re-key the hand gains/armature/friction for this side (they are authored
    # for "right" in the class body). Armature is load-bearing: see robot_constants.
    for _k, _v in _hand_gain_dicts(hand_side).items():
        setattr(cfg.robot_cfg.actuators["hand_joints"], _k, _v)
    
    # Update contact_sensor paths
    cfg.contact_sensor = [
        # elastomer
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{hand_side}_thumb_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=32,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{hand_side}_index_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=32,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{hand_side}_middle_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=32,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{hand_side}_ring_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=32,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{hand_side}_pinky_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=32,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        # DP
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{hand_side}_thumb_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{hand_side}_index_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{hand_side}_middle_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{hand_side}_ring_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{hand_side}_pinky_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        )
    ]
    
    # Update actuated_joint_names
    cfg.actuated_joint_names = [
        f"{hand_side}_thumb_CMC_FE",
        f"{hand_side}_thumb_CMC_AA",
        f"{hand_side}_thumb_MCP_FE",
        f"{hand_side}_thumb_MCP_AA",
        f"{hand_side}_thumb_IP",
        f"{hand_side}_index_MCP_FE",
        f"{hand_side}_index_MCP_AA",
        f"{hand_side}_index_PIP",
        f"{hand_side}_index_DIP",
        f"{hand_side}_middle_MCP_FE",
        f"{hand_side}_middle_MCP_AA",
        f"{hand_side}_middle_PIP",
        f"{hand_side}_middle_DIP",
        f"{hand_side}_ring_MCP_FE",
        f"{hand_side}_ring_MCP_AA",
        f"{hand_side}_ring_PIP",
        f"{hand_side}_ring_DIP",
        f"{hand_side}_pinky_CMC",
        f"{hand_side}_pinky_MCP_FE",
        f"{hand_side}_pinky_MCP_AA",
        f"{hand_side}_pinky_PIP",
        f"{hand_side}_pinky_DIP",
    ]
    
    # Update fingertip_body_names
    cfg.fingertip_body_names = [
        f"{hand_side}_thumb_fingertip",
        f"{hand_side}_index_fingertip",
        f"{hand_side}_middle_fingertip",
        f"{hand_side}_ring_fingertip",
        f"{hand_side}_pinky_fingertip",
    ]


# =============================================================================
# Experiment-specific cfg subclasses
#
# Strategy: keep `FrankaSharpaEnvCfg` as the canonical deploy-friendly default
# (DR on, joint_delta control, modest reset randomization). Define narrow
# subclasses below that flip just the fields that matter for a given experiment.
# Each subclass inherits everything else, so you only see the *deltas* and don't
# have to grep for what's different vs base.
#
# Add a new subclass when starting a new experiment family. Keep the diff small
# (5-15 lines max). If you find yourself copy-pasting > 20 fields, that's a
# sign the base default is wrong, not that you need a fatter subclass.
# =============================================================================


@configclass
class FrankaSharpaDeployCfg(FrankaSharpaEnvCfg):
    """Deploy / sim2real-friendly defaults — explicit name for the policy that
    eventually runs on the real Franka + Sharpa.

    Inherits the canonical base. Listed here only for naming clarity and as a
    single grep-able anchor for the "deploy stack" — when you change deploy
    behavior, change this class so other experiments don't accidentally pick
    it up.

    Concrete deploy/training entry-points should reference this class (not
    FrankaSharpaEnvCfg directly) so the intent is documented.
    """
    pass


@configclass
class FrankaSharpaSimTeacherCfg(FrankaSharpaEnvCfg):
    """Fastest-path sim teacher cfg — for getting a high success-rate policy
    inside Isaac Lab as quickly as possible, no sim2real concern.

    Differences vs deploy default (and *why* each one matters in sim):

      Controller:
        OSC instead of joint_delta. OSC's action space is Cartesian
        (pos_error / rot_error_6d / hand 22), so the policy doesn't need to
        implicitly learn 7-DoF arm IK. Standalone tracking test gave
        pos_mean=15mm / rot_mean=9° — close to perfect — so reward gradients
        flow back along a much shorter path than they do through joint_delta.

      Reset curriculum:
        Enabled with start=0.7 → end=0.0 over 2000 steps. Early training only
        samples reset frames in the last 30% of the trajectory (close to /
        post-grasp). Avoids the cold-start where every env starts pre-approach
        and has to learn the entire pipeline at once.

      Adaptive sampling:
        Off by default — base behaviour with curriculum is already strong;
        adaptive only helps once you have a baseline. Flip on later if some
        bins of the trajectory turn out to be persistently hard.

      All domain randomization:
        Off. DR is non-negotiable for real transfer but it costs sample
        efficiency. A teacher that never sees DR will overfit to nominal
        dynamics, which is fine — we'll BC/DAgger distil to a deploy student
        with DR on later.

      Observation noise:
        Zero. Same logic as DR.

    Use via task ID 'franka-sharpa-force-critic-horizon-simteacher' (register
    in tasks/franka_sharpa/__init__.py with this cfg).
    """

    # ---- Controller: OSC (sim-only fast path) ----
    use_osc_control: bool = True
    use_pid_control: bool = False
    use_joint_pos_control: bool = False
    use_joint_delta_control: bool = False

    # ---- Reset / curriculum: easy-first ----
    random_state_init: bool = True
    init_curriculum_enabled: bool = True
    init_curriculum_method: str = "linear"
    init_curriculum_start: float = 0.7
    init_curriculum_end: float = 0.0
    init_curriculum_steps: int = 2000

    # Adaptive sampling stays off; turn on by overriding in a deeper subclass
    # once baseline works. If you do enable, keep compose_with_curriculum=True.
    adaptive_sampling_enabled: bool = False

    # ---- Domain randomization: ALL OFF (sim teacher only) ----
    randomize_pd_gains: bool = False
    randomize_arm_pd_gains: bool = False
    randomize_action_delay: bool = False
    randomize_friction: bool = False
    randomize_mass: bool = False
    randomize_com: bool = False

    # ---- Observation noise: zero ----
    obs_joint_pos_noise: float = 0.0
    obs_wrist_pos_noise: float = 0.0
    obs_wrist_rot_noise: float = 0.0
