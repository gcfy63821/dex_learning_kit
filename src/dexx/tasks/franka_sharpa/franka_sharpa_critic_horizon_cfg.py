"""Cfg subclass for the critic-horizon variant of franka-sharpa-force.

Isolated from the base `FrankaSharpaEnvCfg` so flipping between the original
`franka-sharpa-force` task and this variant is just a task-id change.

Differences vs base cfg:
 - `observation_space` starts from a new base that includes `target_obj_pos`/`quat`
   and `tips_distance` as deployable actor obs (computed at env init).
 - `priv_info_dim` enlarged to hold K future frames of object target + tips_distance
   + 1 frame of delta-vs-current-object + 5 obj-to-fingertip distances.
 - New fields controlling which critic-only signals are written.
"""

from isaaclab.utils import configclass

from .franka_sharpa_env_cfg import FrankaSharpaEnvCfg


# ---------------------------------------------------------------------------
# Arm joint-impedance gains — the single source of truth.
#
# Step-response tuned against the real arm; `tools/sysid/` replays a motion in
# both and reports the gap. EVERY SHIPPED CHECKPOINT WAS TRAINED AGAINST THESE,
# so changing them invalidates the checkpoints — it is a retrain, not a tweak.
#
# History:
#   2026-04-23  damping cut on j1-j3 (240/220/180 -> 145/135/110) after a ROS2
#               system-ID showed sim lagging real by 40-50 ms (sim over-damped).
#   2026-07-16  after the move to Polymetis, a second realignment measured
#               j4-j7 still lagging 50-90 ms and found a better set:
#                 KD = [85, 135, 110, 25, 18, 10, 5]
#               which cut lag to 20-30 ms with roughly neutral RMSE
#               (tools/sysid, "it2"). It was NEVER adopted here, so training
#               still runs the 2026-04 values. Adopting it means retraining.
#               Pass it to the replay tools with --arm_kd to reproduce.
ARM_TUNED_KP = {
    "fr3_joint1": 1600.0, "fr3_joint2": 1600.0, "fr3_joint3": 1200.0,
    "fr3_joint4": 800.0,  "fr3_joint5": 500.0,  "fr3_joint6": 300.0,
    "fr3_joint7": 150.0,
}
ARM_TUNED_KD = {
    "fr3_joint1": 145.0, "fr3_joint2": 135.0, "fr3_joint3": 110.0,
    "fr3_joint4": 100.0, "fr3_joint5": 50.0,  "fr3_joint6": 30.0,
    "fr3_joint7": 15.0,
}

# The 2026-07-16 Polymetis realignment candidate. Recorded, not active.
ARM_KD_POLYMETIS_IT2 = {
    "fr3_joint1": 85.0, "fr3_joint2": 135.0, "fr3_joint3": 110.0,
    "fr3_joint4": 25.0, "fr3_joint5": 18.0,  "fr3_joint6": 10.0,
    "fr3_joint7": 5.0,
}


@configclass
class FrankaSharpaCriticHorizonCfg(FrankaSharpaEnvCfg):
    # Critic-only future horizon for object/tips_distance targets.
    critic_future_length: int = 5

    # Per-component enable flags (all default on).
    critic_enable_target_obj_future: bool = True    # K × (pos 3 + quat 4 + vel 3 + ang_vel 3)
    critic_enable_tips_distance_future: bool = True  # K × 5
    critic_enable_delta_obj_current: bool = True     # 1 × 13  (pos 3 + quat 4 + vel 3 + ang_vel 3)
    critic_enable_obj_to_tips_current: bool = True   # 5 (fingertip distances to object)

    # Actor-visible masks. These keep observation dimensionality fixed
    # and zero selected demo/object/trajectory hints for ablations.
    actor_mask_extra_geometry: bool = False  # masks target object pose + tips_distance + BPS
    actor_mask_target_obj_pose: bool = False # masks actor target_obj_pos/quat only
    actor_mask_tips_distance: bool = False   # masks actor demo/retarget tips_distance only
    actor_mask_bps: bool = False             # masks actor object BPS only
    actor_mask_target_hand: bool = False     # masks actor future hand keypoint targets only

    # This variant is designed to be trained with asymmetric AC by default.
    asymmetric_ac: bool = True

    # Observation-space: actor-visible obs. When enable_bps=True (default),
    # both asymmetric and non-asymmetric paths converge to 550.
    # Matrix (bps=128):
    #   asymmetric=T, enable_bps=T: parent=410, +12 tips+target_obj +128 bps = 550
    #   asymmetric=T, enable_bps=F: parent=410, +12 tips+target_obj          = 422
    #   asymmetric=F, enable_bps=T: parent=543, +7 target_obj                = 550
    #   asymmetric=F, enable_bps=F: parent=415, +7 target_obj                = 422
    observation_space = 550

    # priv_info_dim = 40 (existing slots) + 18*K + 18 (delta + obj_to_tips)
    #              = 40 + 18*5 + 18 = 148   (for K=5 default)
    priv_info_dim = 148

    # ---- Reset / curriculum: easy-first ----
    # Retuned 2026-05-19 from V1 ablation (RECENT_CHANGES.md Week 13 §6):
    #   start=0.5 → 0.3 (broader early exposure, hits reach phase sooner)
    #   steps=10000 → 15000 (gentler ramp)
    # Combined with reward boost (success_pos_weight=30, alpha=15, window=10
    # also retuned 2026-05-19), frame-0 reach success: 34% → 56%.
    # adaptive_sampling_enabled=False here, but V1 had it True via --env_cfg —
    # keep False default for stability; opt-in via CLI for known-good runs.
    random_state_init: bool = True
    init_curriculum_enabled: bool = True
    init_curriculum_method: str = "linear"
    init_curriculum_start: float = 0.3      # 2026-05-19: 0.5 → 0.3 (V1)
    init_curriculum_end: float = 0.0
    init_curriculum_steps: int = 15000      # 2026-05-19: 10000 → 15000 (V1)
    adaptive_sampling_enabled: bool = False

    # ---- Anti-shake training knobs (2026-05-09) ----
    # Lower moving-average → smoother arm command (sacrifices a bit of responsiveness).
    # 0.7 (base) was too aggressive; 0.4 filters high-frequency policy jitter.
    actions_moving_average: float = 0.4

    # Coefficient on `penal_arm_action_rate_l2` reward term. Old hardcoded value
    # was 0.1; raising to 0.5 forces policy to output smoother arm actions.
    arm_action_rate_penalty: float = 0.15

    # Per-joint tuned arm PD (matches real DexhandJointImpedanceController). Set in
    # __post_init__ so we don't have to redefine the whole robot_cfg.
    use_per_joint_tuned_arm_gains: bool = True

    # ---- Reward shaping: no-slip + approach v2 (2026-05-09) ----
    # `no_slip_weight`: penalize fingertip-vs-object relative velocity when in
    # contact. Targets `fail/obj_pos_drift`. Set 0 to disable.
    no_slip_weight: float = 1.5

    # `approach_shaping_v2`: replace exp(-5d) approach reward (which has near-
    # zero gradient when far) with 1/(1+5d) — dense gradient at all distances.
    # Helps policy learn to approach instead of avoiding contact.
    approach_shaping_v2: bool = True

    def __post_init__(self):
        super().__post_init__() if hasattr(super(), "__post_init__") else None
        if self.use_per_joint_tuned_arm_gains:
            self.robot_cfg.actuators["arm_joints"].stiffness = dict(ARM_TUNED_KP)
            self.robot_cfg.actuators["arm_joints"].damping = dict(ARM_TUNED_KD)


@configclass
class FrankaSharpaCriticHorizonSimTeacherCfg(FrankaSharpaCriticHorizonCfg):
    """Sim-only fast teacher cfg for the critic-horizon variant.

    Mirrors `FrankaSharpaSimTeacherCfg` (in franka_sharpa_env_cfg.py) but
    extends the critic-horizon parent so actor obs (550) and priv_info (148)
    stay correct. Use task ID `franka-sharpa-force-critic-horizon-simteacher`.

    All deltas vs `FrankaSharpaCriticHorizonCfg`:

      Controller      : OSC (Cartesian effort), shorter reward gradient path,
                        no implicit IK to learn → fastest sim convergence.
      Reset           : init_curriculum on (start=0.7 → end=0.0 over 2k steps),
                        easy-first warmup.
      Adaptive sample : off; turn on later if a baseline plateaus on hard bins.
      Domain rand     : OFF (PD gains, arm PD, action delay, friction, mass, COM).
      Obs noise       : ZERO (joint pos / wrist pos / wrist rot).

    DR + obs noise will be reintroduced in a deploy-distillation cfg later.
    """

    # ---- Controller: OSC (sim-only fast path) ----
    use_osc_control: bool = False
    use_pid_control: bool = False
    use_joint_pos_control: bool = False
    use_joint_delta_control: bool = True

    # ---- Reset / curriculum: easy-first ----
    random_state_init: bool = True
    init_curriculum_enabled: bool = False
    init_curriculum_method: str = "linear"
    init_curriculum_start: float = 0.7
    init_curriculum_end: float = 0.0
    init_curriculum_steps: int = 2000
    adaptive_sampling_enabled: bool = False

    # ---- DR off ----
    randomize_pd_gains: bool = False
    randomize_arm_pd_gains: bool = False
    randomize_action_delay: bool = False
    randomize_friction: bool = False
    randomize_mass: bool = False
    randomize_com: bool = False

    # ---- Obs noise off ----
    obs_joint_pos_noise: float = 0.0
    obs_wrist_pos_noise: float = 0.0
    obs_wrist_rot_noise: float = 0.0
