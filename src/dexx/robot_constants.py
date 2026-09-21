"""Actuation constants for the FR3 + Sharpa Wave robot.

Deliberately free of ``isaaclab`` imports, so offline tooling can read these
without launching a simulator.

Geometry constants (arm base pose, table) live in :mod:`dexx.deploy_config`,
which is the single source shared with the retarget and deploy sides.
"""

from __future__ import annotations

# Per-joint hand PD gains (N*m/rad, N*m*s/rad), rotor inertia and joint friction.
#
# Read out of the vendor's shipped USD and converted from USD's per-DEGREE drive
# convention (x 180/pi). The legacy `Right_final.usda` and the public hand
# release carry identical values, so this is one calibration, not two.
#
# ``armature`` is the load-bearing entry. URDF cannot express rotor inertia, so a
# URDF-imported hand gets armature=0 -- and these finger links have inertias on
# the order of 1e-6 kg*m^2, so even a 0.2 N*m torque produces enough angular
# acceleration to tunnel straight through the PhysX joint limit within one
# 1/120 s step. Measured: 14 of 29 joints outside their limits after 100
# zero-action steps, one at 15 rad against a [0, 1.75] limit. Restored: 0 of 29.
#
# Since the robot is now spawned from the merged URDF (see
# ``scripts/build_merged_urdf.py``) rather than a pre-converted USD, these values
# are the ONLY place the gains come from. Do not set them to None.
HAND_GAINS = {                    # suffix -> (stiffness, damping, armature, friction)
    "thumb_CMC_FE": (6.955, 0.2410, 0.00320, 0.132000),
    "thumb_CMC_AA": (13.201, 0.4516, 0.00320, 0.132000),
    "thumb_MCP_FE": (4.760, 0.1830, 0.00265, 0.104000),
    "thumb_MCP_AA": (6.622, 0.2080, 0.00265, 0.104000),
    "thumb_IP": (0.908, 0.0400, 0.00061, 0.024760),
    "pinky_CMC": (1.380, 0.0392, 0.00012, 0.013000),
}
_FINGER_ROLE_GAINS = {            # the four 4-finger roles
    "MCP_FE": (4.760, 0.1830, 0.00265, 0.104000),
    "MCP_AA": (6.622, 0.2080, 0.00265, 0.104000),
    "PIP": (0.908, 0.0400, 0.00061, 0.024760),
    "DIP": (0.904, 0.0315, 0.00042, 0.000418),
}
for _f in ("index", "middle", "ring", "pinky"):
    for _role, _g in _FINGER_ROLE_GAINS.items():
        HAND_GAINS.setdefault(f"{_f}_{_role}", _g)

# Arm rotor inertia / joint friction, from the legacy asset's authored
# `physxJoint:armature` / `jointFriction` (uniform across the 7 FR3 joints).
#
# NOTE: the arm's stiffness/damping are NOT here -- they stay in
# `franka_sharpa_env_cfg.py`, where the 2026-04-23 step-response-tuned set is
# what every shipped checkpoint was trained against.
ARM_ARMATURE = 1.0
ARM_FRICTION = 0.2

# Joint effort/velocity limits are deliberately absent: the merged URDF's own
# values (hand 0.19-3.3 N*m, arm 87 N*m) are correct and match the legacy
# asset's `maxForce` exactly. Overriding them is what lets bad gains act.


def hand_gain_dicts(side: str) -> dict:
    """Side-prefixed {stiffness,damping,armature,friction} dicts for an actuator cfg."""
    return {
        "stiffness": {f"{side}_{k}": v[0] for k, v in HAND_GAINS.items()},
        "damping": {f"{side}_{k}": v[1] for k, v in HAND_GAINS.items()},
        "armature": {f"{side}_{k}": v[2] for k, v in HAND_GAINS.items()},
        "friction": {f"{side}_{k}": v[3] for k, v in HAND_GAINS.items()},
    }
