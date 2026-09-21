"""dexx.deploy_config — SINGLE SOURCE OF TRUTH for sim2real / deploy constants.

Everything that used to be hard-coded across the env cfgs, the deploy env, the
depth subscribers and the retarget scripts is collected here so you only edit ONE
file when your table height, camera, or NUC bridge changes.

This module imports NOTHING from `dexx` (leaf module) — safe to import anywhere
without circular-import risk. Values below MUST match what the policy was trained
with; changing them changes runtime geometry.

Edit here, then everything downstream picks it up:
  - env cfgs (arm_base_pos, camera intrinsics, PC workspace crop)
  - deploy env (arm-base offset, intrinsics, workspace fallback)
  - depth subscribers (sim intrinsics)
  - retarget scripts (arm_base_pos, table height)
"""
from __future__ import annotations

import os

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Table & arm-base geometry — ENV-LOCAL frame, meters
# ─────────────────────────────────────────────────────────────────────────────
# Table top surface z in env-local. The manipulated object is placed relative to
# THIS plane (table-relative), independent of the arm base.
TABLE_SURFACE_Z: float = 0.415

# fr3_link0 (arm base) height in env-local. If your real base sits at a
# different height above the table, change THIS.
#
# History: 0.415 -> 0.432 on 2026-07-16, to match a real base mounted 1.7 cm
# above the table. Back to 0.415 on 2026-09-01 because the arm is being
# remounted level with the table surface.
#
# This is not a free parameter. Demonstrations store joint angles, so moving the
# base moves the hand by the same 17 mm against an object that has not moved --
# 13-50% of the hand's 3.3-13.2 cm grasp aperture. Every retarget, every teacher
# and every policy trained against the old value has to be redone, and the
# camera extrinsic re-measured. Do not change it to make a number look right.
ARM_BASE_Z: float = 0.415
ARM_BASE_POS: tuple = (-0.1, 0.0, ARM_BASE_Z)
ARM_BASE_ROT: tuple = (1.0, 0.0, 0.0, 0.0)

# Wrist-pose retarget offset — the table height, which since 2026-09-01 is also
# the arm-base height. They were deliberately different while the base was
# raised; keep this written as the table plane rather than aliased to
# ARM_BASE_Z, so that raising the base again does not silently move it too.
WRIST_POS_OFFSET: tuple = (-0.1, 0.0, TABLE_SURFACE_Z)

# ─────────────────────────────────────────────────────────────────────────────
# Camera intrinsics — 320x240 (the sim/deploy PC back-projection resolution)
# D455 640x480 depth decimated x2 -> 320x240, whose intrinsics match these to <1.5px.
# ─────────────────────────────────────────────────────────────────────────────
SIM_INTRINSICS: dict = {"fx": 193.33, "fy": 193.06, "cx": 160.08, "cy": 121.05}
DEPTH_H: int = 240
DEPTH_W: int = 320

# ─────────────────────────────────────────────────────────────────────────────
# PointCloud workspace crop — ENV-LOCAL frame, meters (drops table + curtain hits)
# z_min is ~5mm above the table so the flat table plane is removed from scene_pc.
# ─────────────────────────────────────────────────────────────────────────────
PC_WORKSPACE_MIN: tuple = (0.00, -0.40, 0.420)
PC_WORKSPACE_MAX: tuple = (0.80,  0.25, 1.30)

# ─────────────────────────────────────────────────────────────────────────────
# Deploy comm — Polymetis joint bridge (NUC) + camera depth publisher (ZMQ)
# ─────────────────────────────────────────────────────────────────────────────
POLYMETIS_STATE_PORT: int = 5560   # bridge PUB (joint/ee state)
POLYMETIS_CMD_PORT: int = 5561     # bridge PULL (joint targets)
# Example camera-host depth publisher addr; override per-run with --depth_zmq_addr.
CAMERA_ZMQ_ADDR_EXAMPLE: str = "tcp://101.6.90.122:5562"


def arm_base_pos_np() -> np.ndarray:
    """ARM_BASE_POS as a float32 numpy array (for the deploy PC offset)."""
    return np.array(ARM_BASE_POS, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Camera extrinsic — a FILE, not a constant
# ─────────────────────────────────────────────────────────────────────────────
# The camera-in-armbase 4x4 is a property of how the camera is bolted down, so it
# is calibrated per mount and lives in `calib/camera_align/`. It is deliberately
# not a literal in this file.
#
# It used to be hard-coded in two places (the sim raycaster and the depth
# subscriber). Both copies were a 2026-05-20 calibration; by the time anyone
# noticed, the camera had moved 20 cm and policies had been trained against the
# wrong viewpoint without a single warning. Loading the shipped file instead
# means the number exists once, and it is the current one.
#
# Always pass --camera_extrinsic explicitly for a run you intend to reproduce.
# This default exists so that forgetting is merely wrong-by-one-calibration
# rather than wrong-by-four-months.
CAMERA_EXTRINSIC_DEFAULT_FILE: str = "calib/camera_align/current.npy"


def _repo_root() -> str:
    # src/dexx/deploy_config.py -> src/dexx -> src -> <repo>
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_camera_extrinsic(required: bool = True) -> "np.ndarray | None":
    """Load the shipped camera-in-armbase 4x4 (ROS optical convention).

    Raises by default rather than returning a guess: a silently wrong camera
    pose produces a policy that trains happily and reaches for the wrong place.
    """
    path = os.path.join(_repo_root(), CAMERA_EXTRINSIC_DEFAULT_FILE)
    if not os.path.exists(path):
        if not required:
            return None
        raise FileNotFoundError(
            f"no camera extrinsic at {path}. Calibrate one (tutorial/06) or pass "
            f"--camera_extrinsic <file.npy>. There is deliberately no hard-coded "
            f"fallback: a stale one cost this project a month of training on the "
            f"wrong viewpoint.")
    T = np.load(path).astype(np.float32)
    if T.shape != (4, 4):
        raise ValueError(f"{path}: expected a 4x4 matrix, got {T.shape}")
    return T
