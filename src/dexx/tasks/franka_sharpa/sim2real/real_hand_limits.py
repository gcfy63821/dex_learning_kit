"""Empirically measured reachable range of the Sharpa HA4 real hand.

Measured on 2026-04-19 with `probe_hand_limits.py` (normal mode, i.e.
calibration_mode=0 — the same mode used at deploy time).

Source of each entry:
  - FE / PIP / DIP / IP / pinky_CMC (non-coupled joints):
        SDK `min_position`/`max_position` (firmware soft limits —
        probe hit these cleanly).
  - Thumb/index/middle/ring/pinky MCP_AA and pinky_CMC:
        probe normal-mode actual values (tighter than SDK soft
        limits due to inter-finger mechanical coupling).

Ordering: cfg / Sharpa / retargeted `opt_dof_pos` order (22 joints).
This is the same order as `cfg.actuated_joint_names` and what
`real_hand.set_joint_position()` expects.
"""
from __future__ import annotations

import numpy as np


# (lower, upper) in radians, cfg (Sharpa) order.
SHARPA_REAL_LIMITS = (
    (-0.087,  1.833),   # 0  thumb_CMC_FE       SDK
    (-0.307,  0.061),   # 1  thumb_CMC_AA       probe AA
    (-0.436,  1.309),   # 2  thumb_MCP_FE       SDK
    (-0.307,  0.308),   # 3  thumb_MCP_AA       probe AA
    ( 0.000,  1.658),   # 4  thumb_IP           SDK
    (-0.175,  1.466),   # 5  index_MCP_FE       SDK
    (-0.309,  0.106),   # 6  index_MCP_AA       probe AA
    ( 0.000,  1.658),   # 7  index_PIP          SDK
    ( 0.000,  1.309),   # 8  index_DIP          SDK
    (-0.175,  1.466),   # 9  middle_MCP_FE      SDK
    (-0.136,  0.159),   # 10 middle_MCP_AA      probe AA (coupled)
    ( 0.000,  1.658),   # 11 middle_PIP         SDK
    ( 0.000,  1.309),   # 12 middle_DIP         SDK
    (-0.175,  1.466),   # 13 ring_MCP_FE        SDK
    (-0.027,  0.147),   # 14 ring_MCP_AA        probe AA (coupled)
    ( 0.000,  1.658),   # 15 ring_PIP           SDK
    ( 0.000,  1.309),   # 16 ring_DIP           SDK
    ( 0.008,  0.250),   # 17 pinky_CMC          probe (tighter than SDK)
    (-0.175,  1.466),   # 18 pinky_MCP_FE       SDK
    (-0.026,  0.306),   # 19 pinky_MCP_AA       probe AA (coupled)
    ( 0.000,  1.658),   # 20 pinky_PIP          SDK
    ( 0.000,  1.309),   # 21 pinky_DIP          SDK
)

# Column 0 = lower, column 1 = upper.
SHARPA_REAL_LIMITS_NP = np.asarray(SHARPA_REAL_LIMITS, dtype=np.float32)  # [22, 2]
SHARPA_REAL_LOWER_NP = SHARPA_REAL_LIMITS_NP[:, 0]
SHARPA_REAL_UPPER_NP = SHARPA_REAL_LIMITS_NP[:, 1]


def clamp_to_real_limits_np(hand_pos_sharpa, margin: float = 0.0):
    """Clamp a hand-joint vector/matrix to the real-hand reachable range.

    Args:
        hand_pos_sharpa: np.ndarray of shape [..., 22] in cfg (Sharpa) order.
        margin:          rad of safety margin shaving off both ends (default 0).

    Returns: np.ndarray same shape, clamped.
    """
    arr = np.asarray(hand_pos_sharpa)
    if arr.shape[-1] != 22:
        raise ValueError(f"last dim must be 22, got {arr.shape}")
    lo = SHARPA_REAL_LOWER_NP + margin
    hi = SHARPA_REAL_UPPER_NP - margin
    return np.clip(arr, lo, hi).astype(arr.dtype, copy=False)


def clamp_to_real_limits_torch(hand_pos_sharpa, margin: float = 0.0):
    """Torch variant. Last dim must be 22 in cfg order.

    Broadcasts the [22] limit vectors against arbitrary batch shape.
    Preserves dtype and device of the input.
    """
    import torch  # local import to keep numpy-only consumers light

    if hand_pos_sharpa.shape[-1] != 22:
        raise ValueError(f"last dim must be 22, got {tuple(hand_pos_sharpa.shape)}")
    lo = torch.tensor(SHARPA_REAL_LOWER_NP, device=hand_pos_sharpa.device,
                      dtype=hand_pos_sharpa.dtype) + margin
    hi = torch.tensor(SHARPA_REAL_UPPER_NP, device=hand_pos_sharpa.device,
                      dtype=hand_pos_sharpa.dtype) - margin
    return torch.minimum(torch.maximum(hand_pos_sharpa, lo), hi)


def report_violations(hand_pos_sharpa, tag: str = "") -> dict:
    """Count per-joint target-out-of-limit events (no clamping)."""
    arr = np.asarray(hand_pos_sharpa)
    if arr.ndim == 1:
        arr = arr[None, :]
    under = (arr < SHARPA_REAL_LOWER_NP).sum(axis=0)   # [22]
    over = (arr > SHARPA_REAL_UPPER_NP).sum(axis=0)
    max_under = np.where(arr < SHARPA_REAL_LOWER_NP,
                         SHARPA_REAL_LOWER_NP - arr, 0).max(axis=0)
    max_over = np.where(arr > SHARPA_REAL_UPPER_NP,
                        arr - SHARPA_REAL_UPPER_NP, 0).max(axis=0)
    info = {
        "tag": tag,
        "num_frames": int(arr.shape[0]),
        "joints_under": under.tolist(),
        "joints_over": over.tolist(),
        "max_under_amount": max_under.tolist(),
        "max_over_amount": max_over.tolist(),
        "any_violation": bool(under.sum() + over.sum() > 0),
    }
    return info
