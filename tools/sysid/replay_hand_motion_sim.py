#!/usr/bin/env python3
"""Replay a canonical HAND motion CSV in Isaac Sim with the same hand-PD config
as the training env. Arm is held fixed at the safe home pose via its own PD.

Output pkl schema matches `replay_hand_motion_real.py` so `analyze_motion.py`
can diff them directly.

Usage:
    python tools/sysid/replay_hand_motion_sim.py \
        --motion tools/sysid/motions_hand/hand_sin_all.csv \
        --output logs/system_id/hand_replay/hand_sin_all_sim.pkl \
        --side right --headless
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

HAND_JOINT_SUFFIXES = [
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP",  "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE",   "ring_MCP_AA",   "ring_PIP",   "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",  "pinky_MCP_AA",  "pinky_PIP",  "pinky_DIP",
]

ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
SAFE_ARM_POS = [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0]

parser = argparse.ArgumentParser(description="Replay a HAND motion CSV in Isaac Sim.")
parser.add_argument("--motion", required=True, type=str)
parser.add_argument("--output", required=True, type=str)
parser.add_argument("--side", type=str, default="right", choices=["left", "right"])
parser.add_argument("--control_freq_override", type=float, default=None)
parser.add_argument("--record_freq", type=float, default=100.0)
parser.add_argument("--physics_freq", type=float, default=120.0)
parser.add_argument("--approach_s", type=float, default=2.0)
parser.add_argument("--dry_run", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _load_motion(path: str):
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        rows = [list(map(float, row)) for row in r]
    import numpy as _np
    positions = _np.asarray(rows, dtype=_np.float64)
    json_path = Path(path).with_suffix(".json")
    meta = {}
    if json_path.exists():
        meta = json.load(open(json_path))
    if header != HAND_JOINT_SUFFIXES:
        try:
            perm = [header.index(n) for n in HAND_JOINT_SUFFIXES]
            positions = positions[:, perm]
        except ValueError:
            pass
    return positions, meta


_positions_np, _meta = _load_motion(args_cli.motion)
_ctrl_freq = args_cli.control_freq_override or float(_meta.get("control_freq_hz", 30.0))
_ctrl_dt = 1.0 / _ctrl_freq
print(f"[INFO] motion={_meta.get('name','?')}  {_positions_np.shape[0]} steps @ "
      f"{_ctrl_freq}Hz ({_positions_np.shape[0]*_ctrl_dt:.1f}s)")
if args_cli.dry_run:
    sys.exit(0)


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def main():
    sim_dt = 1.0 / args_cli.physics_freq
    sim_cfg = SimulationCfg(
        dt=sim_dt, device="cuda:0",
        physx=PhysxCfg(solver_type=1, max_position_iteration_count=8,
                       max_velocity_iteration_count=0),
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        num_envs: int = 1
        env_spacing: float = 1.0

    scene = InteractiveScene(SceneCfg())
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

    # Robot USD depends on side
    usd_name = "Right_final.usda" if args_cli.side == "right" else "Left_final.usda"
    from dexx.tasks.franka_sharpa.franka_sharpa_env_cfg import franka_sharpa_robot_usd
    usd_path = franka_sharpa_robot_usd("right")
    hand_prefix = args_cli.side  # e.g. "right_" pattern in joint_names_expr

    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, linear_damping=0.1, angular_damping=0.1,
                max_linear_velocity=1000.0, max_angular_velocity=64 / math.pi * 180.0,
                max_depenetration_velocity=1000.0, max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.1, 0.0, 0.415), rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "fr3_joint1": 0.0, "fr3_joint2": 0.0, "fr3_joint3": 0.0,
                "fr3_joint4": -1.57, "fr3_joint5": 0.0, "fr3_joint6": 1.57, "fr3_joint7": 0.0,
            },
        ),
        actuators={
            # Arm held rigid at home pose with normal PD (no gravity comp here;
            # hand tests don't care about arm precision, just that it doesn't fall).
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint.*"],
                stiffness={
                    "fr3_joint1": 1600.0, "fr3_joint2": 1600.0,
                    "fr3_joint3": 1200.0, "fr3_joint4": 800.0,
                    "fr3_joint5": 500.0, "fr3_joint6": 300.0, "fr3_joint7": 150.0,
                },
                damping={
                    "fr3_joint1": 145.0, "fr3_joint2": 135.0,
                    "fr3_joint3": 110.0, "fr3_joint4": 100.0,
                    "fr3_joint5": 50.0, "fr3_joint6": 30.0, "fr3_joint7": 15.0,
                },
            ),
            # Hand — must match franka_sharpa_env_cfg training values
            "hand_joints": ImplicitActuatorCfg(
                joint_names_expr=[f"{hand_prefix}_.*"],
                stiffness=500.0, damping=30.0,
            ),
        },
    )

    robot = Articulation(robot_cfg)
    scene.articulations["robot"] = robot
    scene.clone_environments(copy_from_source=False)
    scene.filter_collisions()
    sim.reset()

    device = sim.device

    # Map hand joint suffixes -> full-articulation indices (USD order)
    hand_joint_names = [f"{hand_prefix}_{s}" for s in HAND_JOINT_SUFFIXES]
    hand_indices = [robot.joint_names.index(n) for n in hand_joint_names]
    arm_indices = [robot.joint_names.index(n) for n in ARM_JOINT_NAMES]
    print(f"[INFO] hand_indices ({len(hand_indices)}): {hand_indices}")
    print(f"[INFO] arm_indices ({len(arm_indices)}): {arm_indices}")

    # Set robot at arm-home + motion[0] hand pose
    positions = _positions_np.astype(np.float32)
    all_jp = torch.zeros(1, robot.num_joints, device=device)
    for i, idx in enumerate(arm_indices):
        all_jp[0, idx] = float(SAFE_ARM_POS[i])
    for i, idx in enumerate(hand_indices):
        all_jp[0, idx] = float(positions[0, i])
    robot.write_joint_state_to_sim(all_jp, torch.zeros_like(all_jp))

    # approach target: arm held; hand ramps from BASE to motion[0]
    # positions[0] already ~= BASE (thanks to _prepend_approach), so direct set
    # is fine; run some steps to settle.
    for _ in range(int(args_cli.approach_s / sim_dt)):
        robot.set_joint_position_target(all_jp)
        scene.write_data_to_sim()
        sim.step(render=not args_cli.headless)
        scene.update(sim_dt)

    # Main replay loop
    N = positions.shape[0]
    total_duration = N * _ctrl_dt
    n_phys_steps = int(total_duration / sim_dt)
    rec_timestamps, rec_targets, rec_actuals, rec_vels = [], [], [], []
    rec_dt = 1.0 / args_cli.record_freq
    next_rec_t = 0.0
    cur_target_np = positions[0]

    print(f"[INFO] replaying {N} ctrl steps over {total_duration:.2f}s "
          f"({n_phys_steps} physX steps)")

    for phys_step in range(n_phys_steps):
        t = phys_step * sim_dt
        idx = min(int(t / _ctrl_dt), N - 1)
        if not np.array_equal(positions[idx], cur_target_np):
            cur_target_np = positions[idx]
            for i, hi in enumerate(hand_indices):
                all_jp[0, hi] = float(cur_target_np[i])
        robot.set_joint_position_target(all_jp)
        scene.write_data_to_sim()
        sim.step(render=not args_cli.headless)
        scene.update(sim_dt)

        if t >= next_rec_t:
            next_rec_t += rec_dt
            rec_timestamps.append(t)
            rec_targets.append(cur_target_np.copy())
            act = robot.data.joint_pos[0, hand_indices].detach().cpu().numpy().astype(np.float64)
            vel = robot.data.joint_vel[0, hand_indices].detach().cpu().numpy().astype(np.float64)
            rec_actuals.append(act)
            rec_vels.append(vel)

    print(f"[INFO] recorded {len(rec_timestamps)} frames ({rec_timestamps[-1]:.2f}s)")

    out_path = Path(args_cli.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "side": "sim",
        "motion_meta": _meta,
        "control_freq_hz": _ctrl_freq,
        "record_freq_hz": args_cli.record_freq,
        "physics_freq_hz": args_cli.physics_freq,
        "joint_names": HAND_JOINT_SUFFIXES,
        "timestamps": np.asarray(rec_timestamps),
        "targets": np.asarray(rec_targets),
        "actuals": np.asarray(rec_actuals),
        "velocities": np.asarray(rec_vels),
        "sim_config": {
            "hand_K": 500.0,
            "hand_D": 30.0,
            "side": args_cli.side,
        },
    }
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[OK] Saved {out_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
