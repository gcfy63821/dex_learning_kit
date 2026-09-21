# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Deploy env V2 for `franka-sharpa-force-poseobs` — train/deploy reference fix.

## The bug this fixes

`FrankaSharpaForceCriticHorizonDeployEnvV3.compute_observations()` builds the
actor obs from the **MANO** demo keys:
    demo_data["wrist_pos"], ["wrist_rot"], ["wrist_velocity"],
    ["wrist_angular_velocity"], ["mano_joints"], ["mano_joints_velocity"]

But commit `0179ef5` ("Change actor observation reference from MANO to
retargeted joints") switched the **training** env
(`franka_sharpa_force_critic_horizon_env.py`) to the retargeted references:
    demo_data["target_wrist_pos"], ["target_wrist_rot"],
    ["target_wrist_velocity"], ["target_wrist_angular_velocity"],
    ["target_joints_pos"], ["target_joints_velocity"]
— and never updated the V3 deploy env.

`target_wrist_pos` = `opt_wrist_pos` (the retargeted Sharpa-hand wrist);
`wrist_pos` = the raw MANO (human-hand) wrist. They differ by the human→Sharpa
retargeting geometry. So the deployed policy was fed a wrist / hand reference
it was never trained on → `delta_wrist_pos` / `delta_joints_pos` wrong → the
arm drifted from the demo and the grasp landed offset from the object
(observed: arm 0.007→0.15 rad off demo over a rollout; hand ~10 cm above obj).

## The fix

Without touching the V3 obs code: right after the demo data is built, point
the MANO key names that V3 reads at the retargeted `target_*` arrays, so deploy
obs matches training exactly. `obj_trajectory` / `tips_distance` / `seq_len`
are unchanged (identical in both train and deploy).
"""
from __future__ import annotations

from .franka_sharpa_force_poseobs_deploy_env import FrankaSharpaForcePoseObsDeployEnv
from .franka_sharpa_force_poseobs_cfg import FrankaSharpaPoseObsCfg


class FrankaSharpaForcePoseObsDeployEnvV2(FrankaSharpaForcePoseObsDeployEnv):
    """PoseObs deploy env with the train/deploy reference-key mismatch fixed."""

    cfg: FrankaSharpaPoseObsCfg

    # MANO key (read by V3.compute_observations) -> retargeted key the policy
    # was actually trained on.
    _REF_REMAP = [
        ("wrist_pos",              "target_wrist_pos"),
        ("wrist_velocity",         "target_wrist_velocity"),
        ("wrist_rot",              "target_wrist_rot"),
        ("wrist_angular_velocity", "target_wrist_angular_velocity"),
        ("mano_joints",            "target_joints_pos"),
        ("mano_joints_velocity",   "target_joints_velocity"),
    ]

    def __init__(self, cfg: FrankaSharpaPoseObsCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # demo_data is fully built by now (the base __init__ chain ran). Repoint
        # the MANO keys V3.compute_observations() reads at the retargeted refs.
        remapped, missing = [], []
        for mano_key, target_key in self._REF_REMAP:
            if target_key in self.demo_data and mano_key in self.demo_data:
                self.demo_data[mano_key] = self.demo_data[target_key]
                remapped.append(f"{mano_key}<-{target_key}")
            else:
                missing.append(f"{mano_key}/{target_key}")

        self.get_logger().info(
            f"[PoseObsDeployV2] reference-key fix: {len(remapped)} keys repointed "
            f"to retargeted refs (deploy obs now matches training): {remapped}"
        )
        if missing:
            self.get_logger().warn(
                f"[PoseObsDeployV2] could NOT remap (key missing in demo_data): "
                f"{missing} — deploy obs may still mismatch training for these."
            )
