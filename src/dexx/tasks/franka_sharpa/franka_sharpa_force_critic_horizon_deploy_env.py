# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Deploy counterpart of FrankaSharpaForceCriticHorizonEnv.

Mirrors the actor-side obs additions (target_obj_pos, target_obj_quat,
tips_distance — all read from demo pkl, so available at deploy). Priv_info_buf
is NOT used by act_inference, but allocated via cfg.priv_info_dim=148 so shapes
stay consistent between training and deploy checkpoints.
"""
from __future__ import annotations

import torch
import gymnasium as gym

from isaaclab.envs.utils.spaces import spec_to_gym_space
from isaaclab.utils.math import quat_conjugate, quat_mul

from dexx.tasks.hand_imitation.dataset.transform import aa_to_quat

from .franka_sharpa_env import rotmat_to_quat
from .franka_sharpa_force_deploy_env_v2 import FrankaSharpaForceDeployEnvV2
from .franka_sharpa_critic_horizon_cfg import FrankaSharpaCriticHorizonCfg


class FrankaSharpaForceCriticHorizonDeployEnv(FrankaSharpaForceDeployEnvV2):
    cfg: FrankaSharpaCriticHorizonCfg

    def __init__(self, cfg: FrankaSharpaCriticHorizonCfg, render_mode: str | None = None, **kwargs):
        # Parent deploy env calls ForceEnv.__init__ which may run asymmetric_ac
        # reduce (obs 543 -> 410). We extend afterward — same pattern as training
        # variant, but here actor obs must match the training one exactly.
        super().__init__(cfg, render_mode, **kwargs)

        _asymmetric_ac = getattr(self.cfg, 'asymmetric_ac', False)
        _enable_bps = getattr(self.cfg, 'enable_bps', True)
        bps_dim = self.obj_bps.shape[-1] if (self.obj_bps is not None and _enable_bps) else 0
        if _asymmetric_ac:
            new_obs_dim = 410 + 12 + bps_dim                 # 422 or 550
        else:
            # parent obs-dim already reflects enable_bps
            new_obs_dim = self.cfg.observation_space + 7     # 422 or 550

        self.cfg.observation_space = new_obs_dim
        self.num_obs = new_obs_dim
        self.single_observation_space["policy"] = spec_to_gym_space(new_obs_dim)
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space["policy"], self.num_envs
        )

        self.proprio_hist_dim = new_obs_dim // 3
        self.obs_buf_lag_history = torch.zeros(
            (self.num_envs, max(80, self.cfg.prop_hist_len + 10), self.proprio_hist_dim),
            device=self.device, dtype=torch.float,
        )
        self.proprio_hist_buf = torch.zeros(
            (self.num_envs, self.cfg.prop_hist_len, self.proprio_hist_dim),
            device=self.device, dtype=torch.float,
        )

        print(
            f"[CriticHorizonDeploy] asymmetric_ac={_asymmetric_ac}, "
            f"obs_dim={new_obs_dim}, priv_info_dim={self.cfg.priv_info_dim}"
        )

    def compute_observations(self):
        """Forked from FrankaSharpaForceDeployEnvV2.compute_observations
        with the same actor-side additions as the training variant."""
        self._refresh_lab()

        # ---- Proprioception (unchanged from parent deploy) ----
        q = self.hand_dof_pos
        proprioception_obs = torch.cat([
            q, torch.cos(q), torch.sin(q),
            torch.cat([
                torch.zeros_like(self.base_pos), self.base_quat,
                self.base_lin_vel, self.base_ang_vel,
            ], dim=-1),
        ], dim=-1)

        # ---- Future indices ----
        obs_future_length = self.obs_future_length
        if self.loop_trajectory:
            seq_len = self.demo_data["seq_len"]
            future_indices = torch.stack(
                [(self._get_demo_idx() + 1 + t) % seq_len for t in range(obs_future_length)], dim=-1
            )
        else:
            cur_idx = torch.clamp(
                self.progress_buf + 1,
                torch.zeros_like(self.demo_data["seq_len"]),
                self.demo_data["seq_len"] - 1,
            )
            future_indices = torch.stack([cur_idx + t for t in range(obs_future_length)], dim=-1)
        nE, nT = self.demo_data["wrist_pos"].shape[:2]
        nF = obs_future_length

        def indicing(data, idx):
            assert data.shape[0] == nE and data.shape[1] == nT
            remaining_shape = data.shape[2:]
            expanded_idx = idx
            for _ in remaining_shape:
                expanded_idx = expanded_idx.unsqueeze(-1)
            expanded_idx = expanded_idx.expand(-1, -1, *remaining_shape)
            return torch.gather(data, 1, expanded_idx)

        # ---- Wrist target (unchanged) ----
        target_wrist_pos = indicing(self.demo_data["wrist_pos"], future_indices)
        delta_wrist_pos = (target_wrist_pos - self.base_pos[:, None]).reshape(nE, -1)

        target_wrist_vel = indicing(self.demo_data["wrist_velocity"], future_indices)
        wrist_vel = target_wrist_vel.reshape(nE, -1)
        delta_wrist_vel = (target_wrist_vel - self.base_lin_vel[:, None]).reshape(nE, -1)

        target_wrist_rot_raw = indicing(self.demo_data["wrist_rot"], future_indices)
        if target_wrist_rot_raw.ndim > 3:
            target_wrist_rot_raw = target_wrist_rot_raw[:, :, 0, :]
        target_wrist_quat = aa_to_quat(target_wrist_rot_raw.reshape(nE * nF, -1))
        delta_wrist_quat = quat_mul(
            self.base_quat[:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
            quat_conjugate(target_wrist_quat),
        ).reshape(nE, -1)
        wrist_quat = target_wrist_quat.reshape(nE, -1)

        target_wrist_ang_vel_raw = indicing(self.demo_data["wrist_angular_velocity"], future_indices)
        if target_wrist_ang_vel_raw.ndim > 3:
            target_wrist_ang_vel_raw = target_wrist_ang_vel_raw[:, :, 0, :]
        wrist_ang_vel = target_wrist_ang_vel_raw.reshape(nE, -1)
        delta_wrist_ang_vel = (target_wrist_ang_vel_raw - self.base_ang_vel[:, None]).reshape(nE, -1)

        # ---- Joints target (unchanged) ----
        target_joints_pos = indicing(self.demo_data["mano_joints"], future_indices).reshape(nE, nF, -1, 3)
        cur_joint_pos = self.hand.data.body_pos_w[:, self.hand_body_indices[1:]] - self.scene.env_origins.unsqueeze(1)
        delta_joints_pos = (target_joints_pos - cur_joint_pos[:, None]).reshape(self.num_envs, -1)

        target_joints_vel = indicing(self.demo_data["mano_joints_velocity"], future_indices).reshape(nE, nF, -1, 3)
        cur_joint_vel = self.hand.data.body_lin_vel_w[:, self.hand_body_indices[1:]]
        joints_vel = target_joints_vel.reshape(self.num_envs, -1)
        delta_joints_vel = (target_joints_vel - cur_joint_vel[:, None]).reshape(self.num_envs, -1)

        target_obs_list = [
            delta_wrist_pos, wrist_vel, delta_wrist_vel,
            wrist_quat, delta_wrist_quat, wrist_ang_vel, delta_wrist_ang_vel,
            delta_joints_pos, joints_vel, delta_joints_vel,
        ]

        _asymmetric_ac = getattr(self.cfg, 'asymmetric_ac', False)
        has_object = hasattr(self, 'object') and self.object is not None

        # ---- NEW: target_obj_pos / quat — deployable (read from demo pkl) ----
        if has_object:
            target_obj_transf = indicing(self.demo_data["obj_trajectory"], future_indices)  # [B, nF, 4, 4]
            target_obs_list.append(target_obj_transf[:, :, :3, 3].reshape(nE, -1))
            target_obs_list.append(
                rotmat_to_quat(target_obj_transf.reshape(-1, 4, 4)[:, :3, :3]).reshape(nE, -1)
            )
            # tips_distance always visible to actor (deployable from demo)
            gt_tips_distance = indicing(self.demo_data["tips_distance"], future_indices).reshape(nE, -1)
            target_obs_list.append(gt_tips_distance)

        # BPS is a deployable actor obs, gated by cfg.enable_bps (mirror training).
        if self.obj_bps is not None and getattr(self.cfg, 'enable_bps', True):
            target_obs_list.append(self.obj_bps)

        # ---- REAL tactile data (replaces sim contact sensor) ----
        sensed_contacts = self.last_contacts.clone().reshape(nE, -1)
        target_obs_list.append(sensed_contacts)

        contact_pos = (
            self._last_tactile_pos.clone().reshape(nE, -1)
            if hasattr(self, '_last_tactile_pos')
            else torch.zeros(nE, 15, device=self.device)
        )
        target_obs_list.append(contact_pos)

        # Combine
        target_obs = torch.cat(target_obs_list, dim=-1)
        obs_buf = torch.cat([proprioception_obs, target_obs], dim=-1)

        # History buffer for ProprioAdapt compat
        obs_part_for_hist = obs_buf[:, :self.proprio_hist_dim]
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        cur_obs_buf = obs_part_for_hist.unsqueeze(1)
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)
        if self.cfg.prop_hist_len > 0:
            self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.cfg.prop_hist_len:].clone()

        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(at_reset_env_ids) > 0:
            reset_obs = obs_part_for_hist[at_reset_env_ids]
            self.obs_buf_lag_history[at_reset_env_ids] = reset_obs.unsqueeze(1).repeat(
                1, self.obs_buf_lag_history.shape[1], 1)
            self.proprio_hist_buf[at_reset_env_ids] = reset_obs.unsqueeze(1).repeat(
                1, self.cfg.prop_hist_len, 1)

        # priv_info_buf: not consumed by act_inference at deploy; keep the
        # parent's partial filling (dq + current object state) for consistency
        # with training, new K-frame slots stay zero.
        self.priv_info_buf[:, :22] = self.hand_dof_vel
        if has_object:
            self.priv_info_buf[:, 27:30] = self.object_pos
            self.priv_info_buf[:, 30:34] = self.object_rot
            self.priv_info_buf[:, 34:40] = self.object_velocities

        if not hasattr(self, '_obs_dim_checked_ch_deploy'):
            actual_dim = obs_buf.shape[-1]
            expected_dim = self.cfg.observation_space
            print(
                f"[CriticHorizonDeploy] obs_buf={actual_dim} (expected={expected_dim}, "
                f"diff={actual_dim - expected_dim}), priv_info={self.priv_info_buf.shape[-1]}"
            )
            self._obs_dim_checked_ch_deploy = True

        return obs_buf
