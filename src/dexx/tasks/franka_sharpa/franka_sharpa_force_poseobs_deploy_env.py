# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Deploy counterpart of `franka-sharpa-force-poseobs`.

Inherits the V3 pkfk deploy env (already proven 30Hz, end-to-end working) and
appends the 7-d FoundationPose-observed object pose at the tail of the obs.
The pose comes from a ROS2 PoseStamped topic, transformed to the robot base
frame if needed (via TF). If the publisher is stale or never received, we
fall back to the demo's GT obj_trajectory (so the env still produces a
plausible obs and the policy doesn't blow up before the camera comes online).

Coordinate frame: world / env-local (= base frame at num_envs=1). This is the
same frame the training env uses for `self.object_pos` / `self.object_rot`,
so no coordinate transform is needed inside the env.

ROS2 wiring:
  Topic:    cfg.foundationpose_topic            (default: /foundationpose/object_pose)
  Type:     geometry_msgs/PoseStamped
  Frame:    cfg.foundationpose_base_frame       (default: fr3_link0)
  Max age:  cfg.foundationpose_pose_max_age     (default: 0.20 sec)
"""
from __future__ import annotations

import os

import numpy as np
import torch
import gymnasium as gym

from isaaclab.envs.utils.spaces import spec_to_gym_space

from .franka_sharpa_force_critic_horizon_deploy_env_v3 import (
    FrankaSharpaForceCriticHorizonDeployEnvV3,
)
from .franka_sharpa_env import rotmat_to_quat
from .franka_sharpa_force_poseobs_cfg import FrankaSharpaPoseObsCfg


class FrankaSharpaForcePoseObsDeployEnv(FrankaSharpaForceCriticHorizonDeployEnvV3):
    """V3 pkfk deploy + FoundationPose obj_pose obs at the tail."""

    cfg: FrankaSharpaPoseObsCfg

    def __init__(self, cfg: FrankaSharpaPoseObsCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._is_deploy_env = True

        # Extend obs by 7. Same logic as the training env: do NOT touch the
        # proprio history slice (student only consumes that, and we're keeping
        # it identical to training).
        parent_obs_dim = self.cfg.observation_space
        new_obs_dim = parent_obs_dim + 7

        self.cfg.observation_space = new_obs_dim
        self.num_obs = new_obs_dim
        self.single_observation_space["policy"] = spec_to_gym_space(new_obs_dim)
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space["policy"], self.num_envs
        )

        # Lazy import so non-deploy users don't trip on rclpy / tf2 missing.
        from dexx.scripts.deploy.foundationpose_subscriber import (
            FoundationPoseSubscriber,
        )
        self._fp_sub = FoundationPoseSubscriber(
            topic=self.cfg.foundationpose_topic,
            base_frame=self.cfg.foundationpose_base_frame,
            max_age=self.cfg.foundationpose_pose_max_age,
            device=self.device,
            ros_init=False,   # rclpy already initialized by the deploy entrypoint
        )

        # Last-good pose (held across stale frames). Initialized to identity
        # quat at origin; will be overwritten on the first successful pose msg.
        self._last_fp_pose = torch.zeros(7, device=self.device, dtype=torch.float)
        self._last_fp_pose[3] = 1.0   # w of identity quat

        # Stats for logging
        self._fp_pose_uses = 0
        self._fp_pose_stale_falls = 0

        self.get_logger().info(
            f"[PoseObsDeploy] obs_dim={new_obs_dim} (parent={parent_obs_dim}+7), "
            f"topic={self.cfg.foundationpose_topic}, base={self.cfg.foundationpose_base_frame}"
        )

        # --- RViz object markers ---
        #   /demo_initial_object  (latched) — where to place the object before a rollout
        #   /demo_object_live     (30 Hz)   — demo-expected vs actual object pose
        # Object drawn as its real mesh (MESH_RESOURCE) + an XYZ axis triad.
        self._arm_base_pos_t = torch.tensor(
            self.cfg.arm_base_pos, device=self.device, dtype=torch.float32
        )
        self._demo_marker_start_idx = 0
        self._demo_marker_pub = None
        self._live_marker_pub = None
        self._obj_mesh_uri = self._resolve_obj_mesh()
        self._obj_symmetry_rotmats = self._object_symmetry_rotmats()
        self.get_logger().info(
            f"[DemoObjMarker] object mesh: "
            f"{self._obj_mesh_uri or 'NOT FOUND → CUBE fallback'}"
        )
        self.get_logger().info(
            f"[PoseObsDeploy] object symmetry group: "
            f"{len(self._obj_symmetry_rotmats)} rotation(s) — "
            f"{'FP orientation snapped to demo' if len(self._obj_symmetry_rotmats) > 1 else 'FP orientation used as-is'}"
        )
        try:
            from rclpy.qos import QoSProfile, QoSDurabilityPolicy
            from visualization_msgs.msg import MarkerArray

            qos = QoSProfile(depth=1)
            qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL   # latched
            self._demo_marker_pub = self.ros2_action_publisher.create_publisher(
                MarkerArray, "/demo_initial_object", qos
            )
            self._live_marker_pub = self.ros2_action_publisher.create_publisher(
                MarkerArray, "/demo_object_live", 10
            )
            # Publish frame-0 pose immediately (default deploy start frame) so
            # the marker is visible during env startup before the first reset.
            self._publish_demo_object_marker(frame_idx=0)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"[DemoObjMarker] disabled: {e}")

    # ------------------------------------------------------------------
    # RViz target-object marker
    # ------------------------------------------------------------------
    def _resolve_obj_mesh(self):
        """Resolve the demo object's mesh file → a `file://` URI for a
        MESH_RESOURCE marker. `obj_urdf_path` sits next to a `cleaned_mesh_*.obj`.
        Handles repo-relative paths and absolute paths baked on another machine
        (re-roots at the `data/` segment under cwd). None → CUBE fallback."""
        raw = []
        override = getattr(self.cfg, "demo_marker_mesh", "")
        if override:
            raw.append(str(override))
        try:
            urdf = self.demo_data["obj_urdf_path"][0]
            if isinstance(urdf, str) and urdf:
                raw.append(urdf)
        except Exception:  # noqa: BLE001
            pass

        cands = []
        for p in raw:
            p = str(p).replace("\\", "/")
            stem = os.path.splitext(os.path.basename(p))[0]
            dirs = []
            if os.path.dirname(p):
                dirs.append(os.path.dirname(p))
            k = p.find("data/")
            if k >= 0:                       # re-root under the current repo
                dirs.append(os.path.dirname(os.path.join(os.getcwd(), p[k:])))
            for d in dirs:
                cands.append(os.path.join(d, stem + ".obj"))
                cands.append(os.path.join(d, "cleaned_mesh_10000.obj"))
        for c in cands:
            ap = os.path.abspath(c)
            if os.path.isfile(ap):
                return "file://" + ap
        return None

    @staticmethod
    def _quat_wxyz_to_rotmat(q):
        """[w, x, y, z] → 3x3 numpy rotation matrix."""
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        n = (w * w + x * x + y * y + z * z) ** 0.5
        if n < 1e-9:
            return np.eye(3)
        w, x, y, z = w / n, x / n, y / n, z / n
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
            [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
        ])

    @staticmethod
    def _rotmat_to_quat_wxyz(R):
        """3x3 numpy rotation matrix → [w, x, y, z] numpy quaternion."""
        t = R[0, 0] + R[1, 1] + R[2, 2]
        if t > 0.0:
            s = 0.5 / np.sqrt(t + 1.0)
            w, x, y, z = 0.25 / s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s
        elif R[0,0] >= R[1,1] and R[0,0] >= R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            w, x, y, z = (R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
        elif R[1,1] >= R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            w, x, y, z = (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            w, x, y, z = (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s
        q = np.array([w, x, y, z], dtype=np.float64)
        return q / (np.linalg.norm(q) + 1e-12)

    @staticmethod
    def _cube_rotation_group():
        """The 24 proper rotations of a cube = signed 3x3 permutation matrices
        with determinant +1 (the octahedral rotation group)."""
        import itertools

        mats = []
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product((1.0, -1.0), repeat=3):
                M = np.zeros((3, 3))
                for r, c in enumerate(perm):
                    M[r, c] = signs[r]
                if round(float(np.linalg.det(M))) == 1:
                    mats.append(M)
        return mats   # 24

    def _object_symmetry_rotmats(self):
        """Per-object proper-rotation symmetry group (list of 3x3 numpy mats).
        `cube_small` → full 24-element cube group; everything else → trivial
        {I} (FP orientation used as-is). Category resolved from data_indices[0];
        `cfg.obj_symmetry` ('cube' | 'none') overrides."""
        cat = ""
        try:
            idx = str(self.data_indices[0])
            exp = idx.split("@")[0].rstrip("/").split("/")[-1]
            head, _, tail = exp.rpartition("_")
            cat = head if (head and tail.isdigit()) else exp
        except Exception:  # noqa: BLE001
            pass
        sym = getattr(self.cfg, "obj_symmetry", "") or (
            "cube" if "cube" in cat else "none")
        if sym == "cube":
            return self._cube_rotation_group()
        return [np.eye(3)]

    def _snap_orientation_to_demo(self, quat_fp_wxyz):
        """Resolve FoundationPose's symmetry-flip ambiguity (e.g. cube z-down vs
        z-up): return the member of the FP orientation's symmetry orbit closest
        to the demo's expected orientation at the current frame. Stable across
        FP flips. quat in/out: [w,x,y,z] torch tensor. R_fp (fr3_link0) and
        R_demo (env-local) compare directly — arm_base_rot is identity. Trivial
        symmetry group → passthrough."""
        G = getattr(self, "_obj_symmetry_rotmats", None)
        if not G or len(G) <= 1:
            return quat_fp_wxyz
        try:
            seq_len = int(self.demo_data["seq_len"][0].item())
            idx = max(0, min(int(self.progress_buf[0].item()), seq_len - 1))
            R_demo = self.demo_data["obj_trajectory"][0, idx, :3, :3].detach().cpu().numpy()
            R_fp = self._quat_wxyz_to_rotmat(quat_fp_wxyz.detach().cpu().numpy())
            best_R, best_score = None, -1e18
            for S in G:
                cand = R_fp @ S                              # FP orbit member
                score = float(np.trace(cand.T @ R_demo))     # larger = closer to demo
                if score > best_score:
                    best_score, best_R = score, cand
            q = self._rotmat_to_quat_wxyz(best_R)
            return torch.tensor(q, dtype=quat_fp_wxyz.dtype, device=quat_fp_wxyz.device)
        except Exception:  # noqa: BLE001
            return quat_fp_wxyz

    def _obj_shape_marker(self, ns, mid, pos, quat_wxyz, rgba, stamp):
        """Object marker — real mesh (MESH_RESOURCE) if resolved, else a CUBE.
        pos in fr3_link0; quat_wxyz = [w, x, y, z]."""
        from visualization_msgs.msg import Marker

        m = Marker()
        m.header.frame_id = self.cfg.foundationpose_base_frame
        m.header.stamp = stamp
        m.ns = ns
        m.id = mid
        m.action = Marker.ADD
        m.pose.position.x = float(pos[0])
        m.pose.position.y = float(pos[1])
        m.pose.position.z = float(pos[2])
        m.pose.orientation.x = float(quat_wxyz[1])
        m.pose.orientation.y = float(quat_wxyz[2])
        m.pose.orientation.z = float(quat_wxyz[3])
        m.pose.orientation.w = float(quat_wxyz[0])
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        if self._obj_mesh_uri:
            sc = float(getattr(self.cfg, "demo_marker_mesh_scale", 1.0))
            m.type = Marker.MESH_RESOURCE
            m.mesh_resource = self._obj_mesh_uri
            m.mesh_use_embedded_materials = False
            m.scale.x = m.scale.y = m.scale.z = sc
        else:
            s = float(getattr(self.cfg, "demo_marker_size", 0.06))
            m.type = Marker.CUBE
            m.scale.x = m.scale.y = m.scale.z = s
        return m

    def _axes_markers(self, ns, base_id, pos, R, stamp, length=0.08):
        """XYZ axis triad (3 ARROW markers: X red, Y green, Z blue) at the
        object pose. pos: fr3_link0 numpy(3); R: numpy 3x3 rotation."""
        from visualization_msgs.msg import Marker
        from geometry_msgs.msg import Point

        rgb = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.4, 1.0)]
        out = []
        for i in range(3):
            m = Marker()
            m.header.frame_id = self.cfg.foundationpose_base_frame
            m.header.stamp = stamp
            m.ns = ns + "_axes"
            m.id = base_id + i
            m.type = Marker.ARROW
            m.action = Marker.ADD
            end = pos + R[:, i] * length
            m.points = [
                Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                Point(x=float(end[0]), y=float(end[1]), z=float(end[2])),
            ]
            m.scale.x = 0.006   # shaft diameter
            m.scale.y = 0.013   # head diameter
            m.scale.z = 0.0     # head length (0 → auto)
            m.pose.orientation.w = 1.0
            m.color.r, m.color.g, m.color.b = rgb[i]
            m.color.a = 0.9
            out.append(m)
        return out

    def _publish_live_object_markers(self, actual_pose_envlocal: "torch.Tensor"):
        """Publish /demo_object_live every step — demo-expected (blue) vs actual
        (orange) object pose, each as mesh + axis triad. actual_pose_envlocal:
        (7,) [pos3, quat_wxyz] env-local. try/except — viz never breaks control.
        """
        if getattr(self, "_live_marker_pub", None) is None:
            return
        if not hasattr(self, "demo_data") or "obj_trajectory" not in self.demo_data:
            return
        try:
            from visualization_msgs.msg import MarkerArray

            traj = self.demo_data["obj_trajectory"]
            seq_len = int(self.demo_data["seq_len"][0].item())
            idx = max(0, min(int(self.progress_buf[0].item()), seq_len - 1))
            T = traj[0, idx]
            d_pos = (T[:3, 3] - self._arm_base_pos_t).cpu().numpy()
            d_R = T[:3, :3].cpu().numpy()
            d_quat = rotmat_to_quat(T[:3, :3].unsqueeze(0))[0].cpu().numpy()

            act = actual_pose_envlocal.detach()
            a_pos = (act[:3] - self._arm_base_pos_t).cpu().numpy()
            a_quat = act[3:7].cpu().numpy()   # wxyz
            a_R = self._quat_wxyz_to_rotmat(a_quat)

            stamp = self.ros2_action_publisher.get_clock().now().to_msg()
            arr = MarkerArray()
            arr.markers.append(self._obj_shape_marker(
                "demo_expected", 0, d_pos, d_quat, (0.1, 0.4, 1.0, 0.45), stamp))
            arr.markers.extend(self._axes_markers(
                "demo_expected", 10, d_pos, d_R, stamp))
            arr.markers.append(self._obj_shape_marker(
                "obj_actual", 0, a_pos, a_quat, (1.0, 0.5, 0.0, 0.45), stamp))
            arr.markers.extend(self._axes_markers(
                "obj_actual", 10, a_pos, a_R, stamp))
            self._live_marker_pub.publish(arr)
        except Exception:  # noqa: BLE001
            pass
    def _publish_demo_object_marker(self, frame_idx: int | None = None):
        """Publish a MarkerArray marking where the demo expects the object at
        its start frame, so the operator can place the real object there.

        Frame: cfg.foundationpose_base_frame (= fr3_link0, the RViz fixed
        frame). demo `obj_trajectory` is env-local; convert to fr3_link0 by
        subtracting arm_base_pos — the exact inverse of the +arm_base_pos shift
        `_get_obj_pose_obs` applies to FoundationPose poses.
        """
        if self._demo_marker_pub is None:
            return
        if not hasattr(self, "demo_data") or "obj_trajectory" not in self.demo_data:
            return

        traj = self.demo_data["obj_trajectory"]              # (nE, T, 4, 4) env-local
        seq_len = int(self.demo_data["seq_len"][0].item())
        if frame_idx is None:
            frame_idx = int(self._demo_marker_start_idx)
        frame_idx = max(0, min(int(frame_idx), seq_len - 1))

        T = traj[0, frame_idx]                               # (4, 4)
        pos = (T[:3, 3] - self._arm_base_pos_t).cpu().numpy()            # → fr3_link0
        R = T[:3, :3].cpu().numpy()
        quat = rotmat_to_quat(T[:3, :3].unsqueeze(0))[0].cpu().numpy()

        from visualization_msgs.msg import Marker, MarkerArray

        frame = self.cfg.foundationpose_base_frame
        stamp = self.ros2_action_publisher.get_clock().now().to_msg()
        demo = str(getattr(self, "data_indices", ["?"])[0])
        arr = MarkerArray()

        # object mesh (green translucent) + XYZ axis triad
        arr.markers.append(self._obj_shape_marker(
            "demo_initial_object", 0, pos, quat, (0.0, 1.0, 0.2, 0.45), stamp))
        arr.markers.extend(self._axes_markers(
            "demo_initial_object", 10, pos, R, stamp))

        # text label above the object
        txt = Marker()
        txt.header.frame_id = frame
        txt.header.stamp = stamp
        txt.ns = "demo_initial_object"
        txt.id = 1
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position.x = float(pos[0])
        txt.pose.position.y = float(pos[1])
        txt.pose.position.z = float(pos[2]) + 0.12
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.03
        txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
        txt.text = f"PLACE OBJECT HERE\n{demo}  frame {frame_idx}"
        arr.markers.append(txt)

        self._demo_marker_pub.publish(arr)
        self.get_logger().info(
            f"[DemoObjMarker] target object @frame {frame_idx}: {frame} "
            f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})  "
            f"shape={'mesh' if self._obj_mesh_uri else 'CUBE-fallback'}"
        )

    def _reset_idx(self, env_ids):
        """Reset, then republish the target-object marker for the actual start
        frame (progress_buf after reset = the demo frame the rollout begins at;
        may differ from 0 if a start frame was selected)."""
        super()._reset_idx(env_ids)
        try:
            self._demo_marker_start_idx = int(self.progress_buf[0].item())
            self._publish_demo_object_marker(frame_idx=self._demo_marker_start_idx)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"[DemoObjMarker] reset publish failed: {e}")

    # ------------------------------------------------------------------
    # Resolve the object pose obs (7d). Priority:
    #   1. FoundationPose latest (if fresh enough)
    #   2. self._last_fp_pose (held across stale frames)
    #   3. Demo's GT obj_trajectory at current frame (cold-start fallback)
    #
    # Frame: training's obj_pose obs = self.object_pos = body_pos_w - env_origins
    # (env-local / "world", which has arm spawned at cfg.arm_base_pos).
    # FP subscriber returns pose in cfg.foundationpose_base_frame (= fr3_link0),
    # which is the arm root link. To match training, add arm_base_pos so the
    # pose moves into env-local. Rotation is unchanged (arm_base_rot=identity).
    # Without this fix, the policy sees obj Z ≈ 0.015 in fr3_link0 while training
    # had obj Z ≈ arm_base_pos.z + 0.015 ≈ 0.43 → completely out-of-distribution.
    # ------------------------------------------------------------------
    def _get_obj_pose_obs(self) -> torch.Tensor:
        # Cache arm_base_pos as a tensor (build lazily once).
        if not hasattr(self, "_arm_base_pos_t"):
            self._arm_base_pos_t = torch.tensor(
                self.cfg.arm_base_pos, device=self.device, dtype=torch.float32
            )

        def _to_env_local(pose7: torch.Tensor) -> torch.Tensor:
            """FP pose (fr3_link0) → policy obs (env-local): snap orientation to
            the demo's symmetry-resolved orientation (fixes FP's cube z-up/z-down
            flip ambiguity), then shift position by +arm_base_pos. Rotation frame
            is unchanged (arm_base_rot = identity)."""
            out = pose7.clone()
            out[:3] = pose7[:3] + self._arm_base_pos_t
            out[3:7] = self._snap_orientation_to_demo(pose7[3:7])
            return out

        pose = self._fp_sub.get_latest()
        if pose is not None:
            self._last_fp_pose = pose
            self._fp_pose_uses += 1
            return _to_env_local(pose).unsqueeze(0)   # (1, 7)

        # Stale or never received. If we have a last-good, use it. Otherwise
        # fall back to the demo's GT pose at the current frame so the obs is
        # at least plausible at startup (before FP catches up). Log first miss.
        self._fp_pose_stale_falls += 1
        if self._fp_pose_stale_falls % 100 == 1:
            self.get_logger().warn(
                f"[PoseObsDeploy] FoundationPose stale (count={self._fp_pose_stale_falls}, "
                f"received={self._fp_sub.msg_count}). Falling back to last-good / demo GT."
            )

        if torch.any(self._last_fp_pose[:3] != 0):
            return _to_env_local(self._last_fp_pose).unsqueeze(0)

        # Cold-start fallback: demo GT pose at current frame.
        if hasattr(self, "demo_data") and "obj_trajectory" in self.demo_data:
            traj = self.demo_data["obj_trajectory"]   # (nE, T, 4, 4)
            seq_len = self.demo_data["seq_len"]
            idx = torch.clamp(
                self.progress_buf,
                torch.zeros_like(seq_len),
                seq_len - 1,
            )
            cur_T = traj[torch.arange(self.num_envs, device=self.device), idx]  # (nE, 4, 4)
            pos = cur_T[:, :3, 3]                                # (nE, 3) env-local
            quat = rotmat_to_quat(cur_T[:, :3, :3])              # (nE, 4) wxyz
            return torch.cat([pos, quat], dim=-1)

        # Hard fallback: zeros + identity quat (should never reach in practice)
        out = torch.zeros((self.num_envs, 7), device=self.device)
        out[:, 3] = 1.0
        return out

    # ------------------------------------------------------------------
    # compute_observations: call parent V3, append 7d obj pose obs at tail.
    # ------------------------------------------------------------------
    def compute_observations(self):
        obs_buf = super().compute_observations()
        obj_pose_obs = self._get_obj_pose_obs()                  # (nE, 7)
        full_obs = torch.cat([obs_buf, obj_pose_obs], dim=-1)

        if not hasattr(self, "_poseobs_deploy_dim_checked"):
            self.get_logger().info(
                f"[PoseObsDeploy obs] dim={full_obs.shape[-1]} "
                f"(parent={obs_buf.shape[-1]} + poseobs=7), "
                f"cfg.observation_space={self.cfg.observation_space}"
            )
            assert full_obs.shape[-1] == self.cfg.observation_space, (
                f"obs dim mismatch: got {full_obs.shape[-1]} vs cfg "
                f"{self.cfg.observation_space}"
            )
            self._poseobs_deploy_dim_checked = True

        # Live RViz comparison markers (demo-expected vs actual object pose).
        self._publish_live_object_markers(obj_pose_obs[0])
        return full_obs

    # ------------------------------------------------------------------
    # Clean shutdown of the ROS2 subscriber thread.
    # ------------------------------------------------------------------
    def close(self):
        try:
            self._fp_sub.shutdown()
        except Exception:
            pass
        return super().close()
