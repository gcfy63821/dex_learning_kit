"""PointCloud student deploy env with a Polymetis arm backend.

Combines two existing deploy pieces:
  - `FrankaSharpaPointCloudDeployEnv` : scene/hand/tactile point-cloud obs from a
    RealSense depth source + `set_camera_extrinsic()` + Sharpa-hand SDK (parent).
  - Polymetis arm comm (as in `FrankaSharpaPolymetisDeployEnv`) : the arm is driven
    through a `PolymetisArmClient` (ZMQ -> NUC `polymetis_joint_bridge.py` ->
    RobotInterface joint impedance) instead of ROS2.

Only the arm communication layer changes; the whole PointCloud pipeline (depth ->
unproject -> workspace crop -> subsample -> policy) is inherited unchanged. The
swap is done by overriding `_setup_comm()` (extracted in
FrankaSharpaForceDeployEnvV2), binding the Polymetis client to both the obs
subscriber and action publisher roles (its interface is duck-typed to the ROS2
objects).

Deploy topology (see deploy_pc_polymetis.py):
    inference PC : this env + policy + Sharpa hand SDK
    NUC          : polymetis server + joint bridge -> FR3
    camera host  : realsense_depth_zmq_pub.py -> depth over ZMQ -> set_depth_source()

Polymetis connection params come from the env cfg (set by the deploy entry):
    cfg.polymetis_server_ip                    (NUC IP over the wired link)
    cfg.polymetis_state_port  (default 5560)
    cfg.polymetis_cmd_port    (default 5561)
    cfg.polymetis_kq / cfg.polymetis_kqd       (joint impedance gains; None -> Polymetis default)
"""

from __future__ import annotations

from .franka_sharpa_pointcloud_deploy_env import FrankaSharpaPointCloudDeployEnv


class FrankaSharpaPointCloudPolymetisDeployEnv(FrankaSharpaPointCloudDeployEnv):
    """PointCloud deploy env, arm driven through Polymetis (NUC) instead of ROS2."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        # Stash Polymetis params BEFORE super().__init__(), because the V2 base
        # __init__ calls self._setup_comm() before it assigns self.cfg.
        self._pm_ip = getattr(cfg, "polymetis_server_ip", "localhost")
        self._pm_state_port = int(getattr(cfg, "polymetis_state_port", _dcfg.POLYMETIS_STATE_PORT))
        self._pm_cmd_port = int(getattr(cfg, "polymetis_cmd_port", _dcfg.POLYMETIS_CMD_PORT))
        self._pm_kq = getattr(cfg, "polymetis_kq", None)
        self._pm_kqd = getattr(cfg, "polymetis_kqd", None)
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_comm(self, ros2_namespace: str):
        """Build the Polymetis arm client and bind it as the obs/action objects."""
        from dexx.tasks.hand_imitation.deploy.polymetis_arm_client import (
            PolymetisArmClient,
        )

        client = PolymetisArmClient(
            ip_address=self._pm_ip,
            state_port=self._pm_state_port,
            cmd_port=self._pm_cmd_port,
            kq=self._pm_kq,
            kqd=self._pm_kqd,
        )
        # One object plays both the ROS2ObservationSubscriber and
        # ROS2ActionPublisher roles (duck-typed).
        self.ros2_obs_subscriber = client
        self.ros2_action_publisher = client
        self.ros2_namespace = ros2_namespace  # kept for logging/state-print compat
        # No rclpy executor/spin thread — the client runs its own state poller.
        self.ros2_executor = None
        self.ros2_thread = None
        client.get_logger().info(
            f"[PC-PolymetisDeploy] arm backend = Polymetis @ {self._pm_ip}:"
            f"{self._pm_state_port}/{self._pm_cmd_port}"
        )
