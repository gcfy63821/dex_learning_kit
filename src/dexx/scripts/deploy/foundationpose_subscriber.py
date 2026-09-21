"""ROS2 subscriber for FoundationPose-style object pose tracker.

Listens to a PoseStamped (or PoseWithCovarianceStamped) topic and exposes the
latest pose in the robot base frame, with staleness check. If the publisher's
header.frame_id is not the base frame, looks up TF (camera→base extrinsic)
to transform once per message.

Intended pairing: `franka-sharpa-force-poseobs-deploy` task → this subscriber.

Usage (in deploy env):
    from dexx.scripts.deploy.foundationpose_subscriber import (
        FoundationPoseSubscriber,
    )
    self._fp_sub = FoundationPoseSubscriber(
        topic=self.cfg.foundationpose_topic,
        base_frame=self.cfg.foundationpose_base_frame,
        max_age=self.cfg.foundationpose_pose_max_age,
    )
    pose7 = self._fp_sub.get_latest()  # torch.Tensor (7,) wxyz+pos, or None

Coordinate convention:
    - Output is (7,) [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
      to match isaaclab's wxyz quat layout. NOTE: ROS Pose uses xyzw — the
      subscriber reorders on receive.
    - Output is in `base_frame` (default `fr3_link0`), which is what the env's
      env-local frame collapses to at num_envs=1.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import torch

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from rclpy.time import Time
    from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
    import tf2_ros
    from tf2_geometry_msgs import do_transform_pose
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "FoundationPoseSubscriber requires ROS2 + rclpy + tf2 + tf2_geometry_msgs. "
        "Source your ROS2 setup before deploying."
    ) from e


class FoundationPoseSubscriber(Node):
    """Thread-safe ROS2 subscriber that holds the latest FoundationPose pose.

    Spawns a daemon thread spinning the underlying rclpy executor so the parent
    process (the deploy env) can poll `get_latest()` synchronously.
    """

    def __init__(
        self,
        topic: str = "/foundationpose/object_pose",
        base_frame: str = "fr3_link0",
        max_age: float = 0.2,
        msg_type: str = "pose_stamped",
        device: str | torch.device = "cuda",
        ros_init: bool = True,
    ) -> None:
        if ros_init and not rclpy.ok():
            rclpy.init()
        super().__init__("foundationpose_subscriber")

        self._base_frame = base_frame
        self._max_age = float(max_age)
        self._device = torch.device(device)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        if msg_type == "pose_stamped":
            self._sub = self.create_subscription(PoseStamped, topic, self._cb_ps, qos)
        elif msg_type == "pose_with_cov":
            self._sub = self.create_subscription(
                PoseWithCovarianceStamped, topic, self._cb_pwc, qos,
            )
        else:
            raise ValueError(f"unknown msg_type={msg_type}")

        # TF for cam→base lookup if header.frame_id != base_frame.
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._lock = threading.Lock()
        self._latest: Optional[torch.Tensor] = None   # (7,) cpu fp32
        self._latest_stamp_sec: float = 0.0
        self._msg_count: int = 0
        self._tf_failures: int = 0

        # Spin in background. Use SingleThreadedExecutor so we can clean up.
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self)
        self._stop = threading.Event()
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

        self.get_logger().info(
            f"[FoundationPoseSubscriber] topic={topic}, base_frame={base_frame}, "
            f"max_age={max_age}s, msg_type={msg_type}"
        )

    # ------------------------------------------------------------------

    def _spin_loop(self) -> None:
        while not self._stop.is_set() and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.05)

    def shutdown(self) -> None:
        self._stop.set()
        try:
            self._executor.remove_node(self)
        except Exception:
            pass
        self.destroy_node()

    # ------------------------------------------------------------------
    # Message callbacks (PoseStamped and PoseWithCovarianceStamped variants).
    # ------------------------------------------------------------------

    def _cb_ps(self, msg: "PoseStamped") -> None:
        self._handle_pose(msg.header, msg.pose, msg.header.stamp)

    def _cb_pwc(self, msg: "PoseWithCovarianceStamped") -> None:
        # PoseWithCovarianceStamped → use the .pose.pose inner part.
        self._handle_pose(msg.header, msg.pose.pose, msg.header.stamp)

    def _handle_pose(self, header, pose, stamp) -> None:
        # Transform to base_frame if needed.
        if header.frame_id == self._base_frame or header.frame_id == "":
            pose_base = pose
        else:
            try:
                tf_msg = self._tf_buffer.lookup_transform(
                    self._base_frame,
                    header.frame_id,
                    Time(seconds=0, nanoseconds=0),  # latest
                )
            except Exception as e:
                self._tf_failures += 1
                if self._tf_failures % 50 == 1:
                    self.get_logger().warn(
                        f"[FoundationPose] TF lookup {header.frame_id}→"
                        f"{self._base_frame} failed (count={self._tf_failures}): {e}"
                    )
                return
            # do_transform_pose expects a Pose, returns a Pose. Wrap and unwrap.
            from geometry_msgs.msg import PoseStamped as _PS
            ps_in = _PS(); ps_in.pose = pose; ps_in.header = header
            ps_out = do_transform_pose(pose, tf_msg)
            pose_base = ps_out

        # Pack (7,) wxyz + pos. ROS uses xyzw → reorder.
        t = torch.tensor(
            [
                pose_base.position.x, pose_base.position.y, pose_base.position.z,
                pose_base.orientation.w, pose_base.orientation.x,
                pose_base.orientation.y, pose_base.orientation.z,
            ],
            dtype=torch.float32,
        )
        stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        with self._lock:
            self._latest = t
            self._latest_stamp_sec = stamp_sec
            self._msg_count += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_latest(self) -> Optional[torch.Tensor]:
        """Return latest pose (7,) on `self._device`, or None if stale/empty.

        Pose layout: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z].
        Staleness check uses wall-clock vs message header stamp (ROS time).
        If the message has stamp=0 (some publishers don't fill it), we fall
        back to checking the time since the message was received locally.
        """
        with self._lock:
            if self._latest is None:
                return None
            stamp = self._latest_stamp_sec
            cur = time.time()
            age = cur - stamp if stamp > 0 else 0.0
            if age > self._max_age and self._max_age > 0:
                return None
            return self._latest.to(self._device)

    @property
    def msg_count(self) -> int:
        return self._msg_count

    @property
    def tf_failures(self) -> int:
        return self._tf_failures
