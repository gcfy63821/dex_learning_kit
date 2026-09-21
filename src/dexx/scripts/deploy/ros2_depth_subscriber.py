"""ROS2 depth subscriber for RealSense D455 — deploy-side scene-PC source.

Reads `/camera/depth/image_rect_raw` (16UC1, depth in millimeters by default)
or `/camera/aligned_depth_to_color/image_raw` if you're using color-aligned
depth. Stores the latest depth frame as fp32 meters torch tensor on the
configured device (cpu or cuda).

Usage:
    from dexx.scripts.deploy.ros2_depth_subscriber import RealSenseDepthSubscriber
    sub = RealSenseDepthSubscriber(
        node_name='realsense_depth_sub',
        topic='/camera/aligned_depth_to_color/image_raw',
        height=240, width=320, device='cuda', namespace='',
    )
    rclpy.spin_until_future_complete(...)  # or run a separate executor
    depth_m = sub.get_latest()  # fp32 (H, W) torch on device; or None if no frame yet

Note: real depth pipeline outside this file (back-project → workspace crop →
subsample) is in `dexx/tasks/franka_sharpa/pointcloud/depth_to_pointcloud.py`
and `dexx/scripts/deploy/convert_real_depth_to_pc.py` (offline).
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image


# Sim camera intrinsics — single source of truth: dexx.deploy_config.
# REAL intrinsics must match these for sim2real PC alignment.
from dexx import deploy_config as _dcfg
SIM_INTRINSICS = {
    **_dcfg.SIM_INTRINSICS,
    "height": _dcfg.DEPTH_H, "width": _dcfg.DEPTH_W,
}

# Camera mount extrinsic (D455 in arm-base frame).
#
# Source of truth = `VisualRaycaster._build_camera_extrinsics()` in
# `dexx/tasks/franka_sharpa/visual_raycaster.py` (the actual matrix
# the sim renders with). User confirmed (2026-05-24) the real camera mount
# matches sim, so this is the single calibrated extrinsic.
#
# Calibration history (per visual_raycaster.py):
#  - 2026-05-19: manual hand-tune via calibrate_extrinsic_viz.py
#  - 2026-05-20: refined via viser sim↔real overlay (calibrate_real_extrinsic_viser.py)
# The shipped calibration, loaded from calib/camera_align/ — not a literal.
# This module used to carry its own copy of a 2026-05-20 matrix, identical to a
# second copy in visual_raycaster.py. Both went stale when the camera moved.
DEFAULT_T_CAM_IN_ARMBASE = _dcfg.default_camera_extrinsic()

# arm_base in env-local frame. Read from deploy_config, never copied: this line
# held a literal 0.415 while ARM_BASE_Z went 0.415 -> 0.432 -> 0.415, and was
# correct again only by coincidence.
ARM_BASE_POS_IN_ENV_LOCAL = _dcfg.arm_base_pos_np()


class RealSenseDepthSubscriber(Node):
    """ROS2 subscriber for RealSense depth topic.

    Stores the latest depth frame thread-safely. Call `get_latest()` to read.

    Depth msg format expected:
      - encoding: 16UC1 (raw depth in mm) — converted to fp32 meters
      - OR encoding: 32FC1 (already meters)
    """

    def __init__(
        self,
        node_name: str = 'realsense_depth_sub',
        topic: str = '/camera/aligned_depth_to_color/image_raw',
        height: int = 240,
        width: int = 320,
        device: str = 'cpu',
        namespace: str = '',
        depth_unit_to_m: float = 1.0e-3,  # 16UC1 → meters
    ):
        super().__init__(node_name, namespace=namespace)
        self._topic = topic
        self._H = int(height)
        self._W = int(width)
        self._device = torch.device(device)
        self._unit_to_m = float(depth_unit_to_m)

        self._lock = threading.Lock()
        self._latest: Optional[torch.Tensor] = None
        self._latest_stamp: Optional[float] = None
        self._n_received = 0

        # RealSense depth is typically BestEffort + small depth queue.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._sub = self.create_subscription(Image, topic, self._cb, qos)
        self.get_logger().info(
            f'[depth_sub] subscribing {topic} → ({height}x{width}) on {device}'
        )

    def _cb(self, msg: Image):
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            if msg.encoding == '16UC1':
                # 2 bytes per pixel
                depth_raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                depth_m = depth_raw.astype(np.float32) * self._unit_to_m
            elif msg.encoding == '32FC1':
                depth_m = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            else:
                self.get_logger().warn(f'[depth_sub] unsupported encoding: {msg.encoding}')
                return

            # Crude resize if camera output ≠ expected. Use nearest to keep
            # depth values sharp. Production: replace with a proper resizer.
            if depth_m.shape != (self._H, self._W):
                # Center-crop or zero-pad to expected size. Cheap version: nearest indexing.
                import cv2
                depth_m = cv2.resize(depth_m, (self._W, self._H), interpolation=cv2.INTER_NEAREST)

            depth_t = torch.from_numpy(depth_m).to(self._device, dtype=torch.float32)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            with self._lock:
                self._latest = depth_t
                self._latest_stamp = stamp
                self._n_received += 1
        except Exception as e:
            self.get_logger().error(f'[depth_sub] cb error: {e}')

    def get_latest(self) -> Optional[torch.Tensor]:
        """Return latest depth frame as fp32 (H, W) on configured device.
        None if no frame received yet."""
        with self._lock:
            return self._latest

    def get_latest_with_stamp(self) -> tuple[Optional[torch.Tensor], Optional[float]]:
        with self._lock:
            return self._latest, self._latest_stamp

    @property
    def n_received(self) -> int:
        return self._n_received
