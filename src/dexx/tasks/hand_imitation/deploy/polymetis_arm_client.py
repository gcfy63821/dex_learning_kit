"""Polymetis arm client (ZMQ) for the Sharpa deploy env.

Drop-in replacement for the ROS2 arm comm layer (ROS2ObservationSubscriber +
ROS2ActionPublisher). A single ``PolymetisArmClient`` instance duck-types BOTH.

Because Polymetis (py38 / torch 1.13) cannot coexist with Isaac Lab (py310 /
torch 2.7) in one env, this client does NOT import polymetis. Instead it talks
over ZMQ to ``polymetis_joint_bridge.py`` running on the NUC (in the working
polymetis env). The bridge owns the real Polymetis ``RobotInterface`` and runs
the joint-impedance loop; this client only:
    - SUB (conflate) the bridge's state stream  -> arm/wrist obs
    - PUSH joint targets / control commands to the bridge

    deploy env (this PC, py310)   --ZMQ over Ethernet-->   NUC bridge (py38)
        publish_arm_joint_pos ---- {"cmd":"joint_target"} --> update_desired_joint_positions
        state SUB (conflate)  <---- {joint_pos, joint_vel, ee_*} <-- get_joint_* / get_ee_pose

Interface exposed (matches what the deploy env reads/calls):
    read attrs : arm_joint_positions, arm_joint_velocities,
                 wrist_position, wrist_quaternion (w,x,y,z),
                 wrist_linear_velocity, wrist_angular_velocity,
                 arm_data_received, wrist_data_received,
                 wrist_state_topic_active, wrist_msg_count
    methods    : publish_arm_joint_pos(q7), publish_hand_joints(...) (no-op),
                 get_logger(), shutdown()
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
from dexx import deploy_config as _dcfg
import torch


def _make_logger(name: str = "PolymetisArmClient") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    if not hasattr(logger, "warn"):
        logger.warn = logger.warning  # type: ignore[attr-defined]
    return logger


# Optional: match the ROS2 DexhandJointImpedanceController gains. Default None ->
# the bridge lets Polymetis use its own config-tuned FR3 gains (recommended).
ROS2_EQUIVALENT_KQ = [200.0, 200.0, 200.0, 200.0, 100.0, 100.0, 50.0]
ROS2_EQUIVALENT_KQD = [20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 5.0]


class PolymetisArmClient:
    """ZMQ client to the NUC polymetis_joint_bridge."""

    def __init__(
        self,
        ip_address: str = "localhost",
        state_port: int = _dcfg.POLYMETIS_STATE_PORT,
        cmd_port: int = _dcfg.POLYMETIS_CMD_PORT,
        kq=None,
        kqd=None,
        start_impedance: bool = True,
        connect_timeout_s: float = 10.0,
        logger: logging.Logger | None = None,
        # kept for signature compatibility with the direct-polymetis version:
        port: int = None,
    ):
        self.logger = logger or _make_logger()
        self.ip_address = ip_address
        self.state_port = int(state_port)
        self.cmd_port = int(cmd_port)
        self.kq = list(kq) if kq is not None else None
        self.kqd = list(kqd) if kqd is not None else None

        # ---- state buffers (duck-type the ROS2 subscriber) ----
        self.arm_joint_positions = torch.zeros(7, dtype=torch.float32)
        self.arm_joint_velocities = torch.zeros(7, dtype=torch.float32)
        self.wrist_position = torch.zeros(3, dtype=torch.float32)
        self.wrist_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)  # w,x,y,z
        self.wrist_linear_velocity = torch.zeros(3, dtype=torch.float32)
        self.wrist_angular_velocity = torch.zeros(3, dtype=torch.float32)
        self.arm_data_received = False
        self.wrist_data_received = False
        self.wrist_state_topic_active = True  # we provide wrist pose+vel
        self.wrist_msg_count = 0
        self._prev_ee_pos = None
        self._prev_ee_t = None

        # ---- ZMQ sockets ----
        try:
            import zmq
            import msgpack
            import msgpack_numpy as _mnp
            _mnp.patch()
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "pyzmq + msgpack + msgpack-numpy are required for PolymetisArmClient. "
                "Install with: pip install pyzmq msgpack msgpack-numpy\n  %r" % (exc,)
            )
        self._zmq = zmq
        self._msgpack = msgpack
        self._ctx = zmq.Context.instance()

        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.CONFLATE, 1)          # keep only newest state
        self._sub.setsockopt(zmq.RCVTIMEO, 200)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub.connect(f"tcp://{ip_address}:{self.state_port}")

        self._push = self._ctx.socket(zmq.PUSH)
        self._push.setsockopt(zmq.SNDHWM, 2)
        self._push.connect(f"tcp://{ip_address}:{self.cmd_port}")

        self.logger.info(
            f"connecting to NUC bridge {ip_address} (state:{self.state_port} cmd:{self.cmd_port}) ..."
        )

        # ---- background state poller ----
        self._stop = threading.Event()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

        # wait for the first state so downstream code sees valid arm data
        t0 = time.time()
        while time.time() - t0 < connect_timeout_s:
            if self.arm_data_received:
                break
            time.sleep(0.05)
        if not self.arm_data_received:
            self.logger.warn(
                f"no state from bridge within {connect_timeout_s}s — is "
                f"polymetis_joint_bridge.py running on the NUC?"
            )

        if start_impedance:
            self.start_joint_impedance()

    # ------------------------------------------------------------------
    def _send(self, msg: dict):
        try:
            self._push.send(self._msgpack.packb(msg), flags=self._zmq.NOBLOCK)
        except Exception as exc:  # noqa: BLE001
            self.logger.warn(f"cmd send failed: {exc!r}")

    def start_joint_impedance(self):
        self._send({"cmd": "start_impedance", "kq": self.kq, "kqd": self.kqd})
        gains = f"Kq={self.kq}" if self.kq else "Polymetis default gains"
        self.logger.info(f"requested start_joint_impedance ({gains})")

    def go_home(self):
        self._send({"cmd": "go_home"})

    # ------------------------------------------------------------------
    def _poll_loop(self):
        while not self._stop.is_set():
            try:
                raw = self._sub.recv()
            except self._zmq.Again:
                continue
            except Exception as exc:  # noqa: BLE001
                self.logger.warn(f"state recv error: {exc!r}")
                continue
            try:
                s = self._msgpack.unpackb(raw, raw=False)
                self.arm_joint_positions = torch.as_tensor(
                    np.asarray(s["joint_pos"], dtype=np.float32)).flatten()
                self.arm_joint_velocities = torch.as_tensor(
                    np.asarray(s["joint_vel"], dtype=np.float32)).flatten()
                self.arm_data_received = True

                ee_pos = torch.as_tensor(np.asarray(s["ee_pos"], dtype=np.float32)).flatten()
                q_xyzw = torch.as_tensor(np.asarray(s["ee_quat_xyzw"], dtype=np.float32)).flatten()
                self.wrist_position = ee_pos
                self.wrist_quaternion = torch.tensor(
                    [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=torch.float32)  # -> w,x,y,z
                t = float(s.get("t", time.time()))
                if self._prev_ee_pos is not None and self._prev_ee_t is not None:
                    dt = max(t - self._prev_ee_t, 1e-4)
                    self.wrist_linear_velocity = (ee_pos - self._prev_ee_pos) / dt
                self._prev_ee_pos = ee_pos
                self._prev_ee_t = t
                self.wrist_data_received = True
                self.wrist_msg_count += 1
            except Exception as exc:  # noqa: BLE001
                self.logger.warn(f"state decode error: {exc!r}")

    # ------------------------------------------------------------------
    # Publisher-side interface (duck-types ROS2ActionPublisher)
    # ------------------------------------------------------------------
    def publish_arm_joint_pos(self, arm_joint_pos_des):
        if isinstance(arm_joint_pos_des, torch.Tensor):
            q = arm_joint_pos_des.detach().cpu().flatten().float().numpy()
        else:
            q = np.asarray(arm_joint_pos_des, dtype=np.float32).flatten()
        if q.size != 7:
            self.logger.warn(f"publish_arm_joint_pos expected 7 values, got {q.size}; ignoring")
            return
        if not np.isfinite(q).all():
            self.logger.warn("non-finite arm target; ignoring")
            return
        self._send({"cmd": "joint_target", "q": q})

    def publish_hand_joints(self, *args, **kwargs):
        """No-op: the Sharpa hand is driven directly via the Sharpa SDK in the env."""
        return

    # ------------------------------------------------------------------
    def get_logger(self):
        return self.logger

    def shutdown(self):
        self._stop.set()
        try:
            if self._poll_thread.is_alive():
                self._poll_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._send({"cmd": "terminate"})
        except Exception:
            pass
        for sock in (getattr(self, "_sub", None), getattr(self, "_push", None)):
            try:
                if sock is not None:
                    sock.close(linger=0)
            except Exception:
                pass
