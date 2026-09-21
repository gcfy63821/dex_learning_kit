"""ZMQ RealSense depth subscriber — runs ON THE INFERENCE PC.

Duck-types `RealSenseDepthSubscriber` (the ROS2 one) so the PointCloud deploy
env can consume it unchanged: same `get_latest()` (fp32 (H,W) torch on the
configured device, or None) and `n_received` property. Instead of a ROS2 topic,
it SUBs depth frames from `realsense_depth_zmq_pub.py` running on the camera host.

    sub = RealSenseDepthZmqSubscriber(addr="tcp://101.6.90.122:5562", device="cuda")
    ...
    depth_m = sub.get_latest()   # fp32 (240,320) meters torch on device, or None

A background thread polls the ZMQ SUB (CONFLATE=1 keeps only the newest frame),
mirroring the PolymetisArmClient state poller.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import torch


class RealSenseDepthZmqSubscriber:
    def __init__(
        self,
        addr: str,
        height: int = 240,
        width: int = 320,
        device: str = "cpu",
        rcvtimeo_ms: int = 200,
    ):
        self._H = int(height)
        self._W = int(width)
        self._device = torch.device(device)
        self._latest: Optional[torch.Tensor] = None
        self._latest_stamp: Optional[float] = None
        self._n_received = 0
        self._lock = threading.Lock()

        try:
            import zmq
            import msgpack
            import msgpack_numpy as _mnp
            _mnp.patch()
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "pyzmq + msgpack + msgpack-numpy are required for "
                "RealSenseDepthZmqSubscriber. pip install pyzmq msgpack msgpack-numpy\n"
                f"  {exc!r}"
            )
        self._zmq = zmq
        self._msgpack = msgpack
        self._ctx = zmq.Context.instance()
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.CONFLATE, 1)        # keep only newest depth frame
        self._sub.setsockopt(zmq.RCVTIMEO, int(rcvtimeo_ms))
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub.connect(addr)
        print(f"[depth-zmq-sub] connecting {addr} -> ({self._H}x{self._W}) on {device}", flush=True)

        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._poll_loop, daemon=True)
        self._thr.start()

    def _poll_loop(self):
        while not self._stop.is_set():
            try:
                raw = self._sub.recv()
            except self._zmq.Again:
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[depth-zmq-sub] recv error: {exc!r}", flush=True)
                continue
            try:
                msg = self._msgpack.unpackb(raw, raw=False)
                depth = np.array(msg["depth"], dtype=np.float32)  # copy: msgpack buffer is read-only
                if depth.shape != (self._H, self._W):
                    # tolerate size drift: crop/pad to (H,W)
                    out = np.zeros((self._H, self._W), dtype=np.float32)
                    hh, ww = min(depth.shape[0], self._H), min(depth.shape[1], self._W)
                    out[:hh, :ww] = depth[:hh, :ww]
                    depth = out
                t = torch.from_numpy(depth).to(self._device, dtype=torch.float32)
                with self._lock:
                    self._latest = t
                    self._latest_stamp = float(msg.get("t", time.time()))
                    self._n_received += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[depth-zmq-sub] decode error: {exc!r}", flush=True)

    def get_latest(self) -> Optional[torch.Tensor]:
        with self._lock:
            return self._latest

    def get_latest_with_stamp(self):
        with self._lock:
            return self._latest, self._latest_stamp

    @property
    def n_received(self) -> int:
        return self._n_received

    def shutdown(self):
        self._stop.set()
        try:
            self._sub.close(0)
        except Exception:
            pass

    # ROS2-compat no-ops (deploy_pc treats it like a node)
    def destroy_node(self):
        self.shutdown()
