#!/usr/bin/env python
"""ZMQ joint-control bridge — runs ON THE NUC (polymetis env).

Bridges the local Polymetis server to a remote deploy client over ZMQ, so the
training PC (py310 / isaaclab, no polymetis) can drive the FR3 with joint
impedance without installing polymetis. Mirrors the pattern of the existing
~/controller/polymetis_bridge.py (which was Cartesian); this one is JOINT.

Run on the NUC:
    conda activate polymetis-local
    python polymetis_joint_bridge.py            # robot=localhost:50051
Sockets (bind on all interfaces so the wired-link client can reach them):
    state : PUB  tcp://*:5560   (msgpack) {joint_pos[7], joint_vel[7],
                                            ee_pos[3], ee_quat_xyzw[4], t}
    cmd   : PULL tcp://*:5561   (msgpack) {"cmd": ...}
              joint_target  {"q":[7]}
              start_impedance {"kq":[7]|None, "kqd":[7]|None}
              terminate / go_home
"""
import argparse
import threading
import time

import numpy as np
import torch
import zmq
import msgpack
import msgpack_numpy as m
m.patch()

from polymetis import RobotInterface


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_ip", default="localhost", help="Polymetis server IP (on the NUC = localhost).")
    ap.add_argument("--robot_port", type=int, default=50051)
    # These two literals are deliberate. This file runs ON THE NUC inside the
    # `polymetis-local` env, where `dexx` is not installed — importing
    # deploy_config here would break the bridge. They must stay in sync with
    # POLYMETIS_STATE_PORT / POLYMETIS_CMD_PORT by hand.
    ap.add_argument("--state_port", type=int, default=5560)
    ap.add_argument("--cmd_port", type=int, default=5561)
    ap.add_argument("--state_hz", type=float, default=200.0)
    args = ap.parse_args()

    print(f"[bridge] connecting to Polymetis {args.robot_ip}:{args.robot_port} ...")
    robot = RobotInterface(ip_address=args.robot_ip, port=args.robot_port, enforce_version=False)
    print("[bridge] connected.")

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{args.state_port}")
    pull = ctx.socket(zmq.PULL)
    pull.bind(f"tcp://*:{args.cmd_port}")
    print(f"[bridge] state PUB tcp://*:{args.state_port}   cmd PULL tcp://*:{args.cmd_port}")

    def cmd_loop():
        while True:
            try:
                msg = msgpack.unpackb(pull.recv(), raw=False)
                cmd = msg.get("cmd")
                if cmd == "joint_target":
                    q = torch.as_tensor(np.asarray(msg["q"], dtype=np.float32))
                    robot.update_desired_joint_positions(q)
                elif cmd == "start_impedance":
                    kq, kqd = msg.get("kq"), msg.get("kqd")
                    if kq is not None and kqd is not None:
                        robot.start_joint_impedance(
                            Kq=torch.tensor(kq, dtype=torch.float32),
                            Kqd=torch.tensor(kqd, dtype=torch.float32),
                        )
                        print(f"[bridge] started joint impedance (Kq={kq}, Kqd={kqd})")
                    else:
                        robot.start_joint_impedance()
                        print("[bridge] started joint impedance (default gains)")
                elif cmd == "terminate":
                    robot.terminate_current_policy()
                    print("[bridge] terminated current policy")
                elif cmd == "go_home":
                    robot.go_home()
                    print("[bridge] go_home done")
                else:
                    print(f"[bridge] unknown cmd: {cmd}")
            except Exception as exc:  # noqa: BLE001
                print(f"[bridge] cmd error: {exc!r}")

    threading.Thread(target=cmd_loop, daemon=True).start()

    dt = 1.0 / args.state_hz
    n = 0
    while True:
        t0 = time.time()
        try:
            q = robot.get_joint_positions().detach().cpu().numpy().astype(np.float32)
            qd = robot.get_joint_velocities().detach().cpu().numpy().astype(np.float32)
            ee_pos, ee_quat = robot.get_ee_pose()
            state = {
                "joint_pos": q,
                "joint_vel": qd,
                "ee_pos": ee_pos.detach().cpu().numpy().astype(np.float32),
                "ee_quat_xyzw": ee_quat.detach().cpu().numpy().astype(np.float32),
                "t": t0,
            }
            pub.send(msgpack.packb(state))
            n += 1
            if n % (int(args.state_hz) * 5) == 0:
                print(f"[bridge] streaming... q0={q[0]:.3f} ({n} msgs)")
        except Exception as exc:  # noqa: BLE001
            print(f"[bridge] state error: {exc!r}")
            time.sleep(0.05)
        sleep = dt - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)


if __name__ == "__main__":
    main()
