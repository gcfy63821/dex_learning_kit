"""Overlay CROPPED real PC vs CROPPED sim frame in env-local (what the policy sees).

Both clouds get the SAME workspace bbox crop (deploy = sim = pc_workspace_min/max),
so the table (below z=0.420) is removed from both. Confirms real object/hand
points that survive the crop line up with sim.

  python tools/calib/viz_cropped_overlay.py \
      --real_ply logs/real_pc_zmq_sq1/dump_XXXX/cropped_env_local.ply \
      --sim_link0_ply calib/camera_align/sim_frame_squeegee1_link0.ply \
      --port 8081
"""
import argparse
import numpy as np
from dexx import deploy_config as _dcfg


def read_ply(path):
    pts, hdr = [], True
    for ln in open(path):
        if hdr:
            if ln.startswith("end_header"):
                hdr = False
            continue
        pts.append([float(x) for x in ln.split()[:3]])
    return np.array(pts, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_ply", required=True, help="cropped_env_local.ply (already cropped)")
    p.add_argument("--sim_link0_ply", required=True, help="gen_sim_frame_ply link0 output")
    p.add_argument("--arm_base", type=str,
                   default=",".join(str(x) for x in _dcfg.ARM_BASE_POS))
    p.add_argument("--ws_min", type=str, default="0.0,-0.4,0.42")
    p.add_argument("--ws_max", type=str, default="0.8,0.25,1.3")
    p.add_argument("--port", type=int, default=8081)
    args = p.parse_args()

    arm = np.array([float(x) for x in args.arm_base.split(",")], dtype=np.float32)
    wmin = np.array([float(x) for x in args.ws_min.split(",")], dtype=np.float32)
    wmax = np.array([float(x) for x in args.ws_max.split(",")], dtype=np.float32)

    real = read_ply(args.real_ply)  # already env-local + cropped
    sim0 = read_ply(args.sim_link0_ply)  # link0 frame
    sim = sim0 + arm  # -> env-local
    # apply the SAME crop to sim
    inbox = np.all((sim >= wmin) & (sim <= wmax), axis=1)
    sim_c = sim[inbox]

    print(f"[overlay] real cropped n={len(real)}  sim cropped n={len(sim_c)} "
          f"(sim before crop {len(sim)}, table removed {len(sim)-len(sim_c)})", flush=True)
    for nm, P in [("real", real), ("sim_cropped", sim_c)]:
        if len(P):
            print(f"  {nm}: x[{P[:,0].min():.3f},{P[:,0].max():.3f}] "
                  f"y[{P[:,1].min():.3f},{P[:,1].max():.3f}] "
                  f"z[{P[:,2].min():.3f},{P[:,2].max():.3f}]", flush=True)

    import viser
    srv = viser.ViserServer(port=args.port)
    srv.scene.add_point_cloud(
        "/real_cropped", points=real,
        colors=np.tile(np.array([50, 130, 255], np.uint8), (len(real), 1)),
        point_size=0.004)
    srv.scene.add_point_cloud(
        "/sim_cropped", points=sim_c,
        colors=np.tile(np.array([255, 140, 0], np.uint8), (len(sim_c), 1)),
        point_size=0.004)
    # workspace bbox outline at crop floor
    z = float(wmin[2])
    loop = np.array([
        [wmin[0], wmin[1], z], [wmax[0], wmin[1], z], [wmax[0], wmax[1], z],
        [wmin[0], wmax[1], z], [wmin[0], wmin[1], z]], np.float32)
    srv.scene.add_spline_catmull_rom(
        "/ws_floor", positions=loop, color=(120, 120, 120), line_width=2.0)
    print(f"[overlay] viser at http://localhost:{args.port}  "
          f"(blue=real cropped, orange=sim cropped, table removed from both)", flush=True)
    import time
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
