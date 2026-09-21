# 06 — Camera extrinsic calibration

**Goal:** make the simulated depth camera sit where the real one sits, and be able
to prove it.

## What the number is

A 4×4 **camera-in-armbase** transform in the ROS optical convention (x right,
y down, z forward). Everything else about the camera — intrinsics, resolution,
depth clamps — lives in `deploy_config.py` as a constant. The extrinsic does not,
because it is a property of how you bolted the camera down, not of the camera.

Shipped calibrations are in `calib/camera_align/`.

## Look at them first

```bash
python tutorial/06_camera_calibration/inspect_extrinsic.py
```

No camera needed. It validates each matrix (orthonormal rotation, `det = +1`,
proper bottom row), describes where the camera is and which way it looks, and
reports the drift between any two.

That last part is the lesson. Running it here prints:

```
hard-coded fallback -> current.npy    |dt|= 20.1 cm  dtheta= 4.42 deg
```

`visual_raycaster._build_camera_extrinsics()` carries a hard-coded matrix from
2026-05-20. The camera has been re-mounted since. For a while, training used that
stale matrix because passing an extrinsic was a deploy-only option — so policies
were trained on a viewpoint 20 cm from the one they would meet, and nothing
warned anybody. Training converged. Losses looked fine.

> **A calibration is an input, not a constant.** If it can go stale, it must be
> passed explicitly at every stage that consumes it.

It now is:

```bash
--camera_extrinsic calib/camera_align/current.npy
```

on `train_dagger_pc.py`, `eval.py` and `deploy_pc.py` alike. The hard-coded value
survives only as a fallback, and the inspector exists so that fallback is never
silently in play.

## Calibrating your own

The toolchain ships in `tools/calib/`. Every step below is a real command.

What you need: the robot powered and reachable over Polymetis (lesson 09's steps
1–2), the camera publishing depth over ZMQ, and an initial guess for the
extrinsic — measuring the mount with a tape gets you close enough for ICP to
converge.

### 1. Put the arm somewhere both sides agree on

Calibration works by matching a real picture of the robot against a simulated
one, so both have to be in the same configuration. Drive the real arm to a
demonstration frame:

```bash
python deploy/move_to_frame_polymetis.py \
    --data_idx rt/0416_grasp/cube_small_2 --frame 0
```

Pick a frame where the arm is *in view and spread out*. A folded arm gives ICP
almost nothing to lock onto.

### 2. Capture the real cloud

With the camera host publishing depth+colour on ZMQ:

```bash
python tools/calib/capture_multiframe_zmq.py \
    --addr tcp://<CAM_HOST>:5562 --n_frames 10 \
    --out_dir logs/calib_real
```

Ten frames of a held-still scene are accumulated into one dense cloud
(`logs/calib_real/accum.npz`, points in the **camera** frame). Overlaying frames
fills in dropout and shows you how much the sensor jitters.

### 3. Render the matching sim cloud

Forward kinematics on the same frame, sampled to a point cloud:

```bash
python tools/calib/gen_sim_frame_ply.py \
    --pkl data/retargeting/robotool_batch/mano2sharpa_rh/0416_grasp/cube_small_2@0.pkl \
    --frame 0 --out logs/calib_sim_frame.ply
```

This is the reference the real cloud gets aligned to. The table plane is included
(at `TABLE_SURFACE_Z`) because it is a large, unambiguous surface — useful for
pinning down height and tilt even though it says nothing about yaw.

### 4. Align automatically (ICP)

```bash
python tools/calib/icp_align_extrinsic.py \
    --real_npz logs/calib_real/accum.npz \
    --init_extrinsic calib/camera_align/current.npy \
    --pkl data/retargeting/robotool_batch/mano2sharpa_rh/0416_grasp/cube_small_2@0.pkl \
    --frame 0 --box -0.2 0.6 -0.2 0.2 0.0 0.6 \
    --out calib/camera_align/icp_it1.npy
```

Kabsch/SVD ICP in the arm-base frame: apply the initial extrinsic to the real
cloud, crop both to the same box, match, reject correspondences beyond
`--max_corr`, and fold the resulting rigid transform into the extrinsic.

**Iterate.** Feed its output back in as `--init_extrinsic` two or three times; the
translation correction should shrink each round. If it does not, the initial
guess is too far off or the crop box contains something that is not in the sim
model (a clamp, a cable, your hand).

### 5. Verify by eye, and nudge

ICP will converge confidently to a wrong answer on a scene that is mostly a flat
table. This step is not optional.

```bash
python tools/calib/live_calibrate_extrinsic.py \
    --real_pc_npz logs/calib_real/accum.npz \
    --sim_frame_pkl data/retargeting/robotool_batch/mano2sharpa_rh/0416_grasp/cube_small_2@0.pkl \
    --sim_frame_idx 0 \
    --init_extrinsic calib/camera_align/icp_it1.npy \
    --out calib/camera_align/current.npy --port 8080
```

Opens a viser session in your browser: the real coloured cloud over the sim robot
meshes, with six sliders for translation and rotation. Look along each axis in
turn. The arm's *silhouette* is what you are matching — if the real link edges
sit inside or outside the sim mesh consistently, that is a translation error; if
they diverge along the arm, it is rotation.

Save writes the 4×4 to `--out`.

### 6. Verify what the policy will actually see

The policy never sees the raw cloud — it sees the cropped one. A small extrinsic
error can remove different things on each side.

```bash
python tools/calib/viz_cropped_overlay.py \
    --real_ply logs/calib_real/cropped_env_local.ply \
    --sim_link0_ply logs/calib_sim_frame.ply --port 8081
```

Both clouds get the same workspace box (lesson 01). Confirm the table is gone
from both and that the object and hand points that survive line up.

### 7. Record what you changed

```bash
python tutorial/06_camera_calibration/inspect_extrinsic.py
```

It prints the drift between your new file and the previous one. A few millimetres
is re-measurement noise. A few centimetres means the camera actually moved — and
every policy trained before that move is now trained on the wrong viewpoint.

### The constants these tools read

They all import `dexx.deploy_config`, so the table height, arm-base height, crop
box and intrinsics come from the same place the trainer uses. Nothing to keep in
sync by hand — which matters, because the bundled copies of these tools shipped
with `arm_base = 0.432` hardcoded long after the robot had been re-mounted at
0.415.

## Sanity checks that catch most errors

* **Table plane.** Back-project the real depth and fit a plane. It should come out
  at `TABLE_SURFACE_Z` and level. A tilt of half a degree is enough to make the
  table survive the crop on one side of the image and not the other.
* **Object count.** After cropping, the object should be a few hundred points. If
  it is a handful, the crop floor is too high or the extrinsic is off in z.
* **Re-run the inspector** against your previous calibration. A few millimetres is
  re-measurement noise; a few centimetres means the camera moved.

## Porting

Moving the camera, changing lens or resolution, or re-mounting the arm all
invalidate the extrinsic — re-mounting the arm because the transform is expressed
*in the arm base frame*.

Changing resolution or camera model also means new `SIM_INTRINSICS` (lesson 01).
Check the numbers are the **depth** stream's, not the colour stream's: on a D455
those differ substantially, and using colour intrinsics narrows the modelled field
of view and quietly drops a third of the workspace.

`check_frames.py` from lesson 01 prints the horizontal FoV your intrinsics imply —
compare it against the datasheet.
