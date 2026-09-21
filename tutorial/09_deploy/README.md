# 09 — Deploying on the real robot

**Goal:** run the student on hardware, with the checks that stop you from finding
out the hard way.

`docs/DEPLOY.md` is the operational reference; this lesson is the reasoning around
it.

## Three processes on three machines

```
camera host            NUC                      workstation
-----------            ---                      -----------
RealSense              Polymetis server         deploy_pc.py
depth ZMQ PUB   ─────► joint bridge   ◄────────  (the policy)
    :5562              :5560 state
                       :5561 cmd
```

The split is not incidental. Polymetis wants to sit next to the arm on a
real-time-ish machine; the policy wants a GPU. ZMQ between them means either side
can be restarted without the other noticing.

The live path is **Polymetis + ZMQ**. Any ROS2 instructions you find in older
notes are stale.

## Launch order

Order matters because each stage assumes the previous one is up.

```bash
# 1. NUC: Polymetis server (FCI active, joints unlocked)
# 2. NUC: the bridge
python deploy/polymetis_joint_bridge.py

# 3. camera host: depth publisher

# 4. workstation: confirm the arm before trusting it
python deploy/test_polymetis_arm.py

# 5. workstation: move to a demo frame
python deploy/move_to_frame_polymetis.py --data_idx rt/0416_grasp/cube_small_2

# 6. workstation: run the policy
python deploy/deploy_pc.py \
  --load_path logs/my_student/dagger_final.pth \
  --camera_extrinsic calib/camera_align/current.npy \
  --depth_zmq_addr tcp://<camera-host>:5562
```

Step 4 exists so that a bridge problem surfaces as a failed sanity check rather
than as a policy that appears to be misbehaving.

## Before the first run

Work through these in order. Each one has cost somebody a day.

| check | lesson |
|---|---|
| `check_frames.py` passes and matches your physical setup | 01 |
| the extrinsic you are deploying is the one you **trained** with | 06 |
| the real cropped cloud overlays the sim cropped cloud | 06 |
| joint ordering verified end to end — the hand, not just the arm | 02 |
| impedance gains on the arm match what the policy trained against | 07 |
| the checkpoint's `student_obs_slots` matches the live env | 05 |

The extrinsic one deserves emphasis: **train and deploy must use the same file.**
A policy trained on the stale fallback and deployed on `current.npy` sees a
20 cm viewpoint shift at exactly the moment it matters.

## What the student receives at deploy

417 proprioceptive dims plus a 64-d point-cloud feature. The dropped channels
(lesson 05) are precisely the ones unavailable on hardware — object shape encoding,
demo-precomputed fingertip distances, and the object pose estimate. That is the
point of dropping them: **the student's input vector is the same shape in sim and
on the robot**, so there is no deploy-time substitution to get wrong.

What is *not* dropped is `target_obj_pose` — the demonstration's target object
pose, which comes from the demo file at deploy just as it does in sim. The policy
is told where the object should go, not where it is.

## When it behaves differently than in simulation

Work down lesson 07's symptom table first — most sim2real gaps on this platform
have been dynamics, not perception. Then check the extrinsic. Then check that the
cropped point cloud actually contains the object, by dumping it and looking at it.

Resist retraining until you have a measurement that says which of the three it is.
