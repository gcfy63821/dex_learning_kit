# 05 — Distilling a point-cloud student

**Goal:** replace privileged state with a camera and a PointNet, and understand
what the student is and is not given.

## Run it

```bash
python scripts/train_dagger_pc.py \
  --task franka-sharpa-pointcloud \
  --teacher_ckpt checkpoints/teacher_poseobs.pth \
  --side right \
  --data_idx '["rt/0416_grasp/cube_small_1","rt/0416_grasp/cube_small_2",
               "rt/0420_manip/squeegee_1","rt/0420_manip/squeegee_2"]' \
  --num_envs 128 --dagger_iters 30 --rollout_steps 4096 \
  --beta_init 1.0 --beta_decay 0.85 \
  --train_epochs 8 --batch_size 512 --lr 5e-5 --max_buffer 200000 \
  --hand_body_subset minimal6 \
  --student_drop_slots obj_bps,tips_distance,obj_pose_tail \
  --camera_extrinsic calib/camera_align/current.npy \
  --seed 42 --out_dir logs/my_student --headless
```

About 30 minutes on one GPU. Loss should fall by roughly a factor of 40 across the
run while β decays 1.0 → 0.009.

**`--camera_extrinsic` is not optional.** Lesson 06 explains what happens without
it.

## DAgger, precisely

The executed action is a **convex blend**, not a coin flip between two policies:

```python
action = beta * expert_action + (1 - beta) * student_action
```

β starts at 1.0 and is multiplied by 0.85 each iteration. The replay buffer is
aggregated across iterations (capacity 2×10⁵) and never cleared — that is what
makes this DAgger rather than iterated behaviour cloning. The loss is a plain MSE
against the expert's action.

## What the student gives up

The teacher keeps all 557 dims for labelling. The student's copy is **sliced**:

| dropped | dim | why |
|---|---|---|
| `obj_bps` | 128 | object shape must come from the point cloud, not a mesh encoding |
| `tips_distance` | 5 | demo-precomputed, unavailable at deploy |
| `obj_pose_tail` | 7 | the object pose *estimate* — the student must localize visually |

557 − 140 = **417**. Slicing rather than zeroing means the student has no dead
inputs and a layout mismatch fails on a shape error instead of silently feeding
unlearned signal.

Note what is **not** dropped: `target_obj_pose` (390:397), the demonstration's
*target* object pose. The student does not know where the object is; it does know
where the object is supposed to go.

## The point cloud

| source | points | per-point channels |
|---|---|---|
| scene (depth camera) | 1024 | xyz + type + force(0) |
| hand keypoints | 6 | wrist + 5 fingertips |
| tactile | 25 | 5 fingertips × 5 surface samples, force attached |

One unified cloud of **1055 × 5** through a shared PointNet with masked
max-pooling → 64 dims. The student MLP takes 417 + 64 = **481** and outputs the
same 29 actions.

## Check

The checkpoint records its own layout. After training:

```python
c = torch.load("logs/my_student/dagger_final.pth", weights_only=False)
print(c["cfg"].proprio_dim, c["student_drop_slots"], c["student_obs_slots"])
# 417  ['obj_bps','tips_distance','obj_pose_tail']  {'proprioception': (0,79), ...}
```

Evaluation re-applies that layout automatically and refuses to run if the live
slot map disagrees with the saved one.

## Porting

Changing the observation layout changes the slot map, which invalidates saved
`keep_idx`. The guard will catch it. Re-train the student; the teacher is
unaffected only if its own observation did not change.
