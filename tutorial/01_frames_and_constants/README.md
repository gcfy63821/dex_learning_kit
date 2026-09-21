# 01 — Frames and constants: the sim2real contract

**Goal:** know every number that has to mean the same thing in the retargeter, the
simulator, the point-cloud crop and the real robot — and know what happens when
one copy drifts.

**Why this is lesson 01:** frame bugs do not raise exceptions. They produce a
policy that trains beautifully and reaches for empty air. Every other lesson
assumes these are right.

## One file

Everything lives in `src/dexx/deploy_config.py`. Edit there; every consumer
imports it.

| constant | value | meaning |
|---|---|---|
| `TABLE_SURFACE_Z` | 0.415 | table top, env-local. Objects rest relative to this |
| `ARM_BASE_Z` / `ARM_BASE_POS` | 0.415 / (-0.1, 0, 0.415) | `fr3_link0` origin |
| `WRIST_POS_OFFSET` | (-0.1, 0, 0.415) | places the human demo into the scene |
| `SIM_INTRINSICS` | fx 193.33, fy 193.06, cx 160.08, cy 121.05 | depth camera, 320×240 |
| `PC_WORKSPACE_MIN/MAX` | z 0.420 … 1.30 | point-cloud crop box |
| `POLYMETIS_STATE_PORT` / `_CMD_PORT` | 5560 / 5561 | ZMQ bridge to the arm |

## The table and the arm base are independent

They happen to be equal right now (both 0.415) because the robot is bolted level
with the table surface. They were **not** equal between 2026-07-16 and
2026-09-01, when the base sat 1.7 cm proud of the table.

That coincidence is a trap, and it caught this project. `WRIST_POS_OFFSET` — the
constant that places the human demonstration into the scene — had been written as
`ARM_BASE_Z` rather than as the table height, because the two were the same
number. When the base was lowered by 1.7 cm, the demonstration's hand trajectory
followed it down while the object, anchored to the table, stayed put. The grasp
was silently 1.7 cm too low.

The rule that falls out:

> A constant must be written as **what it physically is**, not as whatever other
> constant happens to share its value today.

`WRIST_POS_OFFSET` is now written as `TABLE_SURFACE_Z` and carries a comment
saying why. Moving the arm base no longer moves the demonstration.

### What *should* change when the base moves

Moving the robot changes which joint angles reach a given point. It does not
change where the object is or where the hand must go to grasp it. So after a
re-mount and a re-retarget, expect exactly this:

| quantity | expected |
|---|---|
| `object_pos` | **unchanged** — it is anchored to the table |
| `arm_joint_pos` | **changed** — the arm solves from a new base |
| closest fingertip-to-object distance | **unchanged** to within a millimetre or two |

If the fingertip-object relationship moved, the demonstration was dragged along
with the base and something is still coupled that should not be.

## Retarget placement offsets are part of the data

Each demonstration is retargeted with a placement offset deciding where its object
lands. They are CLI arguments and they differ per sequence. Re-running a retarget
without them moves the object 10 cm and nobody notices.

| sequence | arguments |
|---|---|
| `rt/0416_grasp/cube_small_1` | defaults (`--target_offset_xy 0.45 0.0`, `--z_offset 0`) |
| `rt/0416_grasp/cube_small_2` | defaults |
| `rt/0420_manip/squeegee_1` | `--target_offset_xy 0.45 0.1 --z_offset 0.01` |
| `rt/0420_manip/squeegee_2` | `--target_offset_xy 0.45 0.1 --z_offset 0.01` |

They are recoverable from data if lost: `target_offset_xy` **is** the first
object's xy, and `z_offset` is the difference in its z. Add a row when you
retarget a new sequence — lesson 03.

## The workspace crop

`PC_WORKSPACE_MIN/MAX` is applied before the point cloud is subsampled, so the
1024 points land on the manipulation region instead of being spent on the table
and the背景 curtain. The floor sits 5 mm above the table top on purpose: lower
and the table dominates every cloud; much higher and short objects vanish. An
earlier value of 0.50 cut off the bottom 8.5 cm and removed the objects
entirely.

This box must match on both sides. The deploy-side depth converter crops with the
same numbers; if they diverge, the student sees a differently-shaped world than it
trained on.

## Check

```bash
python tutorial/01_frames_and_constants/check_frames.py
```

Pure arithmetic, no simulator. It asserts the relationships rather than the
values, so it still passes after you re-mount your robot — and fails if you break
the decoupling.

## Porting

| you changed | edit | then |
|---|---|---|
| table height | `TABLE_SURFACE_Z` | re-retarget (03), re-crop check |
| arm mount height | `ARM_BASE_Z` | re-retarget (03); expect joints to change and object not to |
| camera | `SIM_INTRINSICS`, `DEPTH_H/W` | recalibrate the extrinsic (06) |
| workspace | `PC_WORKSPACE_MIN/MAX` | keep the deploy-side crop identical |
| arm networking | `POLYMETIS_*` ports | lesson 09 |

Changing either height invalidates existing retargets and every policy trained on
them. That is not a warning to be careful — it is a re-run.
