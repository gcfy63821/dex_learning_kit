# 02 — Assets and joint ordering

**Goal:** understand what a robot description has to carry beyond link geometry,
and why three different joint orderings coexist without anyone getting hurt.

**Read this before swapping the arm or the hand.** It is the lesson that decides
whether your new robot behaves or tunnels through its own joint limits.

## A URDF is not a complete physics description

URDF describes kinematics and inertia. It cannot express two things PhysX needs,
and both of them fail silently.

### Rotor inertia (`armature`)

The Sharpa finger links have inertias around 1e-6 kg·m². Without rotor inertia
reflected at the joint, a 0.2 N·m torque produces enough angular acceleration to
pass straight through a joint limit inside one 1/120 s step.

Measured on this hand: **14 of 29 joints ended up outside their limits after 100
zero-action steps**, one of them at 15 rad against a `[0, 1.75]` limit. With
armature restored: 0 of 29.

The values live in `src/dexx/robot_constants.py`, read out of the vendor's USD and
converted from USD's per-degree drive convention. They are the only source — the
comment there says *do not set them to None*, and means it.

### Self-collision filter pairs

The hand needs 14 `physics:filteredPairs` so adjacent finger links do not collide
with each other. URDF has no way to say this. Convert a URDF at spawn time and
they vanish — the fingers then cannot close, and the teacher's success rate drops
from 95% to 38% with no error anywhere.

The fix in this repo is to stop converting at runtime:

```bash
python scripts/build_merged_urdf.py          # URDF from the vendor parts
python scripts/build_robot_usd.py --side both  # convert once, author the filters, commit
```

`franka_sharpa_robot_usd()` records a `.source_hash` and warns when the built USD
and the URDF have drifted apart. Any code path that spawns the URDF directly
loses the filters again — that has already happened once, in `scripts/retarget.py`.

> The general shape: **if an asset needs post-processing to be correct, bake it
> once and commit the result.** Reconstructing it at runtime means every entry
> point must remember to, and one of them will not.

## Three joint orderings

They all exist for a reason, and confusing them maps a command to the wrong motor
— which on real hardware is a safety problem, not a bug.

| ordering | built by | used by |
|---|---|---|
| **cfg order** (= Sharpa order = the real hand) | `cfg.actuated_joint_names` | retarget output, the real hand's `set_joint_position()` |
| **USD order** (= sorted) | `actuated_dof_indices` after `.sort()` | the policy's action space, `hand_dof_pos` |
| legacy "isaaclab" order | `dof_isaaclab2sharpa()` | **deprecated, do not use** |

The policy thinks in USD order. The hardware thinks in cfg order. Exactly one
conversion sits between them, and `docs/JOINT_ORDERING.md` is the reference for
where.

### How this is guarded

A scrambled joint order passes the asset check (it compares geometry), passes a
zero-action rollout (it only looks at limits), and trains — toward a garbled hand
shape, forever. It reads like "the reward needs tuning".

The guard is a kinematic replay: drive the URDF with the stored joint trajectory
and compare against the stored body positions. A correct pipeline reproduces
itself to well under a millimetre; a joint-order bug lands two orders of magnitude
above. Lesson 03 runs it.

## Porting

**Different hand.** Expect to supply: the actuated joint name list in cfg order,
per-joint stiffness/damping/armature/friction, the self-collision filter pairs,
and the fingertip/elastomer link names the tactile sensing uses. Then re-run the
two build scripts. The action dimension changes, so every checkpoint is invalid.

**Different arm.** Fewer moving parts: the joint names, the per-joint impedance
gains (lesson 07), and `ARM_BASE_POS`. The 7-DoF assumption shows up in the action
split (`7 arm + 22 hand`), so a 6-DoF arm needs that changed too.

**Do not skip the armature and collision-filter steps for a new hand.** They are
not Sharpa-specific; they are what URDF cannot express. Any hand with small
finger inertias and adjacent links has both problems.

## Check

```bash
python scripts/check_asset_equivalence.py
```

Compares the built USD against the merged URDF and reports drift.
