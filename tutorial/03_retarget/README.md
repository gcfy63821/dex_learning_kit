# 03 — Retargeting a human demonstration onto the robot

**Goal:** turn a MANO hand trajectory into a robot joint trajectory the simulator
can replay, and verify it rather than trust it.

## Run it

```bash
python scripts/retarget.py --side right \
  --data_idx rt/0416_grasp/cube_small_2 --iter 5000 --headless
```

Two stages of Adam: stage 1 solves the arm alone to reach the wrist target, stage
2 opens up the hand and the arm's base yaw together to match fingertips. 5000
iterations takes roughly three minutes on one GPU.

**Pass the sequence's placement offsets** (lesson 01's table). Omitting them moves
the object and confounds whatever you were actually testing.

## The reachability gate

After fitting, the mean arm end-effector error over the trajectory is compared
against `--reachability_th` (0.08 m). Above it, the variant is marked
`reachable=False` and a partial pkl is written, which the training loader skips.

Healthy values for the shipped demos are 1.3–2.1 cm. A sequence at 7 cm is
telling you the demonstration does not fit your robot's workspace, not that it
needs more iterations.

## Two things that will waste your afternoon

**The output has to be copied to where it is read.** `scripts/retarget.py` writes
into `data/retargeting/...`, but the motion-preparation step goes through the
dataset loader, which reads the same relative path under its own root. A retarget
that is not copied across is silently ignored — the next stage reports success and
emits a file built from whatever was there before.

**Isaac Sim does not exit when the work is done.** The process finishes its
optimization, writes its pkl, and then sits at 100% CPU forever. A `for` loop over
sequences therefore never reaches the second one. One such process was found still
spinning **seven days** after it had written its output.

So do not key a runner on whether the process is alive, and do not key it on CPU
either — a hung process is indistinguishable from a working one by load. Key it on
**the artefact the process was supposed to produce**:

```bash
before=$(stat -c %Y "$PKL" 2>/dev/null || echo 0)
python -u scripts/retarget.py ... &          # note -u: block-buffered logs look like hangs
pid=$!
while sleep 10; do
  [ "$(stat -c %Y "$PKL" 2>/dev/null || echo 0)" != "$before" ] && break
done
sleep 20; kill -9 $pid                        # let it flush, then stop it
```

The `-u` matters: without it Python block-buffers stdout to a file and the log can
sit unchanged for 45 minutes while the run is perfectly healthy. Two separate
diagnoses were wasted on that.

## Verify before you train on it

```bash
python scripts/check_asset_equivalence.py     # asset side
```

and, for the motion itself, replay the stored joint trajectory through the URDF
and compare against the stored body positions. Under a millimetre means the joint
ordering, the frame convention and the frame rate all line up. Two orders of
magnitude worse means a joint-order bug (lesson 02).

Also look at the closest fingertip-to-object distance across the demo. It is the
one number that tells you the *hand-object relationship* survived, which is what
lesson 01 is protecting.

## Porting: a new task or new objects

1. Collect the MANO demonstration and object trajectory in the source format.
2. Retarget it; record its placement offsets in lesson 01's table.
3. Check reachability, then replay-verify.
4. Add it to `--data_idx` for the teacher (04) and the student (05).

Object *shape* reaches the policy through a BPS encoding of the mesh, so a new
object needs its mesh present — not just its trajectory.
