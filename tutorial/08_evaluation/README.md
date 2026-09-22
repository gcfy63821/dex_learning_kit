# 08 — Evaluation that means something

**Goal:** produce a number you would defend, which mostly means knowing what your
evaluator is quietly choosing for you.

## Run it

```bash
python scripts/eval.py \
  --load_path logs/my_student/dagger_final.pth \
  --out_dir logs/my_eval --side right \
  --data_idx '["rt/0416_grasp/cube_small_1","rt/0416_grasp/cube_small_2",
               "rt/0420_manip/squeegee_1","rt/0420_manip/squeegee_2"]' \
  --num_envs 64 --max_episodes 200 \
  --camera_extrinsic calib/camera_align/current.npy --headless
```

The student's observation layout is restored from the checkpoint automatically,
and the run aborts if the live slot map disagrees with the saved one.

## Two defaults that used to be wrong

### Collecting the first N episodes biases the result

A successful episode ends sooner than a failing one. Collect "the first 200
episodes to finish" and whichever demonstration finishes fastest contributes the
most of them. Measured here, a 200-episode budget split **41 / 81 / 37 / 41**
across four demos.

The aggregate then depends on an accident of timing. And the direction is not
predictable: under randomization the biased estimate came out **2.5 points low**,
because the over-represented demo happened to be the weakest one.

`--per_demo_quota` now defaults to `ceil(max_episodes / n_demos)`. The summary
reports closest-approach `success_rate_per_demo`, a demo-averaged **macro** rate and an
episode-weighted **micro** rate; with the quota they coincide, which is the point.

Pass `--per_demo_quota 0` for the old behaviour if you need to reproduce an old
number.

### A clean evaluation hides what tactile is for

Physical domain randomization is disabled by default so the evaluation is
deterministic. But that also removes the only variation contact force could help
with: a policy cannot demonstrate a benefit from sensing grip force when every
object weighs exactly its nominal mass.

```bash
--keep_physics_dr    # mass, friction, COM and PD gains stay randomized
```

Measured on the same checkpoint, 200 balanced episodes:

| | success |
|---|---|
| clean | 95.5% |
| `--keep_physics_dr` | 82.0% |

Neither number is wrong. They answer different questions, and a paper that reports
only the first is answering the easier one.

## The number to report is strict3, and the script now computes it

The env's own success flag means "reached the end of the trajectory without a
failure termination". That is survival, not task success. The protocol metric,
**strict3**, checks whether the object finished within 3 cm of the demo's
final pose, with no object-position drift, excluding bad inits
(`survival_len <= 5`).

The separate `success_rate_*` fields in this evaluator check whether the object
came within `--success_dist` at any point in the episode. They are closest-approach
rates, not the env's trajectory-completion flags.

```
[EvalPC] strict success (end_final_dist < N cm, no obj_pos_drift, bad inits excluded: 1/200):
    strict2   134/199  ( 67.3%)
    strict3   160/199  ( 80.4%)
    strict5   179/199  ( 89.9%)
```

Note that strict3 deliberately excludes **only** object-position drift, not the
other failure causes. ORing them all in would change what the number means
without a reader being able to see it. It does not require the env's success
flag. Training's `Strict` is a separate conservative proxy: it requires successful
trajectory completion and counts bad inits as zeros rather than removing them
from the denominator. See [TRAINING.md](../../docs/TRAINING.md).

## Read the per-demo breakdown

The aggregate hides the interesting part. From the same run:

```
               closest       strict3
cube_small_1      96.0%        86.0%
cube_small_2      92.0%        89.8%
squeegee_1        86.0%        84.0%
squeegee_2        98.0%        62.0%     <-- 36 points
```

`squeegee_2` frequently approaches the target but has lower endpoint accuracy.
An aggregate, on either metric alone, hides that distinction. These historical
numbers illustrate the breakdown; rerun evaluation after metric fixes for
release results.

## Check

The summary JSON carries `per_demo_quota`, `success_rate_per_demo`,
`success_rate_macro` and `success_rate_micro`. If macro and micro differ, your
episodes are unbalanced and you should say so when you quote the number.
