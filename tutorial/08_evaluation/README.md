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
reports `success_rate_per_demo`, a demo-averaged **macro** rate and an
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

## Read the per-demo breakdown

The aggregate hides the interesting part. From the same run:

```
cube_small_1   clean 100.0%   DR 76.0%
cube_small_2   clean  90.0%   DR 82.0%
squeegee_1     clean  92.0%   DR 84.0%
squeegee_2     clean 100.0%   DR 86.0%
```

The demo that is perfect when clean is the one that degrades most under
randomization. An aggregate would have hidden that entirely.

## Check

The summary JSON carries `per_demo_quota`, `success_rate_per_demo`,
`success_rate_macro` and `success_rate_micro`. If macro and micro differ, your
episodes are unbalanced and you should say so when you quote the number.
