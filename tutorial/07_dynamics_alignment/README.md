# 07 — Dynamics: aligning the simulated and real plant

**Goal:** make the machine the policy trains against behave like the machine it
will be deployed on — stiffness, damping, rotor inertia, latency and filtering.

**Why it matters:** a policy is a function fitted to a plant. Change the plant and
you have changed the problem. Vision errors announce themselves; a stiffness
mismatch just makes the robot shake, and everyone blames the policy.

## The four things that define the plant

### 1. Arm impedance

The real Franka runs a joint-impedance controller. Polymetis is started with a
gain vector; the simulator's `ImplicitActuator` is configured with the matching
per-joint stiffness and damping.

| joint | sim stiffness (`kp`) | sim damping (`kd`) |
|---|---|---|
| fr3_joint1 | 1600 | 145 |
| fr3_joint2 | 1600 | 135 |
| fr3_joint3 | 1200 | 110 |
| fr3_joint4 | 800 | 100 |
| fr3_joint5 | 500 | 50 |
| fr3_joint6 | 300 | 30 |
| fr3_joint7 | 150 | 15 |

Set in `FrankaSharpaCriticHorizonCfg.__post_init__`; step-response tuned, and what
every shipped checkpoint was trained against. Do not change them casually — a
checkpoint is only valid for the plant it was fitted to.

The real side takes the same numbers:

```bash
# cfg fields, settable from the CLI
--env_cfg polymetis_kq='[1600,1600,1200,800,500,300,150]' \
          polymetis_kqd='[145,135,110,100,50,30,15]'
```

Leave them `None` to accept the Polymetis defaults — which is a decision, not a
neutral option, and should be a deliberate one.

The stiffness profile is steeply decreasing from base to wrist. That is not
arbitrary: the proximal joints carry the whole arm's inertia, the distal ones
almost none, so equal gains give wildly different closed-loop bandwidths.

### 2. Hand gains and rotor inertia

`src/dexx/robot_constants.py`, read out of the vendor's USD and converted from its
per-degree drive convention. Per joint role:

| role | stiffness | damping | armature | friction |
|---|---|---|---|---|
| thumb CMC AA | 13.201 | 0.4516 | 0.00320 | 0.1320 |
| MCP FE | 4.760 | 0.1830 | 0.00265 | 0.1040 |
| MCP AA | 6.622 | 0.2080 | 0.00265 | 0.1040 |
| PIP | 0.908 | 0.0400 | 0.00061 | 0.0248 |
| DIP | 0.904 | 0.0315 | 0.00042 | 0.0004 |

`armature` is the load-bearing entry — lesson 02 has the failure it prevents.
Note that effort and velocity limits are deliberately **not** overridden here: the
URDF's own values are correct, and overriding them is what allows bad gains to act
instead of saturating.

### 3. The temporal chain

Between the policy's output and the joint, four things happen. Every one of them
exists to match a real delay.

| stage | value | matching |
|---|---|---|
| action clip | ±1.0 | |
| **action delay** | 0–3 steps, sampled per environment | ≈100 ms at 30 Hz, end to end |
| arm: delta scale | 0.2 | was 0.1 at 60 Hz; doubled for 30 Hz to keep arm speed |
| **arm EMA** | α = 0.15 | the real Franka impedance loop's ~50 ms low-pass |
| **hand EMA** | α = 0.4 | |
| saturate to joint limits | | |

Two separate filters, deliberately. The arm is filtered harder because its
controller is slower and because sim2real arm shake was traced to high-frequency
policy jitter getting through.

Order matters: the delay is applied to the **raw 29-d action before** it is split
into arm and hand, and before either filter. Getting that order wrong changes the
effective latency.

### 4. Randomization around all of it

The point is not to match one plant exactly — it is to make the policy indifferent
to the range your real plant could be in.

| quantity | range |
|---|---|
| arm PD gains | ×[0.80, 1.20] |
| hand PD gains | ×[0.5, 2.0] |
| object mass | 0.01–0.15 kg |
| friction | ×[1.0, 2.5] |
| centre of mass | ±2 cm per axis |
| action delay | 0–3 steps |

The arm band is deliberately narrower than the hand's: you measure the arm's gains,
so the uncertainty is small; the hand's effective gains vary with temperature,
wear and payload.

## Measuring the gap: chirp replay

`tools/sysid/` replays the same motion in simulation and on the real arm, then
diffs them. This is the instrument the gain table above was tuned with.

### The motions

`tools/sysid/motions/` holds pre-generated trajectories, each a `.csv` of joint
targets plus a `.json` of metadata. All start and end at a safe middle pose so
they chain without discontinuities, and all passed an FR3 safety check at 70% of
the velocity and acceleration spec.

| motion | what it is for |
|---|---|
| **`chirp_sweep`** | linear 0.2 → 3 Hz sweep, 0.15 rad amplitude, 7 joints phase-shifted, 36.6 s. Excites the whole bandwidth in one run — the default choice |
| `step_per_joint` | rise time and overshoot, one joint at a time |
| `sin_j1` … `sin_j6` | single-joint damping ratio, cleanest signal per joint |
| `backlash_detection` | direction reversals, for gearbox slop |
| `coupled_joints`, `diagonal_sweep`, `circular_wrist` | multi-joint coupling the single-axis PD cannot fix |

`motions_hand/` has the hand equivalents (`hand_chirp_all`, `hand_grasp_cycle`, …).

Regenerate or add motions with `generate_motions.py` / `generate_hand_motions.py`.

### Run it

```bash
# 1. simulation — uses the TRAINING gains by default
python tools/sysid/replay_motion_sim.py \
    --motion tools/sysid/motions/chirp_sweep.csv \
    --output logs/sysid/chirp_sim.pkl

# 2. the real arm, through Polymetis (bridge up, no other controller running)
python tools/sysid/replay_motion_polymetis.py \
    --ip <NUC_IP> \
    --motion tools/sysid/motions/chirp_sweep.csv \
    --output logs/sysid/chirp_real.pkl

# 3. diff them
python tools/sysid/analyze_motion.py \
    --sim  logs/sysid/chirp_sim.pkl \
    --real logs/sysid/chirp_real.pkl \
    --out  logs/sysid/compare_chirp
```

Both replays write the same pkl schema, which is what lets step 3 diff them
without caring which side is which.

**Start with `--dry_run`** on the real replay. It prints the trajectory and the
start-pose delta without commanding anything.

### What it saves

`--out` is a directory, and everything lands in it:

| file | contents |
|---|---|
| `metrics.csv` | one row per joint — the numbers you quote |
| `overlay_all_joints.png` | sim and real actuals over time, per joint |
| `errors_per_joint.png` | the residual, per joint |

The replay pkls themselves record `sim_config.arm_K` / `arm_D` read back off the
live actuator, so a saved run says which gains produced it. You can re-analyse an
old pair months later and know what it was measuring.

### Reading `metrics.csv`

| column | meaning |
|---|---|
| `max_lag_ms` | **the one that matters.** Negative = sim **lags** real ⇒ sim is over-damped, cut damping. Positive = sim **leads** real ⇒ add `action_delay_steps` |
| `rmse_rad`, `rmse_deg` | position error between sim and real actuals |
| `corr`, `cos_sim` | shape agreement, insensitive to a constant offset |
| `rmse_sim_tracking` / `rmse_real_tracking` | each side against **its own** target — how well either tracks at all |
| `active` | false when a joint barely moves in this motion; its metrics are nulled rather than left to mislead |

The sign convention on `max_lag_ms` was wrong in an earlier version of the
analyzer and produced exactly the wrong advice — it suggested adding latency when
the fix was less damping. If you port this code, keep the convention and its test.

### Evaluating a candidate gain set

```bash
python tools/sysid/replay_motion_sim.py \
    --motion tools/sysid/motions/chirp_sweep.csv \
    --arm_kd 85,135,110,25,18,10,5 \
    --output logs/sysid/chirp_sim_candidate.pkl
```

Re-analyse against the same real recording. The real arm does not need to move
again — that is the point of recording it.

### What the recorded runs say

Two alignments are on file. The second is worth knowing about because **it was
never adopted**.

*2026-04, under ROS2.* Single-joint sine showed sim lagging real by 40–50 ms on
j1–j3 (sim ζ≈1.34, real ζ≈0.3). Damping cut 240/220/180 → **145/135/110**. Lag
fell to 0–10 ms, correlation 0.993 → 0.996–0.999. That is the `ARM_TUNED_KD`
this repository trains with.

*2026-07, after moving to Polymetis.* The plant changed, and j4–j7 were found
still lagging. A second realignment produced `ARM_KD_POLYMETIS_IT2`:

| joint | training KD | it2 KD | lag before | lag after |
|---|---|---|---|---|
| j1 | 145 | 85 | −40 ms | **−20 ms** |
| j2 | 135 | 135 | +10 ms | +10 ms |
| j3 | 110 | 110 | +20 ms | +20 ms |
| j4 | 100 | 25 | −90 ms | **−30 ms** |
| j5 | 50 | 18 | −50 ms | **−20 ms** |
| j6 | 30 | 10 | −60 ms | **−20 ms** |
| j7 | 15 | 5 | −70 ms | **−30 ms** |

RMSE was neutral to better (j4 .0161 → .0117; j5 and j7 slightly worse). It is a
clear improvement on the metric the exercise targets, and it is **not** what
training uses — adopting it invalidates every shipped checkpoint, because a
policy is a function fitted to a plant. Both sets are named constants in
`franka_sharpa_critic_horizon_cfg.py`; the choice is deliberate and reversible.

### The residual that gain tuning cannot fix

Multi-joint chirp kept +70 to +90 ms on j2/j3 after the single-axis tuning
converged. That is inter-joint coupling, and no diagonal PD fixes it. It is why
the arm PD randomization band exists (§4) — the policy is made indifferent to the
part that cannot be matched.

## How to actually align a new arm

1. **Measure, do not guess.** Command a step on one joint at a time and record the
   response. `tune_arm_gains.py` in the development repository sweeps Kq/Kqd on
   hardware.
2. **Match rise time and overshoot**, in that order. Stiffness sets rise time;
   damping sets overshoot. Fit the simulator to the real response, not the other
   way around.
3. **Then measure latency.** Command a step and timestamp when the state reflects
   it. Set `action_delay_max` to cover the worst case you measured, not the mean.
4. **Then set the EMA** so the simulated command spectrum resembles what the real
   controller actually passes. An arm that tracks a 50 ms low-pass will not follow
   an unfiltered 30 Hz command however stiff it is.
5. **Randomize around the measurement**, not around a guess.

The order is deliberate: latency measured against badly-tuned gains is a
measurement of the gains.

## The symptom table

| what you see on hardware | look at |
|---|---|
| arm shakes, sim is smooth | arm EMA too high; gains too stiff; delay under-modelled |
| arm lags, overshoots the target | damping too low; delay over-modelled |
| fingers fail to close on contact | hand armature missing (lesson 02); collision filters lost |
| grasp slips under load | object mass/friction randomization too narrow |
| works on light objects only | mass range does not cover your real object |

## Porting

**Different arm:** the seven-joint gain table, the `7 + 22` action split, the
armature and friction constants, and `ARM_BASE_POS`. Re-measure the step response;
do not port the numbers.

**Different hand:** the per-role gain table and armature. Assume every value needs
re-deriving from the vendor asset — the drive convention (per-degree vs per-radian)
is a factor-of-57 trap.

**Same hardware, new mount:** nothing here changes. This lesson is about the
machine, not its pose.
