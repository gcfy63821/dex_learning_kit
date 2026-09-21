# Motion Files for Sim2Real Action Tracking

## Safety

All default motions are guaranteed to stay within **70% of FR3 position /
velocity / acceleration limits** (datasheet values hard-coded in
`generate_motions.py:FR3_{VEL,ACC}_LIMIT`):

- **Position**: each motion stays ≥ 20 mrad away from every joint's limit
  (notably joint 4 is one-sided `[-3.04, -0.15]` and joint 6 is `[0.54, 4.52]`).
- **Velocity**: peak < 0.7 × FR3 spec (j1-4: 2.175 rad/s, j5-7: 2.61 rad/s).
- **Acceleration**: peak < 0.7 × datasheet accel limit (j2 is most restrictive
  at 7.5 rad/s² spec → 5.25 usable).

All segment transitions use **minimum-jerk** profiles (zero velocity AND
zero acceleration at endpoints), sinusoids are wrapped with **min-jerk
fade-in/out envelopes** so they start and stop at zero velocity.

The generator runs `_safety_check()` on every motion and prints per-joint
vel/acc peaks. Before sending to real hardware, `replay_motion_real.py`
also runs `preflight_safety()` and refuses to publish unless the motion
passes (override: `--force_unsafe`, dangerous).



Canonical Franka FR3 joint-target trajectories used to calibrate the sim/real
action-tracking gap. Ported from the
[SAGE](https://github.com/NVIDIA-Isaac-Sim/sage) project (`motion_files/so101/custom/*`)
and adapted to 7-DOF FR3.

## Format

Each motion is two files:

| File | Content |
|------|---------|
| `{name}.csv` | Header row = joint names (`fr3_joint1..7`); each subsequent row = one target sample (rad) |
| `{name}.json` | Metadata — `control_freq_hz`, `duration_s`, `description`, `base_pose` |

Rows are at `control_freq_hz` (default 30 Hz, matching deploy). All motions
start and end at the safe home pose `[0, 0, 0, -1.57, 0, 1.57, 0]` to allow
chaining and safe re-entry.

## Regenerating

```bash
python rl_isaaclab/scripts/system_id/generate_motions.py \
    --output_dir rl_isaaclab/scripts/system_id/motions \
    --control_freq 30
```

Edit `generate_motions.py` to tweak amplitudes, frequencies, durations. Safer
to modify that script and regenerate than to hand-edit CSVs.

## Motions

| Name | Purpose |
|------|---------|
| `step_per_joint` | Sequential step up/center/down/center per joint (replicates `step_response_real.py`) |
| `chirp_sweep` | 0.2→3 Hz linear chirp, all joints phase-shifted. **Best single test** — full Bode from one run |
| `sin_j4`, `sin_j6` | Pure sinusoid on one joint (elbow / wrist pitch). Clean single-joint Bode |
| `circular_wrist` | Joints 5&6 circular motion. Detects wrist backlash/hysteresis |
| `backlash_detection` | Small reversals per joint — exposes dead-band / stiction |
| `diagonal_sweep` | All 7 joints in-phase sinusoid — coordinated motion stress test |
| `coupled_joints` | Anti-phase pairs (j1,-j3), (j2,-j4), (j5,-j7) — cross-joint torque coupling |

## Usage Pipeline

```bash
# 0. (once) Regenerate CSVs if you tweaked params
python rl_isaaclab/scripts/system_id/generate_motions.py

# 1. Real hardware — start controller first
ros2 launch franka_bringup dexhand_joint_impedance_controller.launch.py
python rl_isaaclab/scripts/system_id/replay_motion_real.py \
    --motion rl_isaaclab/scripts/system_id/motions/chirp_sweep.csv \
    --output logs/system_id/motion_replay/chirp_sweep_real.pkl

# 2. Sim
python rl_isaaclab/scripts/system_id/replay_motion_sim.py \
    --motion rl_isaaclab/scripts/system_id/motions/chirp_sweep.csv \
    --output logs/system_id/motion_replay/chirp_sweep_sim.pkl \
    --headless

# 3. Analyze
python rl_isaaclab/scripts/system_id/analyze_motion.py \
    --sim  logs/system_id/motion_replay/chirp_sweep_sim.pkl \
    --real logs/system_id/motion_replay/chirp_sweep_real.pkl \
    --out  logs/system_id/motion_replay/compare_chirp
```

Output:

- `metrics.csv` — per-joint RMSE, MAPE, corr, cos_sim, **max_lag_ms**, sim-tracking, real-tracking
- `overlay_all_joints.png` — target / sim / real overlay per joint
- `errors_per_joint.png` — residuals per joint
- console: suggested `action_delay_steps` based on mean lag

## Interpreting Key Metrics

- **`max_lag_ms > 0`**: sim lags real. Probably too much damping / too small K in sim,
  OR real has feed-forward velocity that sim lacks. Try ↑ sim K, ↓ sim D, or add vel-ff.
- **`max_lag_ms < 0`**: sim leads real. Real's internal delay (controller + ROS2) is larger
  than your `action_delay_steps`. Increase `action_delay_steps` by `round(|lag|/ctrl_dt)`.
- **`corr < 0.95`**: waveforms shape-mismatch (overshoot / ringing differ). Gains problem, not just lag.
- **`sim_trk ≫ real_trk`**: sim PD is too soft — increase K.
- **`sim_trk ≪ real_trk`**: sim PD is stiffer than real — decrease K (and/or add vel filter to sim).
