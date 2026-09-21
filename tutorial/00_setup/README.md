# 00 — Setup, repo map, and a smoke test

**Goal:** get the environment installed and prove it by running something, not by
reading a version number.

## Install

One script does everything except Isaac Sim:

```bash
bash tutorial/00_setup/setup_env.sh --isaaclab /path/to/IsaacLab
```

It checks your GPU and driver, creates a conda environment, installs PyTorch and
this package, and then **runs the verification checks** — so it tells you whether
the machine is ready rather than whether the commands exited zero.

Useful variants:

```bash
bash tutorial/00_setup/setup_env.sh --name myenv          # different env name
bash tutorial/00_setup/setup_env.sh --name myenv --verify-only   # check, install nothing
```

`--verify-only` is for confirming an existing machine is ready without touching
what is installed on it.

### Isaac Sim is the one part you install yourself

It ships through NVIDIA's own installer and its version has to match your driver.

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && ./isaaclab.sh --install
```

Then re-run `setup_env.sh` with `--isaaclab` pointing at that checkout.

### Known-good versions

These are read off a machine where the whole pipeline runs, not off a changelog.

| | version |
|---|---|
| python | 3.10 |
| torch | 2.5.1+cu118 |
| isaaclab | 0.46.3 (Isaac Sim 4.5) |
| numpy | 1.26.4 |
| gymnasium | 0.29.1 |
| GPU used for the numbers in these lessons | RTX 4090, 24 GB |

`requirements-full.txt` is a frozen snapshot of that environment if you need to
reproduce it exactly; `requirements.txt` is the curated set.

## Check it

```bash
python tutorial/00_setup/check_install.py
```

This imports the package, resolves the task registry, and confirms the assets,
demonstrations, camera calibrations and pretrained teacher are all present and
consistent with each other. It does **not** start Isaac Sim, so it takes a second
and tells you about missing files before a twenty-minute run does.

## Check it can move

```bash
python tutorial/00_setup/check_portable.py
```

Answers a different question: if you copied this directory to another machine,
would it still work? It looks for symlinks that leave the repository, absolute
paths baked into code or config, and demonstrations whose source data or meshes
are not actually committed.

Run it after adding a demonstration or an asset. The first time it ran it found
four external symlinks — three of the four shipped demonstrations resolved only
on the author's machine — and eleven absolute paths. Nothing had failed; the
repository simply could not have been used anywhere else.

## What runs where

Three machines are involved once you reach hardware. In simulation, only the
first exists.

| | runs | lessons |
|---|---|---|
| **workstation** | training, evaluation, the policy at deploy time | 00–08 |
| **NUC** | Polymetis server + the ZMQ joint bridge, wired to the Franka | 09 |
| **camera host** | RealSense depth publisher (ZMQ) | 06, 09 |

## The shape of the pipeline

```
human demo (MANO)
   │  lesson 03 — retarget
   ▼
robot reference trajectory  ────────┐
   │  lesson 04 — PPO                │ the reference is an input to
   ▼                                 │ every stage after this one
state expert  (sees privileged state)│
   │  lesson 05 — DAgger             │
   ▼                                 │
point-cloud student  ◄───────────────┘
   │  lesson 08 — evaluate
   │  lessons 06, 07 — calibrate the camera, align the dynamics
   ▼
real robot  (lesson 09)
```

The expert is never deployed. It exists to label the student, because it is
allowed to see things a camera cannot — object velocity, contact state, the
randomized physical parameters. Lesson 04 makes that split concrete.

## Check

`check_install.py` prints `ALL CHECKS PASSED`. If it does not, it names the file
it wanted and the lesson that explains what that file is for.
