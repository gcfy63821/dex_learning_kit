# 00 — Setup, repo map, and a smoke test

**Goal:** get the environment installed and prove it by running something, not by
reading a version number.

## Install

```bash
bash tutorial/00_setup/setup_env.sh \
    --isaacsim /path/to/isaac-sim \
    --isaaclab /path/to/IsaacLab
```

It checks your GPU, creates a conda environment, links Isaac Sim into Isaac Lab,
installs Isaac Lab **without** the extra RL libraries, then matches pytorch3d to
whatever torch that produced, installs the two `--no-deps` packages and this
repository — and finally **runs the verification checks**.

Variants:

```bash
--name myenv                    # a different environment name
--verify-only                   # check an existing env, install nothing
PYTORCH3D_BASE=0.7.8+5043d15    # override the pytorch3d base version
```

**[MANUAL_SETUP.md](MANUAL_SETUP.md)** is the same procedure done by hand. Read
it when the script fails, or before trusting it.

### Isaac Sim is the one part you install yourself

It ships through NVIDIA's own installer and the build has to match your driver.
Download [Isaac Sim 4.5.0](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html),
unzip it, and pass the path as `--isaacsim`.

### The version rule

> **torch, Isaac Lab and pytorch3d must agree.** Nothing else about the torch
> version matters.

Do not pin torch to a number from any document — Isaac Lab installs it, and
pytorch3d then has to be the build compiled against *that* torch. The coupling is
visible in the package name:

```
torch      2.5.1+cu118
pytorch3d  0.7.8+5043d15pt2.5.1cu118
                      ^^^^^^^^^^^^^
```

The script derives the right string automatically; MANUAL_SETUP.md step 3 shows
how to do it by hand.

### Known-good combination

A reference point, not a target: python 3.10, Isaac Sim 4.5.0, isaaclab 0.46.3,
torch 2.5.1+cu118, pytorch3d 0.7.8+5043d15pt2.5.1cu118, numpy 1.26.4, gymnasium
0.29.1, on an RTX 4090. `requirements-full.txt` is the frozen snapshot.

## Check it

```bash
python tutorial/00_setup/check_install.py
```

This imports the package, resolves the task registry, and confirms the assets,
demonstrations, camera calibrations and pretrained teacher are all present and
consistent with each other. It does **not** start Isaac Sim, so it takes a second
and tells you about missing files before a twenty-minute run does.

## Check the code's dependencies

```bash
python tutorial/00_setup/check_imports.py
```

Walks the source, collects every third-party import, and tries each one —
derived from the code, not from `requirements.txt`. A package can be missing
from requirements and still imported at module load; you then find out twenty
minutes into a run. `simple_raycaster` was exactly that: a hard requirement of
the point-cloud env, in no requirements file, working only because it happened
to be installed locally.

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
