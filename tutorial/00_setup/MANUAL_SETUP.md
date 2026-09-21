# Manual environment setup

`setup_env.sh` automates most of this. Follow this page when the script fails,
when you want to understand what it is doing, or when your CUDA / driver
combination needs a different torch than the one Isaac Lab picks.

## The one rule that matters

> **torch, Isaac Lab and pytorch3d must agree.**

Nothing else about the torch version is important. Do not pin it to a number
from this document — pin it to *whatever Isaac Lab installs on your machine*,
then make pytorch3d match that.

You can see the coupling in the package name itself:

```
torch      2.5.1+cu118
pytorch3d  0.7.8+5043d15pt2.5.1cu118
                      ^^^^^^^^^^^^^  torch 2.5.1, CUDA 11.8
```

If those two halves disagree, pytorch3d imports and then segfaults or silently
produces wrong results — it is compiled against torch's ABI.

## Workspace layout

Isaac Lab expects Isaac Sim to sit inside it as `_isaac_sim`. Lay the workspace
out like this and open the outer folder in your editor:

```
<workspace>/
  IsaacLab/
    _isaac_sim/      -> symlink to your Isaac Sim install
  dexx_release/      <- this repository
```

## 1. Isaac Sim

Download [Isaac Sim 4.5.0](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html)
and unzip it somewhere. Call that path `$ISAACSIM_PATH`.

This is the one piece nothing can automate: the build has to match your NVIDIA
driver.

## 2. Conda environment and Isaac Lab

```bash
conda create -n dexx python=3.10
conda activate dexx

git clone git@github.com:isaac-sim/IsaacLab.git     # SSH recommended
cd IsaacLab
ln -s $ISAACSIM_PATH _isaac_sim

./isaaclab.sh -c dexx        # point Isaac Lab at this conda env
./isaaclab.sh -i none        # install WITHOUT the extra RL libraries

conda activate dexx          # reactivate so PYTHONPATH is picked up
echo $PYTHONPATH
```

`-i none` matters. The other options pull in RL frameworks this repository does
not use and that can drag in a conflicting torch.

**Isaac Lab installs torch.** Whatever it chose is now the version everything
else has to match:

```bash
python -c "import torch; print(torch.__version__)"
# e.g. 2.5.1+cu118
```

## 3. pytorch3d, matched to that torch

pytorch3d has no matching wheel on PyPI for most torch builds. Use the
prebuilt index, and construct the version string from what you just printed:

```bash
# torch 2.5.1+cu118  ->  ...pt2.5.1cu118
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder \
    pytorch3d==0.7.8+5043d15pt2.5.1cu118
```

The pattern is `pytorch3d==0.7.8+5043d15pt<TORCH_VERSION><CUDA>`, where
`<CUDA>` is `cu118`, `cu121`, … with no dot. Browse
<https://miropsota.github.io/torch_packages_builder> to find the build for your
torch if the one above does not exist.

pytorch3d is not optional here: the demonstration loader uses
`pytorch3d.ops.sample_points_from_meshes` and `pytorch3d.transforms`, so
training cannot start without it.

## 4. The two `--no-deps` packages

Both would otherwise pull their own torch and break the agreement from step 2:

```bash
pip install git+https://github.com/otaheri/chamfer_distance --no-deps
pip install git+https://github.com/KailinLi/bps_torch.git --no-deps
```

`bps_torch` produces the 128-d object shape encoding in the teacher's
observation; `chamfer_distance` is used by the retargeting loss.

## 4b. simple-raycaster — required, and easy to miss

The point-cloud environment renders depth with
`simple_raycaster.MultiMeshRaycaster`, constructed unconditionally in
`FrankaSharpaPointCloudEnv.__init__`. Without it the entire student pipeline
fails at env creation.

```bash
pip install --no-deps git+https://github.com/Agent-3154/simple-raycaster.git
```

`--no-deps` again: it declares `torch` and would undo step 2. Its real runtime
imports are `warp`, `mujoco`, `jaxtyping` and `trimesh`, which
`requirements.txt` carries — so install it before step 5, not after.

This one went undeclared for a while and worked anyway, because it happened to
be pip-installed from a local checkout. `check_imports.py` (step 6) exists
because of it.

## 5. This repository

```bash
cd <workspace>/dexx_release
pip install -e .
pip install -r requirements.txt
```

## 6. Verify

Three checks, none of which starts the simulator:

```bash
python tutorial/00_setup/check_install.py         # files, constants, calibrations
python tutorial/00_setup/check_portable.py        # no paths pointing off this machine
python tutorial/00_setup/check_imports.py         # every package the CODE imports
python tutorial/01_frames_and_constants/check_frames.py
```

`check_imports.py` walks the source, collects every third-party import, and tries
each one. It reports three categories: importable, hardware-only (ROS, Polymetis,
the Sharpa SDK — absent on a workstation by design), and installed-but-needs-the-
Isaac-app (`isaaclab_tasks` raises `ModuleNotFoundError: omni.physics` until
`AppLauncher` has run, which is why the scripts import it after that line). Only
genuinely absent packages are failures.

Then one that does:

```bash
bash tutorial/run_acceptance.sh                   # ~6 min, trains and evaluates
```

If the acceptance test passes, the environment is correct. Nothing short of it
proves that — `import torch` succeeding says very little.

## Known-good combination

Read off a machine where the full pipeline runs. Use it as a reference point,
not as a target:

| | version |
|---|---|
| python | 3.10 |
| Isaac Sim | 4.5.0 |
| isaaclab | 0.46.3 |
| torch | 2.5.1+cu118 |
| pytorch3d | 0.7.8+5043d15pt2.5.1cu118 |
| numpy | 1.26.4 |
| gymnasium | 0.29.1 |
| GPU | RTX 4090, driver 580.x |

## Troubleshooting

**`./isaaclab.sh -c` does nothing useful.** It has to run from the IsaacLab
checkout, with the conda env already active.

**`import pytorch3d` segfaults, or works but gives nonsense.** The build does not
match your torch. Re-read step 3 and compare the two halves of the version
string.

**`set -u` kills the shell on `conda activate`.** Isaac Sim ships an activation
hook that reads `$ZSH_VERSION` unguarded. Relax `set -u` around the activate
call — `setup_env.sh` does this.

**Training starts and immediately errors in the dataset loader.** Usually
pytorch3d missing or mismatched; it is imported at module load in
`tasks/hand_imitation/dataset/`.
