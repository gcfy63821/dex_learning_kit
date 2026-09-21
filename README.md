# Dex-X

Dexterous manipulation reinforcement learning for a **Sharpa Wave** dexterous hand (22 DOF)
mounted on a **Franka FR3** arm (7 DOF), trained in **Isaac Lab** to imitate human hand
manipulation demos, with a sim-to-real deployment path.

The release pipeline distills a privileged **PPO teacher** into a deployable
**point-cloud + tactile student** via DAgger, then fine-tunes with PPO.

## Pipeline

```
  retarget            PPO teacher              DAgger (PointCloud)          PPO fine-tune            deploy (optional)
  MANO → Sharpa   →   force-poseobs (557d)  →  proprio + PointNet student  →  asymmetric PPO      →  Polymetis + ZMQ
  scripts/retarget    scripts/train_teacher    scripts/train_dagger_pc        scripts/train_ppo_pc    deploy/
```

## Start here

**[`tutorial/`](tutorial/)** — ten lessons that build the pipeline in the order you
would actually rebuild it, each with a check you can run. Lessons 00–05 and 08 run
end to end on one GPU with the demonstrations this repository ships.

| if you want to | read |
|---|---|
| understand the pipeline | [00](tutorial/00_setup/) → [01](tutorial/01_frames_and_constants/) → [03](tutorial/03_retarget/) → [04](tutorial/04_teacher/) → [05](tutorial/05_distillation/) → [08](tutorial/08_evaluation/) |
| move the table, camera or robot mount | [01](tutorial/01_frames_and_constants/) → [06](tutorial/06_camera_calibration/) → [08](tutorial/08_evaluation/) |
| use a different arm or hand | [02](tutorial/02_assets_and_joints/) → [01](tutorial/01_frames_and_constants/) → [07](tutorial/07_dynamics_alignment/) → [03](tutorial/03_retarget/) |
| add a new task or object | [03](tutorial/03_retarget/) → [04](tutorial/04_teacher/) → [05](tutorial/05_distillation/) |
| get it onto hardware | [01](tutorial/01_frames_and_constants/) → [06](tutorial/06_camera_calibration/) → [07](tutorial/07_dynamics_alignment/) → [09](tutorial/09_deploy/) |

`docs/` stays the reference for details; the tutorial is the path through it.
See **[docs/PIPELINE.md](docs/PIPELINE.md)** for a condensed end-to-end walkthrough.

## Install

```bash
bash tutorial/00_setup/setup_env.sh --isaaclab /path/to/IsaacLab
```

Creates the conda environment, installs everything except Isaac Sim, and runs the
verification checks. See [tutorial/00](tutorial/00_setup/) for Isaac Sim itself,
the known-good versions, and `--verify-only` for checking a machine you already
set up.

## Quickstart

```bash
# 1. retarget one demo (MANO → Sharpa)
python scripts/retarget.py --side right --data_idx rt/0416_grasp/cube_small_2 --headless

# 2. train the PPO teacher
python scripts/train_teacher.py --task franka-sharpa-force-poseobs --side right \
    --data_idx '["rt/0416_grasp/cube_small_2"]' --num_envs 2048 --headless

# 3. distill a point-cloud student (DAgger)
python scripts/train_dagger_pc.py --task franka-sharpa-pointcloud \
    --teacher_ckpt checkpoints/teacher_poseobs.pth --side right \
    --data_idx '["rt/0416_grasp/cube_small_2"]' \
    --camera_extrinsic calib/camera_align/current.npy --headless

# 4. evaluate  (add --keep_physics_dr for the harder, more honest number)
python scripts/eval.py --task franka-sharpa-pointcloud \
    --load_path <dagger_final.pth> --side right --out_dir logs/eval \
    --data_idx '["rt/0416_grasp/cube_small_2"]' \
    --camera_extrinsic calib/camera_align/current.npy --headless
```

`--camera_extrinsic` is not optional: without it the sim camera falls back to a
stale hard-coded pose ~20 cm from the real one. See
[tutorial/06](tutorial/06_camera_calibration/).

## Repository layout

| Path | What |
|---|---|
| `src/dexx/tasks/franka_sharpa/` | Isaac Lab environments (base → force → poseobs → pointcloud) |
| `scripts/retarget.py` | MANO → Sharpa retargeting (the whole optimizer lives here) |
| `src/dexx/tasks/hand_imitation/` | demo dataset loaders |
| `src/dexx/algo/ppo/` | PPO + asymmetric point-cloud PPO |
| `src/dexx/algo/dagger/` | DAgger point-cloud distillation |
| `src/dexx/algo/models/` | shared nets + running-mean-std |
| `scripts/` | training / eval / play / record entry points |
| `deploy/` | real-robot runtime (Polymetis + ZMQ + Sharpa SDK) |
| `assets/franka_fr3/` · `assets/sharpa_wave/` | upstream FR3 arm + Sharpa Wave hand models (Apache-2.0, vendored) |
| `assets/generated/` | merged FR3+Wave articulation URDF (built by `scripts/build_merged_urdf.py`) |
| `assets/ASSETS.md` | **asset provenance & licensing — read before changing assets** |
| `data/` | sample demos + retargeted trajectories |
| `checkpoints/` | pretrained teacher checkpoint |
| `calib/camera_align/` | camera extrinsics — pass one with `--camera_extrinsic` |
| `tools/calib/` | the camera-calibration chain (tutorial 06) |
| `tools/sysid/` | chirp / step motion replay for sim-real gain alignment (tutorial 07) |
| `tutorial/` | the ten lessons |

## Docs

- **[tutorial/](tutorial/)** — start here
- [PIPELINE.md](docs/PIPELINE.md) — end-to-end walkthrough
- [RETARGET.md](docs/RETARGET.md) · [TRAINING.md](docs/TRAINING.md) · [DISTILLATION.md](docs/DISTILLATION.md) · [EVAL.md](docs/EVAL.md) · [DEPLOY.md](docs/DEPLOY.md)
- [JOINT_ORDERING.md](docs/JOINT_ORDERING.md) — **critical** hand-joint ordering gotchas for sim2real
- [DEPLOY_CONFIG.md](docs/DEPLOY_CONFIG.md) — the sim2real constants, in one file
- [DEBUG_TOOLS.md](docs/DEBUG_TOOLS.md) — the calibration and debugging toolchain

## License

This project is released under the **MIT License** (see [LICENSE](LICENSE)).

`assets/franka_fr3/` (Franka FR3 description) and `assets/sharpa_wave/` (from
[sharpa-robotics/sharpa-urdf-usd-xml](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml))
are third-party content redistributed under the **Apache License 2.0**; their
`LICENSE` / `LICENSE.txt` / `NOTICE.txt` are retained in those directories and must not be removed.
See [assets/ASSETS.md](assets/ASSETS.md) for the full provenance of every asset.
