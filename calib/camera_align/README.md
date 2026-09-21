# Camera extrinsics

4x4 **camera-in-armbase** transforms in the ROS optical convention
(x-right, y-down, z-forward). Pass one to training and evaluation with
`--camera_extrinsic calib/camera_align/<file>.npy`.

| file | use |
|---|---|
| `current.npy` | the live calibration — use this unless reproducing an old run |
| `refined_extrinsic_manual.npy` | the extrinsic the 2026-08 reported runs were trained and evaluated with |

Without `--camera_extrinsic`, `visual_raycaster._build_camera_extrinsics()`
falls back to a hard-coded **2026-05-20** matrix. The camera has been moved
since: that fallback sits ~19 cm and ~4 deg away from `current.npy`, which is
enough to change what the scene point cloud contains without raising any error.
Always pass the file.
