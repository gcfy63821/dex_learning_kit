"""Verify every third-party package the code imports is actually installed.

Derived from the source, not from requirements.txt — which is the point. A
dependency can be missing from requirements and still be imported at module
load; you then find out twenty minutes into a training run, or not until the
point-cloud env is constructed.

`simple_raycaster` is the case this exists for: a hard requirement of the
point-cloud env that appeared in no requirements file and only worked because it
happened to be installed from a local checkout on one machine. It passed every
other check in this directory, because those look at files and constants rather
than at what the code imports. On a fresh machine this check fails loudly
instead.

    python tutorial/00_setup/check_imports.py
"""
from __future__ import annotations

import ast
import importlib
import importlib.metadata as md
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCAN = ("src/dexx", "scripts", "tools", "deploy")
SKIP_DIRS = {"__pycache__", ".git"}

FIRST_PARTY = {"dexx"}

# Expected to be absent on a workstation: they live on the robot, the camera
# host, or inside Isaac Sim's own interpreter. Missing here is not an error.
HARDWARE_ONLY = {
    "polymetis", "rclpy", "sensor_msgs", "geometry_msgs", "std_msgs",
    "visualization_msgs", "cv_bridge", "rosgraph_msgs", "builtin_interfaces",
    "tf2_ros", "tf2_geometry_msgs", "tf2_py", "ament_index_python",
    "rclpy_message_converter", "nav_msgs", "trajectory_msgs",
    "pyrealsense2", "sharpa", "sharpa_sdk", "omni", "pxr", "isaacsim", "carb",
    "usdrt",
}

# import name -> what to install, when the two differ
PIP_NAME = {
    "warp": "warp-lang",
    "simple_raycaster": "git+https://github.com/Agent-3154/simple-raycaster.git (--no-deps)",
    "bps_torch": "git+https://github.com/KailinLi/bps_torch.git (--no-deps)",
    "chamfer_distance": "git+https://github.com/otaheri/chamfer_distance (--no-deps)",
    "pytorch3d": "see MANUAL_SETUP.md step 3 — must match your torch",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
}


def top_level_imports() -> dict[str, set[str]]:
    """module name -> set of files that import it."""
    found: dict[str, set[str]] = {}
    for rel in SCAN:
        base = os.path.join(ROOT, rel)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for f in filenames:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dirpath, f)
                try:
                    tree = ast.parse(open(p, encoding="utf-8", errors="ignore").read())
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for a in node.names:
                            found.setdefault(a.name.split(".")[0], set()).add(
                                os.path.relpath(p, ROOT))
                    elif isinstance(node, ast.ImportFrom):
                        if node.level:            # relative import, first-party
                            continue
                        if node.module:
                            found.setdefault(node.module.split(".")[0], set()).add(
                                os.path.relpath(p, ROOT))
    return found


def main() -> int:
    stdlib = getattr(sys, "stdlib_module_names", set())
    found = top_level_imports()

    third_party = {
        m: files for m, files in found.items()
        if m not in stdlib and m not in FIRST_PARTY and not m.startswith("_")
    }

    missing, hardware, isaac_runtime, ok = [], [], [], []
    for m in sorted(third_party):
        if m in HARDWARE_ONLY:
            hardware.append(m)
            continue
        try:
            importlib.import_module(m)
            ok.append(m)
            continue
        except Exception as e:  # noqa: BLE001
            err = type(e).__name__
        # An import can fail because the package is absent, or because it is
        # present but only usable once Isaac Sim's app is running (the `omni.*`
        # modules appear then). Those are different problems and only the first
        # is an installation error: `isaaclab_tasks` is pip-installed and still
        # raises ModuleNotFoundError('omni.physics') until AppLauncher has run.
        installed = False
        try:
            installed = importlib.util.find_spec(m) is not None
        except Exception:  # noqa: BLE001
            installed = False
        if not installed:
            for dist in md.distributions():
                names = (dist.read_text("top_level.txt") or "").split()
                if m in names:
                    installed = True
                    break
        if installed:
            isaac_runtime.append((m, err))
        else:
            missing.append((m, err, sorted(third_party[m])[:2]))

    print(f"=== {len(third_party)} third-party modules imported by the code")
    print(f"  {len(ok)} importable, {len(hardware)} hardware-only, "
          f"{len(isaac_runtime)} need the Isaac app, {len(missing)} missing")

    if hardware:
        print("\n  hardware-only, not expected on a workstation:")
        print("    " + ", ".join(hardware))

    if isaac_runtime:
        print("\n  installed, but only importable once Isaac Sim's app is running")
        print("  (the scripts import these after AppLauncher, so this is fine):")
        for m, err in isaac_runtime:
            print(f"    {m}  ({err})")

    if missing:
        print("\n=== MISSING")
        for m, err, files in missing:
            how = PIP_NAME.get(m, f"pip install {m}")
            print(f"  {m}  ({err})")
            print(f"     install: {how}")
            print(f"     used by: {', '.join(files)}")
        print("\nNOT READY — the code imports packages this environment does not have.")
        return 1

    print("\nALL IMPORTS RESOLVE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
