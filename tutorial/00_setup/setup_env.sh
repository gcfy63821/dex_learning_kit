#!/usr/bin/env bash
# Create the conda environment for Dex-X and prove it works.
#
#   bash tutorial/00_setup/setup_env.sh --isaaclab /path/to/IsaacLab [--name dexx]
#   bash tutorial/00_setup/setup_env.sh --isaacsim /path/to/isaac-sim --isaaclab /path/to/IsaacLab
#   bash tutorial/00_setup/setup_env.sh --name <existing> --verify-only
#
# It does NOT pin a torch version. Isaac Lab installs torch; this script reads
# whatever that turned out to be and matches pytorch3d to it. See
# tutorial/00_setup/MANUAL_SETUP.md for the same procedure done by hand, and for
# what to do when a step fails.
#
# --verify-only skips every install step and just runs the checks against an
# environment you already have.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD

ENV_NAME=dexx
ISAACLAB=${ISAACLAB_PATH:-}
ISAACSIM=${ISAACSIM_PATH:-}
VERIFY_ONLY=0
PYTORCH3D_BASE=${PYTORCH3D_BASE:-0.7.8+5043d15}
PY_VER=3.10

while [ $# -gt 0 ]; do
  case "$1" in
    --name)        ENV_NAME=$2; shift 2 ;;
    --isaaclab)    ISAACLAB=$2; shift 2 ;;
    --isaacsim)    ISAACSIM=$2; shift 2 ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    -h|--help)     sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1"; exit 1 ;;
  esac
done

step () { echo; echo "=== $*"; }
die  () { echo "ERROR: $*" >&2; exit 1; }

step "0/6  prerequisites"
command -v conda >/dev/null || die "conda not found. Install Miniconda first."
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — this needs an NVIDIA GPU."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/  GPU: /'
echo "  conda: $(conda --version)"
if [ "$VERIFY_ONLY" -eq 0 ]; then
  [ -n "$ISAACLAB" ] || die "--isaaclab is required (or set ISAACLAB_PATH).
       Isaac Sim and Isaac Lab are installed separately; see
       tutorial/00_setup/MANUAL_SETUP.md steps 1-2."
  [ -x "$ISAACLAB/isaaclab.sh" ] || die "no isaaclab.sh under $ISAACLAB"
  echo "  IsaacLab: $ISAACLAB"
  if [ -n "$ISAACSIM" ]; then
    [ -d "$ISAACSIM" ] || die "--isaacsim path does not exist: $ISAACSIM"
    if [ -e "$ISAACLAB/_isaac_sim" ]; then
      echo "  _isaac_sim: already present, leaving it alone"
    else
      ln -s "$ISAACSIM" "$ISAACLAB/_isaac_sim"
      echo "  _isaac_sim: linked -> $ISAACSIM"
    fi
  elif [ ! -e "$ISAACLAB/_isaac_sim" ]; then
    die "$ISAACLAB/_isaac_sim is missing and --isaacsim was not given.
       Isaac Lab needs Isaac Sim linked in as _isaac_sim."
  fi
fi

step "1/6  conda environment '$ENV_NAME' (python $PY_VER)"
# conda activation hooks are hostile to `set -eu`: Isaac Sim's reads
# $ZSH_VERSION unguarded, and an env's deactivate.d may `unalias isaaclab`,
# which returns non-zero when the alias is absent and silently kills the script.
# Relax both flags around every conda call.
set +eu
source "$(conda info --base)/etc/profile.d/conda.sh"
set -eu
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "  exists — reusing it"
elif [ "$VERIFY_ONLY" -eq 1 ]; then
  die "environment '$ENV_NAME' does not exist (--verify-only creates nothing)"
else
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi
set +eu; conda activate "$ENV_NAME"; set -eu
echo "  python: $(python --version)"

if [ "$VERIFY_ONLY" -eq 0 ]; then

step "2/6  Isaac Lab (it brings its own torch)"
if python -c "import isaaclab" 2>/dev/null; then
  python -c "import isaaclab; print('  already installed:', isaaclab.__version__)"
else
  echo "  running: ./isaaclab.sh -c $ENV_NAME  &&  ./isaaclab.sh -i none"
  echo "  (-i none = no extra RL libraries; they can drag in a conflicting torch)"
  ( cd "$ISAACLAB" && ./isaaclab.sh -c "$ENV_NAME" && ./isaaclab.sh -i none )
  set +eu; conda activate "$ENV_NAME"; set -eu
fi

step "3/6  pytorch3d, matched to the torch Isaac Lab installed"
python - "$PYTORCH3D_BASE" <<'PY'
import subprocess, sys
base = sys.argv[1]
try:
    import torch
except ImportError:
    sys.exit("torch is not importable — Isaac Lab did not install. See MANUAL_SETUP.md step 2.")
full = torch.__version__                      # e.g. 2.5.1+cu118
ver, _, cuda = full.partition("+")
if not cuda:
    sys.exit(f"torch {full} has no CUDA suffix; a CPU build cannot run this. "
             f"Reinstall Isaac Lab against a CUDA build.")
want = f"{base}pt{ver}{cuda}"
print(f"  torch      = {full}")
print(f"  pytorch3d -> {want}")
try:
    import pytorch3d
    have = getattr(pytorch3d, "__version__", "")
    # the installed build encodes the torch it was compiled against
    import importlib.metadata as md
    dist = md.version("pytorch3d")
    if f"pt{ver}{cuda}" in dist:
        print(f"  already installed and matching: {dist}")
        sys.exit(0)
    print(f"  installed {dist} does NOT match torch {full} — reinstalling")
except ImportError:
    pass
cmd = [sys.executable, "-m", "pip", "install", "--extra-index-url",
       "https://miropsota.github.io/torch_packages_builder", f"pytorch3d=={want}"]
print("  " + " ".join(cmd))
r = subprocess.run(cmd)
if r.returncode != 0:
    sys.exit(f"""
pytorch3d=={want} is not available on the prebuilt index.
Browse https://miropsota.github.io/torch_packages_builder for a build matching
torch {full}, then either install it by hand or re-run with
  PYTORCH3D_BASE=<base> bash tutorial/00_setup/setup_env.sh ...
See tutorial/00_setup/MANUAL_SETUP.md step 3.""")
PY

step "4/6  git packages (--no-deps)"
# --no-deps on purpose: each declares its own torch and would break the match.
# simple-raycaster is a HARD requirement -- the point-cloud env instantiates its
# MultiMeshRaycaster at init -- so its real runtime deps are installed after it.
install_nodeps () {   # install_nodeps <import-name> <pip-spec>
  if python -c "import $1" 2>/dev/null; then
    echo "  $1: already installed"
  else
    pip install --quiet --no-deps "$2" && echo "  $1: installed"
  fi
}
install_nodeps chamfer_distance  "git+https://github.com/otaheri/chamfer_distance"
install_nodeps bps_torch         "git+https://github.com/KailinLi/bps_torch.git"
install_nodeps simple_raycaster  "git+https://github.com/Agent-3154/simple-raycaster.git"

step "5/6  Dex-X and its remaining dependencies"
pip install --quiet -e .
pip install --quiet -r requirements.txt
echo "  installed dexx (editable) + requirements.txt"

else
  echo; echo "=== 2-5/6  install steps skipped (--verify-only)"
fi

step "6/6  verify"
ok=0
python tutorial/00_setup/check_install.py  || ok=1
python tutorial/00_setup/check_portable.py || ok=1
python tutorial/00_setup/check_imports.py  || ok=1
python tutorial/01_frames_and_constants/check_frames.py || ok=1
python - <<'PY' || ok=1
import importlib, sys
missing = []
for m in ("torch", "isaaclab", "pytorch3d", "bps_torch", "chamfer_distance",
          "pytorch_kinematics", "trimesh", "gymnasium"):
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(f"{m} ({type(e).__name__})")
print("\n=== imports")
if missing:
    print("  MISSING: " + ", ".join(missing))
    sys.exit(1)
import torch
print(f"  all present; torch {torch.__version__}, cuda available={torch.cuda.is_available()}")
PY

echo
if [ "$ok" -eq 0 ]; then
  cat <<MSG
READY.

  conda activate $ENV_NAME
  bash tutorial/run_acceptance.sh      # ~6 min, trains and evaluates end to end

Only the acceptance test proves the environment is correct. Then start at
tutorial/README.md
MSG
else
  echo "Some checks failed — see the output above, and"
  echo "tutorial/00_setup/MANUAL_SETUP.md for the same steps done by hand."
  exit 1
fi
