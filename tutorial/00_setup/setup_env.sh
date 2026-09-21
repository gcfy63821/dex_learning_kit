#!/usr/bin/env bash
# Create the conda environment for Dex-X and prove it works.
#
#   bash tutorial/00_setup/setup_env.sh [--name dexx] [--isaaclab /path/to/IsaacLab]
#   bash tutorial/00_setup/setup_env.sh --name <existing> --verify-only
#
# --verify-only skips every install step and just runs the checks against an
# environment you already have. Use it to confirm a machine is ready without
# touching what is installed on it.
#
# Isaac Sim is the one dependency this cannot install for you — it ships through
# NVIDIA's own installer and its version has to match your driver. Everything
# after that is ordinary pip. Point --isaaclab at an existing IsaacLab checkout
# and the script will install it into the new environment for you.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD

ENV_NAME=dexx
ISAACLAB=${ISAACLAB_PATH:-}
VERIFY_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --name)        ENV_NAME=$2; shift 2 ;;
    --isaaclab)    ISAACLAB=$2; shift 2 ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    -h|--help)  sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1"; exit 1 ;;
  esac
done

# Versions taken from a machine where the full pipeline runs, not from a
# changelog. Deviating is usually fine; these are what is known to work.
PY_VER=3.10
TORCH="torch==2.5.1+cu118 torchvision==0.20.1+cu118"
TORCH_INDEX=https://download.pytorch.org/whl/cu118

step () { echo; echo "=== $*"; }
die  () { echo "ERROR: $*" >&2; exit 1; }

step "0/5  prerequisites"
command -v conda >/dev/null || die "conda not found. Install Miniconda first."
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — this needs an NVIDIA GPU."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/  GPU: /'
echo "  conda: $(conda --version)"
if [ -z "$ISAACLAB" ]; then
  echo "  IsaacLab: not supplied (--isaaclab). You must install it yourself; see below."
elif [ ! -x "$ISAACLAB/isaaclab.sh" ]; then
  die "no isaaclab.sh under $ISAACLAB — point --isaaclab at an IsaacLab checkout."
else
  echo "  IsaacLab: $ISAACLAB"
fi

step "1/5  conda environment '$ENV_NAME' (python $PY_VER)"
# shellcheck disable=SC1091
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "  exists — reusing it"
else
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi
# Isaac Sim ships a conda activation hook that reads $ZSH_VERSION unguarded, so
# `set -u` kills the shell on activate. Relax it just for this call.
set +u
conda activate "$ENV_NAME"
set -u
echo "  python: $(python --version)"

if [ "$VERIFY_ONLY" -eq 0 ]; then
step "2/5  pytorch (CUDA 11.8 build)"
if python -c "import torch" 2>/dev/null; then
  python -c "import torch;print('  already installed:', torch.__version__)"
else
  # shellcheck disable=SC2086
  pip install --quiet $TORCH --index-url "$TORCH_INDEX"
  python -c "import torch;print('  installed:', torch.__version__)"
fi

step "3/5  Isaac Lab"
if python -c "import isaaclab" 2>/dev/null; then
  python -c "import isaaclab;print('  already importable:', isaaclab.__version__)"
elif [ -n "$ISAACLAB" ]; then
  echo "  running $ISAACLAB/isaaclab.sh --install (this takes a while)"
  ( cd "$ISAACLAB" && ./isaaclab.sh --install )
else
  cat <<'MSG'
  SKIPPED — Isaac Lab is not importable and no --isaaclab was given.

  Install it separately, then re-run this script:
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd IsaacLab && ./isaaclab.sh --install
  Known-good here: isaaclab 0.46.3 on Isaac Sim 4.5, python 3.10, torch 2.5.1+cu118.
MSG
fi

step "4/5  Dex-X and its dependencies"
pip install --quiet -e .
pip install --quiet -r requirements.txt
echo "  installed dexx (editable) + requirements.txt"

else
  echo; echo "=== 2-4/5  install steps skipped (--verify-only)"
fi

step "5/5  verify"
ok=0
python tutorial/00_setup/check_install.py  || ok=1
python tutorial/00_setup/check_portable.py || ok=1
python tutorial/01_frames_and_constants/check_frames.py || ok=1

echo
if [ "$ok" -eq 0 ] && python -c "import isaaclab" 2>/dev/null; then
  cat <<MSG
READY.

  conda activate $ENV_NAME
  bash tutorial/run_acceptance.sh      # ~6 min, trains and evaluates end to end

Then start at tutorial/README.md
MSG
elif [ "$ok" -eq 0 ]; then
  echo "Package checks passed, but Isaac Lab is still missing — install it, then"
  echo "re-run this script. Nothing that touches the simulator will work until then."
  exit 1
else
  echo "Some checks failed — see the output above."
  exit 1
fi
