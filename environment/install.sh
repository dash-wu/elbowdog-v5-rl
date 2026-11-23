#!/usr/bin/env bash
# Recreate the elbowdog conda environment from the locked package list.
# Isaac Gym is NOT installed by this script. See 安装说明.md.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx elbowdog; then
  conda create -n elbowdog python=3.8.20 -y
fi
conda activate elbowdog
python -m pip install --upgrade pip==24.2
python -m pip install -r "$ROOT/environment/requirements-runtime.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu118

echo
echo "Python packages installed. Next:"
echo "  1. Download Isaac Gym Preview 4 from NVIDIA and: pip install -e <isaacgym>/python"
echo "  2. pip install -e $ROOT/third_party/legged_gym -e $ROOT/third_party/rsl_rl"
echo "  3. source $ROOT/env.sh"
