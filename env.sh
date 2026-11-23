# source this file:  source env.sh
# Recreates the shell state used for training and MuJoCo playback.
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
conda activate elbowdog
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
ELBOWDOG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ELBOWDOG_ROOT
cd "$ELBOWDOG_ROOT"
