#!/bin/bash
#SBATCH -A pi_adowney_olin_edu
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -J overton_irt
#SBATCH -o unity/logs/overton_irt-%A_%a.out

# One IRT fit per array task; each line of the job list is "set key model".
# Submit from the remote root:
#   sbatch --array=0-$((N-1)) unity/overton_irt_array.sh unity/overton_irt_jobs.txt
# Sampling length from the environment: IRT_TUNE, IRT_DRAWS (defaults below).

set -euo pipefail
ROOT=/work/pi_adowney_olin_edu/ProbablyOverthinkingIt
WORK=/work/pi_adowney_olin_edu
export TMPDIR="${WORK}/tmp"
mkdir -p "$TMPDIR"
source "${WORK}/miniforge3/etc/profile.d/conda.sh"
conda activate CultureWar
cd "$ROOT"

mapfile -t jobs < <(grep -vE '^\s*(#|$)' "$1")
read -r SET KEY MODEL <<<"${jobs[$SLURM_ARRAY_TASK_ID]}"

echo "HOST=$(hostname) DATE=$(date -Is) JOB=${SLURM_ARRAY_JOB_ID:-na}_${SLURM_ARRAY_TASK_ID:-na}"
echo "SET=$SET KEY=$KEY MODEL=$MODEL"
cat unity/UNITY_SYNC_STAMP || true
python -c "import pymc, nutpie, arviz; print('pymc', pymc.__version__, 'nutpie', nutpie.__version__, 'arviz', arviz.__version__)"

python notebooks/overton_irt.py fit --set "$SET" --key "$KEY" --model "$MODEL" \
  --tune "${IRT_TUNE:-2000}" --draws "${IRT_DRAWS:-1000}" --chains 4
echo FIT_OK
