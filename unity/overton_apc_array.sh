#!/bin/bash
#SBATCH -A pi_adowney_olin_edu
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 04:00:00
#SBATCH -J overton_apc
#SBATCH -o unity/logs/overton_apc-%A_%a.out

# One period / period-cohort / age fit per array task; each line of the job
# list is the arguments to `overton_apc.py fit`. Submit from the remote root:
#   sbatch --array=0-$((N-1)) unity/overton_apc_array.sh unity/overton_apc_jobs.txt

set -euo pipefail
ROOT=/work/pi_adowney_olin_edu/ProbablyOverthinkingIt
WORK=/work/pi_adowney_olin_edu
export TMPDIR="${WORK}/tmp"
mkdir -p "$TMPDIR"
source "${WORK}/miniforge3/etc/profile.d/conda.sh"
conda activate CultureWar
cd "$ROOT"

mapfile -t jobs < <(grep -vE '^\s*(#|$)' "$1")
ARGS="${jobs[$SLURM_ARRAY_TASK_ID]}"

echo "HOST=$(hostname) DATE=$(date -Is) JOB=${SLURM_ARRAY_JOB_ID:-na}_${SLURM_ARRAY_TASK_ID:-na}"
echo "ARGS=$ARGS"
cat unity/UNITY_SYNC_STAMP || true

# shellcheck disable=SC2086
python notebooks/overton_apc.py fit $ARGS
echo FIT_OK
