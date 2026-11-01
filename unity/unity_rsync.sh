#!/usr/bin/env bash
# Push / pull the Overton IRT jobs to and from Unity.
#
# Unity is compute only: no git there. Push sends just the job inputs
# (the model scripts, their input data, and the Slurm scripts); pull
# brings back results and logs. Uses the CultureWar conda env on Unity
# (see ~/CultureWar/unity_workflow.md).
#
# Usage (from anywhere):
#   unity/unity_rsync.sh push [--dry-run]
#   unity/unity_rsync.sh pull [--dry-run]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="${UNITY_REMOTE:-unity:/work/pi_adowney_olin_edu/ProbablyOverthinkingIt}"
STAMP="${ROOT}/unity/UNITY_SYNC_STAMP"

cmd="${1:-}"
shift || true
DRY=()
for a in "$@"; do
  case "$a" in
    --dry-run|-n) DRY=(-n) ;;
  esac
done

case "$cmd" in
  push)
    {
      echo "pushed_at=$(date -Is)"
      echo "git_head=$(git -C "$ROOT" rev-parse HEAD)"
      echo "irt_script_md5=$(md5sum "$ROOT/notebooks/overton_irt.py" | cut -d' ' -f1)"
      echo "apc_script_md5=$(md5sum "$ROOT/notebooks/overton_apc.py" | cut -d' ' -f1)"
    } >"$STAMP"
    ssh unity "mkdir -p /work/pi_adowney_olin_edu/ProbablyOverthinkingIt/notebooks/overton_irt"
    ssh unity "mkdir -p /work/pi_adowney_olin_edu/ProbablyOverthinkingIt/notebooks/overton_apc"
    rsync -avP "${DRY[@]}" "$ROOT"/notebooks/overton_irt.py "$ROOT"/notebooks/overton_apc.py "$REMOTE"/notebooks/
    rsync -avP "${DRY[@]}" --delete "$ROOT"/notebooks/overton_apc/data/ "$REMOTE"/notebooks/overton_apc/data/
    rsync -avP "${DRY[@]}" --delete "$ROOT"/notebooks/overton_irt/data/ "$REMOTE"/notebooks/overton_irt/data/
    rsync -avP "${DRY[@]}" --delete --exclude logs/ "$ROOT"/unity/ "$REMOTE"/unity/
    ;;
  pull)
    rsync -avP "${DRY[@]}" "$REMOTE"/notebooks/overton_irt/results/ "$ROOT"/notebooks/overton_irt/results/ || true
    rsync -avP "${DRY[@]}" "$REMOTE"/notebooks/overton_apc/results/ "$ROOT"/notebooks/overton_apc/results/ || true
    rsync -avP "${DRY[@]}" "$REMOTE"/unity/logs/ "$ROOT"/unity/logs/
    ;;
  *)
    echo "Usage: $0 {push|pull} [--dry-run]" >&2
    exit 2
    ;;
esac
