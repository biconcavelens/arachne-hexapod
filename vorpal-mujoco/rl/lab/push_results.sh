#!/usr/bin/env bash
# Commit the lab results so far to the branch lab-results and push it:  bash lab/push_results.sh "message"
# Results only: run configs, logs, eval/diagnosis results, trained weights (v3_* runs) and MP4 videos.
# Large regenerable files (GIFs, evaluation trajectories) stay on the lab machine.
set -e
cd "$(dirname "$0")/.."
B=lab-results
cur=$(git rev-parse --abbrev-ref HEAD)
[ "$cur" = "$B" ] || git checkout -q -B "$B"
git add -f lab/progress.txt lab/logs 2>/dev/null || true
git add -f runs/v3_*/config.json runs/v3_*/log.txt runs/v3_*/DONE runs/v3_*/params.pkl \
  runs/v3_*/results.json runs/v3_*/table.md runs/v3_*/diag_params.pkl 2>/dev/null || true
git add -f media/lab/*.mp4 2>/dev/null || true
git diff --cached --quiet && { echo "nothing new"; exit 0; }
git commit -q -m "Lab results: ${1:-update}"
git push -q -u origin "$B"
echo "pushed: ${1:-update}"
