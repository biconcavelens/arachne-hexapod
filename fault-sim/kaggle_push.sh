#!/usr/bin/env bash
# Upload the sim code as the private Kaggle dataset and start kaggle_run.py as a GPU kernel.
# What runs is set by MODE and STAGES in kaggle_run.py. Needs the kaggle CLI logged in
# (~/.kaggle/kaggle.json) and runs/kaggle_v5 (the plane model; restore it from the GitHub release).
#
#   fault-sim/kaggle_push.sh          # stage, upload, push
#   fault-sim/kaggle_push.sh --dry    # only build the staging folder and print where it is
#   KAGGLE=/path/to/kaggle fault-sim/kaggle_push.sh   # if kaggle is not on PATH
set -euo pipefail
K=${KAGGLE:-kaggle}
OWNER=trishanthmellimi
REPO=$(cd "$(dirname "$0")/.." && pwd)
W=$(mktemp -d)
D=$W/dataset; A=$D/arachne; KD=$W/kernel

mkdir -p "$A/fault-sim/plane_models" "$A/vorpal-mujoco" "$A/hardware" "$A/servo-web-control/assets/gaits" "$KD"
cp "$REPO"/fault-sim/*.py "$REPO"/fault-sim/*.md "$REPO"/fault-sim/requirements.txt "$A/fault-sim/"
cp "$REPO/fault-sim/runs/kaggle_v5/runs/flow_noimu/model.pt" "$A/fault-sim/plane_models/flow_noimu.pt"
cp "$REPO/vorpal-mujoco/hexapod_mjx.xml" "$REPO/vorpal-mujoco/hexapod_physics.xml" "$A/vorpal-mujoco/"
cp -r "$REPO/vorpal-mujoco/meshes" "$A/vorpal-mujoco/"
cp -r "$REPO/hardware/camera_cap" "$A/hardware/"
cp "$REPO/servo-web-control/assets/gaits/forward.json" "$A/servo-web-control/assets/gaits/"

# kaggle_run.py stops at once if the dataset's STAMP differs from its own (a stale dataset version)
STAMP=$(date +%Y%m%d-%H%M%S)
echo "$STAMP" > "$A/fault-sim/STAMP"
sed "s/^STAMP = \"dev\"/STAMP = \"$STAMP\"/" "$REPO/fault-sim/kaggle_run.py" > "$KD/kaggle_run.py"
grep -q "^STAMP = \"$STAMP\"" "$KD/kaggle_run.py"

echo "{\"title\": \"arachne-hexapod\", \"id\": \"$OWNER/arachne-hexapod\", \"licenses\": [{\"name\": \"other\"}]}" > "$D/dataset-metadata.json"
cat > "$KD/kernel-metadata.json" <<EOF
{"id": "$OWNER/arachne-fault-sim", "title": "arachne-fault-sim", "code_file": "kaggle_run.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_tpu": false,
 "enable_internet": true, "machine_shape": "NvidiaTeslaT4", "dataset_sources": ["$OWNER/arachne-hexapod"],
 "competition_sources": [], "kernel_sources": [], "model_sources": []}
EOF
grep -E '^(MODE|STAGES) =' "$KD/kaggle_run.py"

if [ "${1:-}" = "--dry" ]; then
    echo "staged in $W (not uploaded)"
    exit 0
fi
"$K" datasets version -p "$D" --dir-mode zip -m "stamp $STAMP"
for _ in $(seq 1 30); do
    "$K" datasets status "$OWNER/arachne-hexapod" 2>/dev/null | grep -qx ready && break
    sleep 10
done
sleep 60      # pushing right after the upload sometimes attaches the previous dataset version
"$K" kernels push -p "$KD"
echo "stamp $STAMP pushed. Check with: $K kernels status $OWNER/arachne-fault-sim"
