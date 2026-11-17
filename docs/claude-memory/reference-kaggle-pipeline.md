---
name: reference-kaggle-pipeline
description: "How fault-sim runs on Kaggle: private dataset + script kernel, push script, STAMP check, how to download results"
metadata:
  node_type: memory
  type: reference
  originSessionId: cca4871e-0aac-46b0-8408-35683725756e
  modified: 2026-11-15T14:06:03.737Z
---

Kaggle account trishanthmellimi. Private dataset `trishanthmellimi/arachne-hexapod` (the code), private script kernel `trishanthmellimi/arachne-fault-sim` (T4 GPU, internet on) running `fault-sim/kaggle_run.py`. MODE and STAGES at the top of kaggle_run.py pick the run.

- Push: `fault-sim/kaggle_push.sh` (stages the files, writes a STAMP, uploads, waits, pushes). `--dry` stages only. The kernel aborts if the dataset STAMP differs (Kaggle sometimes attaches the previous dataset version).
- The CLI key is the user's: never print, read or enter it.
- Download results in pieces: `kaggle kernels output trishanthmellimi/arachne-fault-sim -p DIR --file-pattern 'results/status.txt'`, then `results/camera_view`, `results/runs/s6_` and so on. Downloads are slow and sometimes return an HTML error page; check JSONs for `<html`.
- Save each run to `fault-sim/runs/kaggle_vN` (gitignored; backed up as a GitHub release asset). Run log in `HANDOVER.md`.
- One stage of MODE "stages" takes about 30 min.

Related: [[feedback-local-compute]], [[project-camera-fault-detection-idea]].
