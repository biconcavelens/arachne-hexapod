---
name: feedback-no-local-compute
description: "The laptop's fans are broken: run all simulation, training and rendering on Kaggle, only tiny checks locally"
metadata:
  node_type: memory
  type: feedback
  originSessionId: cca4871e-0aac-46b0-8408-35683725756e
  modified: 2026-11-15T14:06:03.650Z
---

The user's laptop fans are broken ("we can't run anything locally", said 2026-10-27 and still true 2026-11-15). Everything heavy (MuJoCo stepping, rendering, data generation, training, videos) runs on Kaggle. Locally only: compiling XML, randomising a model without stepping, a few forward-kinematics calls, syntax checks, JSON analysis, report/PDF generation.

**Why:** heavy local runs overheat the machine.

**How to apply:** for any sim change, verify the cheap parts locally, then a Kaggle smoke run, then the full run. Workflow in [[reference-kaggle-pipeline]].
