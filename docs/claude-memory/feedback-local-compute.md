---
name: feedback-local-compute
description: "Laptop fans fixed (2026-11-17): heavy simulation, rendering and training may run locally again; Kaggle stays available for long batches"
metadata:
  type: feedback
---

From 2026-10-27 to 2026-11-15 the laptop's fans were broken, so everything heavy ran on Kaggle and only tiny checks ran locally. On 2026-11-17 the user said the fans are fixed, and that rule no longer applies.

The machine (Ubuntu 26.04 since the OS reinstall, see [[user-machine-ubuntu]]): 22 CPU threads, 14 GB RAM, RTX 4060 Laptop GPU with 8 GB. Fine for MuJoCo stepping, fault-sim data generation, training and Blender Cycles (OptiX). Too little RAM for Isaac Sim.

**Why:** the Kaggle-only rule existed only because of the fans.

**How to apply:** run sim, rendering and training locally when it is quicker to iterate; use Kaggle for long overnight batches or when the laptop is needed. Workflow for Kaggle runs in [[reference-kaggle-pipeline]].
