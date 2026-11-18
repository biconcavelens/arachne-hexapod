---
name: feedback-long-jobs-systemd
description: "Run long GPU jobs as `systemd-run --user` units with MemoryMax; nohup jobs die with the Claude app, and 14 GB RAM OOMs easily"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 409adc76-3808-4306-80db-c311cdc51662
  modified: 2026-11-18T08:16:10.551Z
---

Start long training/eval jobs with `systemd-run --user --unit=arachne-<name> -p MemoryMax=<n>G -p MemorySwapMax=0 --working-directory=... bash -c '...'`, not `nohup ... &`.

**Why:** on 2026-11-18 a diagnosis job (3 s stacked windows) and the fault_v2 training together filled the 14 GB RAM. The OOM killer fired, the Claude desktop app went down, and every nohup job in the app's cgroup died with it, including 16 min of training.

**How to apply:** detach jobs into their own user units, and cap the memory of RAM-heavy side jobs so they die alone. Check with `systemctl --user is-active arachne-<name>`. Training (8192 envs) uses about 2.9 GB RSS. Don't materialise big numpy copies next to it. Run ONE GPU job at a time: a second OOM on 2026-11-18 at 18:04 killed an uncapped training while a capped diagnosis job ran, because Firefox and the desktop take about 6 GB. Chain jobs in one unit (`a && b`) and cap every unit, the training too. See [[feedback-local-compute]] and [[user-machine-ubuntu]].
