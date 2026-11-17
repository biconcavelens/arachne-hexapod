---
name: user-machine-ubuntu
description: "Since the 2026-11 reinstall the user is on Ubuntu 26.04 (native, no Windows/WSL); repo at /home/trii/Projects/arachne-hexapod"
metadata:
  type: user
---

After the OS reinstall (around 2026-11-16) the user's laptop runs Ubuntu 26.04 LTS natively. Windows, Git Bash, PowerShell and WSL2 are gone. The repo is at `/home/trii/Projects/arachne-hexapod` (it used to be `C:\projects\arachne`).

Machine: 22 CPU threads, 14 GB RAM, RTX 4060 Laptop GPU (8 GB, NVIDIA driver working). System Python is 3.14; `gh` is installed and logged in. As of 2026-11-17, `uv`, `adb` and the `kaggle` CLI are not installed yet.

**Why:** older notes and docs (vorpal-mujoco/HANDOVER.md, portfolio-handover.md, research-questions.md) describe the Windows + WSL2 setup.

**How to apply:** ignore Windows-only advice in old notes (Git Bash `//` path prefixes, `wsl -d Ubuntu` wrappers, `/mnt/c/...` paths, the Windows IPv6 localhost issue). Run the RL scripts natively in Linux. fault-sim pins mujoco 3.12.0 and was tested on Python 3.11-3.13, so use a 3.12 or 3.13 venv rather than the system 3.14. Related: [[feedback-local-compute]].
