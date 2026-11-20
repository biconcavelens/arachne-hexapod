---
name: reference-github-repo
description: Private GitHub repo holding the Arachne project (pushed 2026-10-26) and what was deliberately left out
metadata:
  type: reference
---

Repo: https://github.com/biconcavelens/arachne-hexapod (PUBLIC since 2026-11-20 at the user's request, was private; branch main, first commit 54df975 on 2026-10-26). Local folder C:\projects\arachne (now /home/trii/Projects/arachne-hexapod after the 2026-11 OS reinstall) is a git repo with this as `origin`.

Deliberately NOT in it (see .gitignore and NOTICE.md): `.venv/`, `.claude/`, `photos/` (user in frame), `Vorpal needed/` (CC BY-SA original download), `scratch/`, `sim/` (superseded Aug 2026 ChatGPT model), the Hu et al. PDF, all `*.gif`, `params.pkl`, mock run CSVs. The literature review is a Markdown snapshot in `docs/`; the live version is the Claude Docs artifact https://claude.ai/artifact/Ax16AW58LEepcwmj5xCWkj.

**How to apply:** the repo is public: everything pushed is visible to anyone, so scan for secrets and personal data before every push and don't add the excluded paths. Still no licence for our code (all rights reserved by default); keep the Vorpal CC BY-SA attribution. Lab results arrive on branch lab-results.
