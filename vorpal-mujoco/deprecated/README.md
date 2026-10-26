Superseded scripts. Kept for reference only, do not run.

- generate_mjcf_CHATGPT_GUESSED_do_not_use.py: rewrote the model from
  hand-guessed dimensions (44mm hip-to-knee spacing, 15deg knee stand
  offset, hand-picked bracket quaternion) read off assembly photos,
  discarding the exact OnShape-derived ground truth in
  onshape_assembly.json. Looked more "hexapod-like" at a glance but every
  number in it is a guess. Not used.
- render_snapshot_chatgpt.py / calibrate_hinge_offset.py: earlier tooling
  from before the OnShape ground truth was pulled, superseded by
  build_from_onshape.py + render_views.py.

The authoritative pipeline is build_from_onshape.py (reads
onshape_assembly.json, exact CAD mate transforms) -> hexapod.xml, rendered
via render_views.py.
