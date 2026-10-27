# Running fault-sim on Kaggle

Yes, it fits. Generating `gt` data is CPU-only. Training the small Transformer is fast on a
T4/P100. Only `--egomotion flow` needs rendering, and EGL on Kaggle's GPU images is untested
here. If it fails, everything still runs in `gt` mode.

Check your own limits on the account page: there is a weekly GPU quota and a per-session time
limit. `generate.py` skips episodes it already wrote, so a cut-off session loses nothing that
was saved.

## 1. Get the code in

The repo is private, so either:

- **Dataset (simplest):** zip the repo locally and upload it as a *private* Kaggle dataset:
  `git archive -o arachne.zip HEAD` (commit `fault-sim/` and `hardware/camera_cap/` first).
- **Clone:** add a GitHub fine-grained token, read-only on this repo, as a Kaggle Secret
  yourself, then clone with it in the notebook. Don't paste the token into a cell.

The sim needs these from the repo: `fault-sim/`, `vorpal-mujoco/hexapod_mjx.xml`,
`vorpal-mujoco/hexapod_physics.xml`, `vorpal-mujoco/meshes/`,
`hardware/camera_cap/cap_with_camera_mount.stl` and `camera_pose.json`, and
`servo-web-control/assets/gaits/forward.json`.

## 2. Notebook

Settings: Accelerator GPU T4 (or None for `gt` generation only); Internet on.

```bash
cp -r /kaggle/input/<your-dataset>/arachne /kaggle/working/   # /kaggle/input is read-only
cd /kaggle/working/arachne/fault-sim
pip install -q mujoco==3.12.0 opencv-python-headless
python selftest.py
python build_scene.py                 # writes hexapod_camera.xml (no rendering)
python generate.py --out data/gt --episodes 2000 --egomotion gt --workers 4
python train.py --data data/gt --out runs/gt_transformer --baseline
python closed_loop.py --mode oracle --episodes 60 --out runs/cl_oracle.json
python closed_loop.py --mode none   --episodes 60 --out runs/cl_none.json
python closed_loop.py --mode model  --ckpt runs/gt_transformer/model.pt --episodes 60 --out runs/cl_model.json
```

Camera pipeline (optional, GPU session):

```bash
export MUJOCO_GL=egl
python build_scene.py --check          # if this fails, stay in gt mode
python generate.py --out data/flow --episodes 400 --egomotion flow --workers 2
python train.py --data data/flow --out runs/flow_transformer
python closed_loop.py --mode model --ckpt runs/flow_transformer/model.pt --egomotion flow --episodes 30 --out runs/cl_flow.json
```

## 3. Keep the results

Only `/kaggle/working` survives, and only when you **Save Version** (or download it). Download
`runs/` at minimum. `data/` can be regenerated from the seeds.
