"""
Kaggle runner (kernel trishanthmellimi/arachne-fault-sim, dataset trishanthmellimi/arachne-hexapod).

MODE "smoke": a few episodes through every step, to find bugs cheaply.
MODE "full":  2000 gt episodes, training, closed-loop tests, and an EGL / camera-flow check.
Results go to /kaggle/working/results (status.txt says what ran and how long it took).
"""
import glob
import os
import shutil
import subprocess
import tarfile
import time
import zipfile

MODE = "stages"
STAGES = (6,)
W = "/kaggle/working"
REPO, OUT = os.path.join(W, "arachne"), os.path.join(W, "results")
FS = os.path.join(REPO, "fault-sim")
STATUS = []
N = {"smoke": dict(eps=12, epochs=2, cl=2, healthy=1, flow=2),
     "full": dict(eps=2000, epochs=30, cl=60, healthy=30, flow=24),
     "gt": dict(eps=2000, epochs=30, cl=60, healthy=30, flow=0),            # full, without the camera pipeline
     "noimu": dict(eps=2000, epochs=30, cl=60, healthy=30, flow=600, flow_cl=40, flow_healthy=20),
     # environment stages (environment.py), optical flow only, same seeds and sizes as the "noimu" flow runs
     "stages_smoke": dict(epochs=2, cl=1, healthy=1, flow=6, flow_cl=1, flow_healthy=1, video_s=6),
     "stages": dict(epochs=30, cl=40, healthy=20, flow=600, flow_cl=40, flow_healthy=20, video_s=20)}[MODE]


STAMP = "dev"   # set to the dataset's fault-sim/STAMP when pushing, so a stale dataset version is caught


def find_repo():
    def locate(root):
        hits = []
        for h in glob.glob(os.path.join(root, "**", "fault-sim", "config.py"), recursive=True):
            p = os.path.join(os.path.dirname(h), "STAMP")
            stamp = open(p).read().strip() if os.path.exists(p) else "none"
            print(f"code found at {h}, stamp {stamp} (want {STAMP})", flush=True)
            if STAMP in ("dev", stamp):
                hits.append(h)
        return os.path.dirname(os.path.dirname(hits[0])) if hits else None
    src = locate("/kaggle/input")
    if src is None:                                   # dataset kept as an archive
        tmp = os.path.join(W, "unpacked")
        for a in glob.glob("/kaggle/input/**/*", recursive=True):
            if a.endswith(".zip"):
                zipfile.ZipFile(a).extractall(tmp)
            elif a.endswith((".tar", ".tar.gz", ".tgz")):
                tarfile.open(a).extractall(tmp)
        src = locate(tmp)
    if src is None:
        STATUS.append(f"stopped: no dataset copy with stamp {STAMP} (Kaggle attached an older version; push again)")
        finish()
        raise SystemExit
    shutil.copytree(src, REPO, dirs_exist_ok=True)


def step(name, cmd, env=None):
    t0 = time.time()
    print(f"\n===== {name}: {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=FS, env={**os.environ, **(env or {})})
    STATUS.append(f"{name}: {'ok' if r.returncode == 0 else 'FAILED'} in {time.time() - t0:.0f} s")
    print(STATUS[-1], flush=True)
    return r.returncode == 0


def run(seq):
    """Steps in order (a callable is run in-process); stops at the first failed step."""
    for s in seq:
        if callable(s):
            s()
        elif not step(*s):
            STATUS.append("stopped: later steps need this one")
            return False
    return True


def flow_quality(data="data/flow", label=""):
    """How well the camera's ego-motion estimate tracks the true body twist (after warm-up)."""
    import numpy as np
    eps = [np.load(p) for p in sorted(glob.glob(os.path.join(FS, data, "ep_*.npz")))]
    w0 = 40
    est = np.concatenate([z["feat"][w0:, :6] for z in eps])
    tru = np.concatenate([z["twist_true"][w0:] for z in eps])
    lines = [f"flow quality {label} over {len(eps)} episodes ({len(est)} steps):"]
    for i, n in enumerate(["vx m/s", "vy m/s", "vz m/s", "wx rad/s", "wy rad/s", "wz rad/s"]):
        lines.append(f"  {n:9s} corr {np.corrcoef(est[:, i], tru[:, i])[0, 1]:+.2f}  "
                     f"rmse {np.sqrt(((est[:, i] - tru[:, i]) ** 2).mean()):.4f}  true sd {tru[:, i].std():.4f}")
    STATUS.extend(lines)
    print("\n".join(lines), flush=True)


def finish():
    os.makedirs(OUT, exist_ok=True)
    for p in ["runs", "data/gt/config.json"] + [os.path.basename(f) for f in glob.glob(os.path.join(FS, "camera_view*.png"))]:
        src = os.path.join(FS, p)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(OUT, p), dirs_exist_ok=True)
        elif os.path.exists(src):
            shutil.copy(src, os.path.join(OUT, os.path.basename(p)))
    open(os.path.join(OUT, "status.txt"), "w").write(f"mode {MODE}\n" + "\n".join(STATUS) + "\n")
    print("\n".join(STATUS))


if __name__ == "__main__":
    find_repo()
    cpus = os.cpu_count() or 2
    egl = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
    ckpt = "runs/gt_transformer/model.pt"
    ep = f"--epochs {N['epochs']} --healthy-epochs {min(N['epochs'], 20)} --baseline"
    cl = f"python closed_loop.py --episodes {N['cl']} --healthy-episodes {N['healthy']}"
    steps = [
        ("env", "python --version; nproc; nvidia-smi -L; pip install -q mujoco==3.12.0 && "
                "(python -c 'import cv2' || pip install -q opencv-python-headless) && "
                "python -c 'import mujoco, cv2, torch; print(mujoco.__version__, cv2.__version__, torch.__version__)'"),
        ("egl_vendor", "P=$(find /usr /opt /lib /lib64 -name 'libEGL_nvidia.so.0' 2>/dev/null | head -1); "   # GPU rendering
                       "echo \"nvidia EGL: ${P:-not found}\"; if [ -n \"$P\" ]; then mkdir -p /usr/share/glvnd/egl_vendor.d && "
                       "echo \"{\\\"file_format_version\\\":\\\"1.0.0\\\",\\\"ICD\\\":{\\\"library_path\\\":\\\"$P\\\"}}\" "
                       "> /usr/share/glvnd/egl_vendor.d/10_nvidia.json; cat /usr/share/glvnd/egl_vendor.d/10_nvidia.json; fi; true"),
        ("selftest", "python selftest.py"),
    ]
    if MODE.startswith("stages"):  # per stage: previews, flow data, models, transfer from the plane model, closed loop, videos
        fcl = f"python closed_loop.py --egomotion flow --episodes {N['flow_cl']} --healthy-episodes {N['flow_healthy']}"
        if run(steps):
            for st in STAGES:
                e = dict(egl, FAULTSIM_STAGE=str(st))
                d, r = f"data/flow_s{st}", f"runs/s{st}"
                run([
                    (f"s{st}_check", "python build_scene.py --check", e),
                    (f"s{st}_generate", f"python generate.py --out {d} --episodes {N['flow']} --egomotion flow --workers {cpus}", e),
                    lambda d=d, st=st: flow_quality(d, f"stage {st}"),
                    (f"s{st}_train_cam", f"python train.py --data {d} --out {r}_cam {ep} --no-imu", e),
                    (f"s{st}_plane_model", f"python train.py --data {d} --out {r}_plane_model --eval plane_models/flow_noimu.pt", e),
                    (f"s{st}_train_imu", f"python train.py --data {d} --out {r}_imu {ep}", e),
                    (f"s{st}_cl_none", f"python closed_loop.py --mode none --episodes {N['flow_cl']} --out {r}_cl_none.json", e),
                    (f"s{st}_cl_cam", f"{fcl} --mode model --ckpt {r}_cam/model.pt --out {r}_cl_cam.json", e),
                    (f"s{st}_videos", f"python videos.py --ckpt {r}_cam/model.pt --egomotion flow --seconds {N['video_s']} "
                                      f"--out {r}_videos", e),
                ])
        finish()
        raise SystemExit
    steps += [
        ("build_scene", "python build_scene.py"),
        ("generate_gt", f"python generate.py --out data/gt --episodes {N['eps']} --egomotion gt --workers {cpus}"),
    ]
    if MODE == "noimu":          # with IMU / IMU dropped at test / camera-only, on gt and on optical-flow features
        fcl = f"python closed_loop.py --egomotion flow --episodes {N['flow_cl']} --healthy-episodes {N['flow_healthy']}"
        steps += [
            ("train_imu", f"python train.py --data data/gt --out runs/gt_imu {ep}"),
            ("train_noimu", f"python train.py --data data/gt --out runs/gt_noimu {ep} --no-imu"),
            ("cl_none", f"{cl} --mode none --out runs/cl_none.json"),
            ("cl_oracle", f"{cl} --mode oracle --out runs/cl_oracle.json"),
            ("cl_imu", f"{cl} --mode model --ckpt runs/gt_imu/model.pt --out runs/cl_imu.json"),
            ("cl_imu_dropped", f"{cl} --mode model --ckpt runs/gt_imu/model.pt --drop-imu --out runs/cl_imu_dropped.json"),
            ("cl_noimu", f"{cl} --mode model --ckpt runs/gt_noimu/model.pt --out runs/cl_noimu.json"),
            ("generate_flow", f"python generate.py --out data/flow --episodes {N['flow']} --egomotion flow --workers {cpus}", egl),
            ("flow_train_imu", f"python train.py --data data/flow --out runs/flow_imu {ep}"),
            ("flow_train_noimu", f"python train.py --data data/flow --out runs/flow_noimu {ep} --no-imu"),
            ("flow_cl_imu", f"{fcl} --mode model --ckpt runs/flow_imu/model.pt --out runs/flow_cl_imu.json", egl),
            ("flow_cl_imu_dropped", f"{fcl} --mode model --ckpt runs/flow_imu/model.pt --drop-imu "
                                    f"--out runs/flow_cl_imu_dropped.json", egl),
            ("flow_cl_noimu", f"{fcl} --mode model --ckpt runs/flow_noimu/model.pt --out runs/flow_cl_noimu.json", egl),
        ]
    else:
        steps += [
            ("train", f"python train.py --data data/gt --out runs/gt_transformer {ep}"),
            ("closed_loop_oracle", f"{cl} --mode oracle --out runs/cl_oracle.json"),
            ("closed_loop_none", f"{cl} --mode none --out runs/cl_none.json"),
            ("closed_loop_model", f"{cl} --mode model --ckpt {ckpt} --out runs/cl_model.json"),
        ]
    for s in steps:
        if not step(*s):
            STATUS.append("stopped: later steps need this one")
            break
    else:
        if MODE == "noimu":
            flow_quality()
            finish()
            raise SystemExit
        # optional: does rendering work here, and how fast is it?
        step("videos", f"python videos.py --ckpt {ckpt} --out runs/videos", env=egl)
        if N["flow"] and step("egl_check", "ls /usr/lib/x86_64-linux-gnu | grep -i 'egl_nvidia\\|egl_mesa'; python build_scene.py --check",
                env=egl):
            if step("generate_flow", f"python generate.py --out data/flow --episodes {N['flow']} --egomotion flow "
                                     f"--workers {cpus}", env=egl):
                flow_quality()
    finish()
