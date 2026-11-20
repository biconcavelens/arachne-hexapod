import json, re, subprocess, sys, os
from PIL import Image
import imageio_ffmpeg
FF = imageio_ffmpeg.get_ffmpeg_exe()
RL = "/home/trii/Projects/arachne-hexapod/vorpal-mujoco/rl"
S = os.path.dirname(os.path.abspath(__file__)); W = S + "/web"
M = {"v1_room": "1_room_imu_dead_stuck", "v2_dynroom": "2_dynroom_dyn2imu_dead", "v3_terrain": "3_terrain_terrimu_3mixed",
     "v4_tdyn_healthy": "4_tdyn_tdynimu_healthy", "v5_tdyn_faults": "4_tdyn_tdynboth_dead_worn",
     "v6_room_policy": "3_terrain_2dead_roomboth_s4", "v6_terr_policy": "3_terrain_2dead_terrboth_s4"}
for k, g in M.items():
    src = f"{RL}/media/stages/{g}.gif"
    if not os.path.exists(src): print("missing", g); continue
    if "frames" in sys.argv:
        im = Image.open(src); n = im.n_frames
        for name, i in (("mid", n // 2), ("last", n - 1)):
            im.seek(i); im.convert("RGB").save(f"{S}/{k}_{name}.png")
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", src, "-movflags", "+faststart", "-pix_fmt", "yuv420p",
                    "-c:v", "libx264", "-crf", "26", "-preset", "slow", f"{W}/{k}.mp4"], check=True)
    im = Image.open(src); im.seek(min(40, im.n_frames - 1)); im.convert("RGB").save(f"{W}/{k}.jpg", quality=80)
evals = json.load(open(S + "/evals.json"))
diag = []
for p in ["room_imu", "room_both", "room_cam", "terr_both", "tdyn_both"]:
    t = open(f"{RL}/runs_diag_{p}.out").read(); j = json.loads(t[t.index("{"):t.rindex("}") + 1])
    diag.append({k: v for k, v in j.items() if not isinstance(v, dict)})
html = open(f"{RL}/media/stages/report_src.html").read()
caps = json.load(open(S + "/caps.json")) if os.path.exists(S + "/caps.json") else {}
for i in range(1, 7): html = html.replace(f"__CAP{i}__", caps.get(str(i), ""))
html = html.replace("__DATA__", json.dumps({"evals": evals, "diag": diag}))
open(W + "/index.html", "w").write(html)
for f in sorted(os.listdir(W)): print(f, os.path.getsize(f"{W}/{f}") // 1024, "KB")
