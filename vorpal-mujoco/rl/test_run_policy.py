"""
Check run_policy.py against a mock of the board's web API (no hardware):
the loop runs, dry run sends nothing, a real run only sends servo angles inside
the allowed ranges, and it eases back to the stand pose at the end.

    python test_run_policy.py
"""
import csv
import http.server
import json
import os
import subprocess
import sys
import threading
import urllib.parse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sent = []                                    # every /policy_step command, as servo angles


class Board(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urllib.parse.urlparse(self.path).path
        imu = "1000,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,-1.0"          # level, still, gravity down = -z
        self.reply({"mapping": None} if p == "/mapping" else {"angles": [90] * 6 + [60] * 6} if p == "/state"
                   else {"imu": imu})

    def do_POST(self):
        q = urllib.parse.urlparse(self.path)
        if q.path == "/policy_step":
            sent.append([int(v) for v in urllib.parse.parse_qs(q.query)["a"][0].split(",")])
        self.reply({"ok": True, "imu": "1000,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,-1.0"})


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 7000), Board)
threading.Thread(target=srv.serve_forever, daemon=True).start()


def run(*extra):
    try:
        r = subprocess.run([sys.executable, "-u", "run_policy.py", "name=imu", "cmd=0.06,0,0", "seconds=3", "hz=20", *extra],
                           cwd=HERE, capture_output=True, text=True, timeout=40)
    except subprocess.TimeoutExpired as e:
        sys.exit(f"run_policy hung.\nstdout: {e.stdout}\nstderr: {e.stderr}")
    assert r.returncode == 0, r.stderr[-800:]
    return r.stdout


out = run("tag=mock_dry")
assert not sent, "dry run must not send servo commands"
assert "DRY RUN" in out
print("dry run ok:", out.strip().splitlines()[-1])

out = run("send=1", "tag=mock_send")
a = np.array(sent)
assert a.shape[1] == 12 and len(a) > 60, a.shape                  # ramp in + 60 ticks + ramp out
assert a.min() >= 5 and a.max() <= 175
# hips are channels 0,3,4,5,2,1 (servo = 90 - sim), knees 6,9,10,11,8,7 (servo = 90 + sim);
# allowed: hips 90 +- 25, knees 90 + [-55, +10] (1 deg slack for rounding)
for ch in (0, 3, 4, 5, 2, 1):
    assert 64 <= a[:, ch].min() and a[:, ch].max() <= 116, (ch, a[:, ch].min(), a[:, ch].max())
for ch in (6, 9, 10, 11, 8, 7):
    assert 34 <= a[:, ch].min() and a[:, ch].max() <= 101, (ch, a[:, ch].min(), a[:, ch].max())
assert list(a[-1]) == [90] * 6 + [60] * 6, a[-1]                  # ended in the stand pose
with open(os.path.join(HERE, "runs", "imu", "real_mock_send.csv")) as f:
    rows = list(csv.reader(f))
assert len(rows) == 1 + 60, len(rows)
print(f"send run ok: {len(a)} commands, hips {a[:, [0,3,4,5,2,1]].min()}..{a[:, [0,3,4,5,2,1]].max()}, "
      f"knees {a[:, [6,9,10,11,8,7]].min()}..{a[:, [6,9,10,11,8,7]].max()}, ended at stand pose")
print("test_run_policy OK")
