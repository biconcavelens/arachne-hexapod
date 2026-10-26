"""
Read the IMU through the web app (needs `adb forward tcp:7000 tcp:7000`).

    python imu_log.py status                 # is the MPU6050 answering? (who=104 is good)
    python imu_log.py live                   # print gravity / gyro; tilt the robot to check the axes
    python imu_log.py log out.csv 60         # log 200 Hz samples for 60 s -> CSV
    python imu_log.py calibrate              # gyro bias, robot STILL for ~2 s

Frame: x toward L0 (front), y left, z up. At rest on a table gravity ("down")
should read about (0, 0, -1). Check each axis by tilting the robot (hold it by the
body, servos unpowered is fine):
  front UP   -> down.x goes NEGATIVE      front DOWN -> down.x positive
  left side UP -> down.y goes NEGATIVE    right side UP -> down.y positive
If a sign or axis is wrong the chip is mounted turned; tell me what you see and
I'll set the axes map (/imu_axes).
"""
import csv
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:7000"   # not "localhost": on Windows that tries IPv6 first (~2 s per request)


def call(path, post=False):
    req = urllib.request.Request(BASE + path, method="POST" if post else "GET")
    return json.load(urllib.request.urlopen(req, timeout=15))


def status():
    who = call("/imu_status")["who"]
    print(f"WHO_AM_I = {who} ({'MPU6050 OK' if who == 104 else 'no / wrong chip: check wiring, power and address'})")


def live():
    while True:
        s = call("/imu")["imu"]
        if not s:
            print("no IMU data (imuOk is false: check `status`)")
            return
        t, gx, gy, gz, ax, ay, az, dx, dy, dz = map(float, s.split(","))
        print(f"down ({dx:+.2f} {dy:+.2f} {dz:+.2f})   gyro rad/s ({gx:+.3f} {gy:+.3f} {gz:+.3f})   accel g ({ax:+.2f} {ay:+.2f} {az:+.2f})")
        time.sleep(0.2)


def log(path, seconds):
    t_end, n, wrap, last, rows = time.time() + seconds, 0, 0, None, []
    call("/imu_batch?n=40")                              # drop whatever is already buffered
    while time.time() < t_end:
        samples = call("/imu_batch?n=40")["samples"]
        for line in filter(None, samples.split(";")):
            v = line.split(",")
            t = int(v[0])
            if last is not None and t < last:
                wrap += 1 << 32                           # micros() wrapped
            last = t
            rows.append([(t + wrap) * 1e-6, *v[1:]])
        time.sleep(0.04)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "gx", "gy", "gz", "ax", "ay", "az"])
        w.writerows(rows)
    dt = [b[0] - a[0] for a, b in zip(rows, rows[1:])]
    gaps = sum(d > 0.0075 for d in dt)
    print(f"{len(rows)} samples in {seconds} s ({len(rows)/seconds:.0f}/s), {gaps} gaps > 7.5 ms (lost samples) -> {path}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        status()
    elif cmd == "live":
        live()
    elif cmd == "log":
        log(sys.argv[2], float(sys.argv[3]))
    elif cmd == "calibrate":
        print(call("/imu_calibrate", post=True))
    else:
        print(__doc__)
