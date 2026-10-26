# SPDX-License-Identifier: MPL-2.0
#
# Web UI for the 12-servo PCA9685 rig. Serves assets/ and exposes an API that
# calls into sketch/sketch.ino's setServoAngle/getServoAngle over the Bridge.

import json
import os

from arduino.app_utils import *
from arduino.app_bricks.web_ui import WebUI

NUM_SERVOS = 12
# Channel -> twin-joint mapping edited in the web page (assets/sim3d.js).
# /app is the app folder on the board, so this persists across restarts:
# ~/ArduinoApps/servo-web-control/twin_mapping.json
MAPPING_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "twin_mapping.json")

ui = WebUI()


def set_angle(channel: int, angle: int):
    ok = Bridge.call("setServoAngle", channel, angle)
    return {"ok": bool(ok)}


def set_all(a: str):
    # "a0,...,a11" - one animation frame for all 12 servos
    return {"ok": bool(Bridge.call("setAll", a))}


def get_state():
    return {"angles": [Bridge.call("getServoAngle", ch) for ch in range(NUM_SERVOS)]}


def get_raw():
    # Diagnostic only: raw PCA9685 register readback, to tell an I2C
    # problem apart from a wiring/power one downstream of the chip.
    return {"raw": [Bridge.call("readServoRaw", ch) for ch in range(NUM_SERVOS)]}


def get_mapping():
    try:
        with open(MAPPING_PATH) as f:
            return {"mapping": json.load(f)}
    except (FileNotFoundError, ValueError):
        return {"mapping": None}


def save_mapping(m: str):
    mapping = json.loads(m)
    valid = (isinstance(mapping, list) and len(mapping) == NUM_SERVOS
             and all({"joint", "sign", "offset"} <= set(e) for e in mapping))
    if not valid:
        return {"ok": False, "error": "expected 12 {joint, sign, offset} entries"}
    with open(MAPPING_PATH, "w") as f:
        json.dump(mapping, f, indent=1)
    return {"ok": True}


ui.expose_api("POST", "/set", set_angle)
ui.expose_api("POST", "/set_all", set_all)
ui.expose_api("GET", "/state", get_state)
ui.expose_api("GET", "/raw", get_raw)
ui.expose_api("GET", "/mapping", get_mapping)
ui.expose_api("POST", "/mapping", save_mapping)

App.run()  # Blocks until the app is stopped
