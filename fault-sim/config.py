"""
Every tunable number of the fault simulator, in one place.
Units: joint angles and fault sizes in degrees, time in seconds, lengths in metres.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MJX_XML = os.path.join(ROOT, "vorpal-mujoco", "hexapod_mjx.xml")        # lean physics model (foot spheres)
PHYS_XML = os.path.join(ROOT, "vorpal-mujoco", "hexapod_physics.xml")   # source of the visual meshes
MESH_DIR = os.path.join(ROOT, "vorpal-mujoco", "meshes")
CAP_STL = os.path.join(ROOT, "hardware", "camera_cap", "cap_with_camera_mount.stl")
CAMERA_POSE = os.path.join(ROOT, "hardware", "camera_cap", "camera_pose.json")
GAIT_JSON = os.path.join(ROOT, "servo-web-control", "assets", "gaits", "forward.json")
SCENE_XML = os.path.join(HERE, "hexapod_camera.xml")                    # written by build_scene.py

# ---- timing
CTRL_DT = 0.05            # 20 Hz commands, as on the real robot (web page -> Bridge)
SIM_DT = 0.002
N_SUBSTEPS = int(round(CTRL_DT / SIM_DT))
CYCLE_STEPS = 24          # one tripod gait cycle (1.2 s) at 20 Hz

# ---- camera (the mount and lens pose come from hardware/camera_cap/camera_pose.json)
IMG_W, IMG_H = 160, 120
FOVY_DEG = 45.0           # vertical field of view; set it from your camera's datasheet
CAP_Z_OFFSET = 0.0        # move cap and camera up/down if the real lens height differs
CAM_POS_JITTER = 0.002    # per-episode mounting error, metres
CAM_ROT_JITTER_DEG = 2.0  # per-episode mounting error

# ---- faults, per faulty servo
FAULT_TYPES = ("none", "loss", "sag", "deadband")
N_FAULTY = (1, 2)              # a faulty episode has one or two faulty servos
HEALTHY_FRACTION = 0.25        # share of episodes with no fault at all
LOSS_P_DROP = (0.5, 1.0)       # share of commands lost; 1.0 = the servo is frozen
SAG_SPAN_DEG = (2.0, 5.0)      # a sagging servo moves only this far from where the sag began
DEADBAND_DEG = (15.0, 20.0)    # commanded changes smaller than this are ignored
MIN_SERVO_TRAVEL_DEG = 2.0     # servos the gait barely moves (hip0, hip3) are not given faults

# ---- what the simulator assumes each recovery action achieves
RESEND_COPIES = 3        # after a diagnosed loss, every command is sent this many times
LINK_RESET_P = 0.7       # chance one channel reset clears a frozen (p_drop = 1) servo
SAG_CLEAR_P = 0.5        # chance one pause-and-resend clears a sag
PAUSE_S = 1.0            # standing pause before resending after a sag
MAX_ATTEMPTS = 3         # per fault, before giving up and walking around it
PROBE_RAMP_DEG = 24.0    # dead-band probe: slow ramp of one servo while standing
PROBE_S = 3.0
DEADBAND_MARGIN_DEG = 3.0

# ---- episodes
EPISODE_S = 14.0
ONSET_S = (4.0, 7.0)     # faults start at a random time in this window
WARMUP_S = 2.0           # first seconds (standing up, ramping into the gait) are not used for training

# ---- gait variations (the clearance-checked tripod gait, scaled; hips stay within the +-25 deg checked range)
STEER = (-0.3, 0.3)
SPEED = (0.8, 1.2)
HIP_AMP_SCALE = (0.8, 1.1)
KNEE_LIFT_SCALE = (0.8, 1.1)

# ---- sensor noise (as in fault-pilot; "x1" is about what our IMU showed while handled)
GYRO_NOISE = 0.05        # rad/s
ACC_NOISE_G = 0.05
GRAV_NOISE = 0.03
GT_LIN_NOISE = 0.005     # m/s, for the no-rendering (ground-truth ego-motion) mode
GT_ANG_NOISE = 0.05      # rad/s
GT_SCALE_SD = 0.1        # per-episode monocular scale error

# ---- diagnosis windows and features
WINDOW = CYCLE_STEPS     # one gait cycle per window
STRIDE = 6
GRID = 4                 # flow summary: GRID x GRID cells, mean (u, v) each
FEAT_EGO = 6 + 2 * GRID * GRID     # ego twist + flow grid = 38
FEAT_IMU = 9                       # gyro, gravity direction, specific force
FEAT_CMD = 12
FEAT_PHASE = 2


def camera_pose():
    return json.load(open(CAMERA_POSE))


TYPE_ID = {name: i for i, name in enumerate(FAULT_TYPES)}
# actuator order in the model: hip0, knee0, hip1, knee1, ...  (gait files use hip0..5, knee0..5)
JOINT_NAMES = [f"{j}{k}" for k in range(6) for j in ("hip", "knee")]
GAIT_TO_ACT = [k + 6 * j for k in range(6) for j in (0, 1)]
