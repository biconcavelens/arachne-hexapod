"""
Shared definitions for training (JAX, env.py) and playback (NumPy, play_policy.py)
so the two can't drift apart. Everything here works with xp = numpy or jax.numpy.

Joint/actuator order = the model's actuator order: hip0, knee0, hip1, knee1, ...

What the policy may see is what the REAL robot will have, nothing else:
IMU gravity direction + gyro (with an IMU), the command, its own last action
(stands in for joint angles: hobby servos can't report them), a 1 Hz clock.
No joint angles, no base velocity, no foot contacts (those go to the critic only).
"""
import numpy as np

CTRL_DT = 0.05                 # 20 Hz: what the web page -> Bridge path manages today (to be measured)
SIM_DT = 0.002
N_SUBSTEPS = int(round(CTRL_DT / SIM_DT))
CLOCK_HZ = 1.0

# Action a in [-1, 1] -> joint target = centre + half * a.  Limits from
# check_limits.py: hips +-25 deg keep >= 17 mm between neighbouring legs for ANY
# combination; knees -55..+10 deg.  The stand pose is hip 0 / knee -30.
HIP_HALF, KNEE_LO, KNEE_HI = np.radians(25.0), np.radians(-55.0), np.radians(10.0)
ACT_CENTER = np.tile([0.0, (KNEE_LO + KNEE_HI) / 2], 6)
ACT_HALF = np.tile([HIP_HALF, (KNEE_HI - KNEE_LO) / 2], 6)
STAND_QPOS = np.tile([0.0, np.radians(-30.0)], 6)
STAND_ACTION = (STAND_QPOS - ACT_CENTER) / ACT_HALF

# command = [forward m/s, left m/s, yaw rad/s] in the body frame
CMD_MAX = np.array([0.10, 0.05, 0.8])          # sampled in [-CMD_MAX, CMD_MAX]

OBS_SIZE = 3 + 3 + 3 + 12 + 2


def action_to_target(a, xp=np):
    return ACT_CENTER + ACT_HALF * xp.clip(a, -1.0, 1.0)


def build_obs(gravity_body, gyro, command, last_action, step, xp=np):
    """gravity_body: unit gravity direction in the body frame (3); gyro rad/s (3);
    command (3); last_action (12) in [-1, 1]; step: control steps since reset."""
    ph = 2 * np.pi * CLOCK_HZ * CTRL_DT * step
    return xp.concatenate([gravity_body, gyro, command, last_action, xp.stack([xp.sin(ph), xp.cos(ph)])])
