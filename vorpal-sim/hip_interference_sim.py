"""
Hip-to-hip interference simulator for the Vorpal hexapod, grounded in real
measurements taken from the actual STL files (not assumed numbers):

  - Hip pivot layout: measured directly off Hexapod_V1r8d_Base_minibrims.stl
    via radial-symmetry detection -> 6 pivots, radius ~68.5mm from center,
    60 deg apart, at angular offsets 30/90/150/210/270/330 deg.

  - Leg envelope: measured off a single leg part (Hexapod_V1r8a_Single_Leg_6
    .stl) -> reach 46.4mm from the hip-pivot end to the foot tip, max width
    ~30mm near the hip-bracket/knee-servo end, tapering to ~9mm at the foot.

CEILING / KNOWN LIMITATION: the leg is modeled as a full rectangle spanning
its full reach (not the real tapered shape, and not the hinge that connects
it to the hip horn - see calibration note below). This is a coarse
approximation, not the real tapered/hinged geometry. Upgrade path: pull the
public OnShape assembly (onshape.com, search "Vorpal Combat Hexapod") for
exact per-part transforms and redo with the real polygon (leg + hinge)
instead of a bounding rectangle.

CALIBRATION (2026-09-07): the original width, taken from the bare leg STL's
measured envelope (15.0mm half-width), was proven wrong by a real hardware
touch test - legs 0/1 actually made contact at servo0=112 (delta=+22),
servo1=46 (delta=-44), far short of what 15.0mm predicted. The hinge that
connects the leg to the hip horn adds real bulk right at the pivot that the
leg-only STL doesn't capture. Half-width was refit to 31.58mm (rounded up to
32.0mm for safety margin) to exactly reproduce that one touch point - see
calibrate_from_real_touch.py. This is a ONE-POINT fit with ONE free
parameter under an assumed-uniform-rectangle shape: it isn't verified to
generalize to other angle combinations, especially asymmetric ones, or to
other leg pairs. Treat current results as provisional until 2-3 more real
touch points (ideally a roughly-symmetric one, and a different pair) confirm
or correct it.
"""
import json
import math

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import box

HIP_RADIUS_MM = 68.5
LEG_REACH_MM = 46.4
LEG_HALF_WIDTH_MM = 32.0  # calibrated from real touch point, see docstring above
N_LEGS = 6
NEIGHBOR_ANGULAR_STEP_DEG = 60.0

# angle sweep to test, as deviation from neutral (logical 90 deg = 0 delta)
DELTA_RANGE = np.arange(-90, 90.5, 2.0)


def leg_polygon_at(pivot_xy, pointing_deg, delta_deg):
    """Leg rectangle: from the pivot, extending LEG_REACH_MM in the
    `pointing_deg + delta_deg` direction, width LEG_HALF_WIDTH_MM*2."""
    rect = box(0, -LEG_HALF_WIDTH_MM, LEG_REACH_MM, LEG_HALF_WIDTH_MM)
    rect = rotate(rect, pointing_deg + delta_deg, origin=(0, 0), use_radians=False)
    rect = translate(rect, xoff=pivot_xy[0], yoff=pivot_xy[1])
    return rect


def hip_pivot(i):
    ang = math.radians(30 + i * NEIGHBOR_ANGULAR_STEP_DEG)
    return (HIP_RADIUS_MM * math.cos(ang), HIP_RADIUS_MM * math.sin(ang)), math.degrees(ang)


results = {}
for i in range(N_LEGS):
    j = (i + 1) % N_LEGS
    pivot_i, point_i = hip_pivot(i)
    pivot_j, point_j = hip_pivot(j)

    # Full 2D grid: for each (delta_i, delta_j), does the pair collide?
    safe_grid = np.ones((len(DELTA_RANGE), len(DELTA_RANGE)), dtype=bool)
    for a_idx, da in enumerate(DELTA_RANGE):
        poly_i = leg_polygon_at(pivot_i, point_i, da)
        for b_idx, db in enumerate(DELTA_RANGE):
            poly_j = leg_polygon_at(pivot_j, point_j, db)
            if poly_i.intersects(poly_j):
                safe_grid[a_idx, b_idx] = False

    # Headline number: with neighbor j held at neutral (delta_j=0), how far
    # can leg i swing toward j (positive delta, i.e. toward increasing
    # angle/neighbor j is at +60deg from i) before collision?
    j_neutral_idx = np.argmin(np.abs(DELTA_RANGE - 0))
    safe_for_i = DELTA_RANGE[safe_grid[:, j_neutral_idx]]
    max_toward_j = safe_for_i[safe_for_i > 0].max() if (safe_for_i > 0).any() else 0.0
    min_away = safe_for_i[safe_for_i < 0].min() if (safe_for_i < 0).any() else 0.0

    # Symmetric case: both swing toward each other equally - worst case for
    # a "both legs move together" gait.
    equal_swing_safe = []
    for da in DELTA_RANGE:
        a_idx = np.argmin(np.abs(DELTA_RANGE - da))
        b_idx = np.argmin(np.abs(DELTA_RANGE - (-da)))  # j swings toward i = negative db
        if safe_grid[a_idx, b_idx]:
            equal_swing_safe.append(da)
    max_equal_swing = max(equal_swing_safe) if equal_swing_safe else 0.0

    results[f"leg{i}_leg{j}"] = {
        "leg_i": i,
        "leg_j": j,
        "max_delta_i_toward_j_when_j_neutral_deg": round(float(max_toward_j), 1),
        "max_delta_i_away_from_j_when_j_neutral_deg": round(float(min_away), 1),
        "max_equal_and_opposite_swing_deg": round(float(max_equal_swing), 1),
    }
    print(f"legs {i}<->{j}: toward-neighbor max +{max_toward_j:.1f} deg "
          f"(neighbor neutral), symmetric-approach max +/-{max_equal_swing:.1f} deg each")

out_path = r"C:\Users\trish\AppData\Local\Temp\claude\C--projects-arachne\03ce3dac-f436-4314-948b-775fb56cf72b\scratchpad\hip_interference_results.json"
with open(out_path, "w") as f:
    json.dump({
        "model": {
            "hip_radius_mm": HIP_RADIUS_MM,
            "leg_reach_mm": LEG_REACH_MM,
            "leg_half_width_mm": LEG_HALF_WIDTH_MM,
            "note": "PROVISIONAL: half-width calibrated from a single real hardware "
                    "touch point (legs 0/1 @ delta +22/-44), not yet confirmed by "
                    "additional touch points. See script docstring.",
        },
        "pairs": results,
    }, f, indent=2)
print(f"\nwritten: {out_path}")
