"""Generate the visual/kinematic MuJoCo model for Vorpal The Hexapod.

The physical leg is a two-axis serial mechanism:

    chassis hip servo (yaw) -> U-bracket half -> U-bracket half
        -> knee servo (pitch) -> printed leg

The two identical U-bracket meshes are installed 90 degrees apart and snap
together in the middle. They are *not* a third free object and are not a
single loose collision proxy. This generator keeps the two real brackets on
their respective sides of the hip/yaw and knee/pitch coupling.

Geometry sources:
  - official Vorpal assembly instructions, steps 8--9
  - measured base hip centres (68.5 mm radius, six-fold symmetry)
  - the original print STL meshes in ``meshes/``
"""

import math


MM_TO_M = 0.001
N_LEGS = 6
HIP_RADIUS_MM = 68.5
HIP_Z_MM = -25.0

# The proximal, hip-facing end-cap of the extracted leg mesh. The leg STL is
# printed upside-down relative to the assembled robot, so the geometry is
# rotated 180 degrees about its local radial X axis below.
LEG_PIVOT_LOCAL_MM = (106.75, -0.58, -43.33)

# Assembly layout measured against the official assembly artwork and product
# photos. Local +X is radially outward, +Y is tangential, +Z is up.
#
# Each U bracket is about 34 mm end-to-end in the radial direction. Its servo
# ends sit at the respective servo centres; its interlocking nub points toward
# the other bracket. The 44 mm centre separation places the two nubs together
# instead of leaving a loose ring outside the leg.
HIP_TO_KNEE_MM = (44.0, 0.0, 0.0)
HIP_BRACKET_POS_MM = (17.0, 0.0, 0.0)
KNEE_BRACKET_POS_MM = (-17.0, 0.0, 0.0)
# The assembly guide's standing pose has the knee horn slightly past its
# midway point. This is the visual neutral pose; knee joint qpos values are
# offsets from it.
KNEE_STAND_DEG = 15.0

# The knee bracket is the same physical STL, but mounted in the vertical
# radial plane and facing inward. This quaternion is a 180-degree rotation
# around (0, 1, -1) / sqrt(2): x -> -x, y -> -z, z -> -y.
KNEE_BRACKET_QUAT = "0 0 0.70710678 -0.70710678"

# A 180-degree rotation about local X maps the printed leg's long section
# from +Z (print orientation) to -Z (the actual down-and-out stance), while
# preserving local +X as the radial-out direction.
LEG_EULER_DEG = (180.0, 0.0, 0.0)


def mm(value: float) -> float:
    return value * MM_TO_M


def fmt3(values: tuple[float, float, float]) -> str:
    return " ".join(f"{mm(value):.5f}" for value in values)


def leg_mesh_offset() -> tuple[float, float, float]:
    """Return the translation that puts the rotated mesh pivot at body origin."""
    x, y, z = LEG_PIVOT_LOCAL_MM
    # R_x(180) @ (x, y, z) == (x, -y, -z), so translation is -R @ pivot.
    return (-x, y, z)


leg_ox, leg_oy, leg_oz = leg_mesh_offset()
parts: list[str] = []
parts.extend(
    [
        '<mujoco model="vorpal_hexapod">',
        '  <compiler angle="degree" meshdir="meshes"/>',
        '  <option gravity="0 0 0"/>',
        '  <visual><global offwidth="1280" offheight="900"/></visual>',
        '  <asset>',
        '    <mesh name="base_mesh" file="base.stl" scale="0.001 0.001 0.001"/>',
        '    <mesh name="leg_mesh" file="leg.stl" scale="0.001 0.001 0.001"/>',
        '    <mesh name="hinge_mesh" file="hinge_half.stl" scale="0.001 0.001 0.001"/>',
        '  </asset>',
        '  <worldbody>',
        '    <light diffuse="0.9 0.9 0.9" pos="0 0 0.8" dir="0 0 -1"/>',
        '    <light diffuse="0.35 0.35 0.35" pos="0.3 -0.4 0.2" dir="-1 1 -0.3"/>',
        '    <geom name="floor" type="plane" pos="0 0 -0.120" size="1 1 0.1" rgba="0.13 0.15 0.18 1" contype="0" conaffinity="0"/>',
        '    <body name="base" pos="0 0 0">',
        '      <geom name="base_geom" type="mesh" mesh="base_mesh" rgba="0.75 0.75 0.78 1" contype="0" conaffinity="0"/>',
    ]
)

for i in range(N_LEGS):
    heading_deg = 30 + i * 60
    heading_rad = math.radians(heading_deg)
    hip_pos = (
        HIP_RADIUS_MM * math.cos(heading_rad),
        HIP_RADIUS_MM * math.sin(heading_rad),
        HIP_Z_MM,
    )

    parts.extend(
        [
            f'      <body name="hip{i}" pos="{fmt3(hip_pos)}" euler="0 0 {heading_deg}">',
            f'        <joint name="hip{i}_joint" type="hinge" axis="0 0 1" range="-90 90"/>',
            # Horizontal U-bracket: mounted to the chassis hip servo.
            f'        <geom name="hip{i}_bracket_geom" type="mesh" mesh="hinge_mesh" pos="{fmt3(HIP_BRACKET_POS_MM)}" rgba="0.95 0.56 0.12 1"/>',
            # The knee servo is held at the far side of the interlocking pair.
            f'        <body name="knee{i}" pos="{fmt3(HIP_TO_KNEE_MM)}" euler="0 {KNEE_STAND_DEG:.0f} 0">',
            f'          <joint name="knee{i}_joint" type="hinge" axis="0 1 0" range="-85 85"/>',
            # Vertical U-bracket: mounted to the knee servo/leg side.
            f'          <geom name="knee{i}_bracket_geom" type="mesh" mesh="hinge_mesh" pos="{fmt3(KNEE_BRACKET_POS_MM)}" quat="{KNEE_BRACKET_QUAT}" rgba="0.95 0.68 0.20 1"/>',
            f'          <geom name="leg{i}_geom" type="mesh" mesh="leg_mesh" pos="{mm(leg_ox):.5f} {mm(leg_oy):.5f} {mm(leg_oz):.5f}" euler="{LEG_EULER_DEG[0]:.0f} {LEG_EULER_DEG[1]:.0f} {LEG_EULER_DEG[2]:.0f}" rgba="0.24 0.48 0.90 1"/>',
            '        </body>',
            '      </body>',
        ]
    )

parts.extend(['    </body>', '  </worldbody>', '</mujoco>'])

out_path = r"C:\projects\arachne\vorpal-mujoco\hexapod.xml"
with open(out_path, "w", encoding="utf-8") as file:
    file.write("\n".join(parts) + "\n")

print(f"written: {out_path}")
