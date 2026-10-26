import math
from shapely.affinity import rotate, translate
from shapely.geometry import box

HIP_RADIUS_MM = 68.5
LEG_REACH_MM = 46.4

def hip_pivot(i):
    ang = math.radians(30 + i * 60.0)
    return (HIP_RADIUS_MM * math.cos(ang), HIP_RADIUS_MM * math.sin(ang)), math.degrees(ang)

def leg_polygon_at(pivot_xy, pointing_deg, delta_deg, half_width):
    rect = box(0, -half_width, LEG_REACH_MM, half_width)
    rect = rotate(rect, pointing_deg + delta_deg, origin=(0, 0), use_radians=False)
    rect = translate(rect, xoff=pivot_xy[0], yoff=pivot_xy[1])
    return rect

pivot0, point0 = hip_pivot(0)
pivot1, point1 = hip_pivot(1)

# Real touch point from hardware: servo0=112 (delta=+22), servo1=46 (delta=-44)
delta0, delta1 = 112 - 90, 46 - 90

lo, hi = 0.1, 60.0  # binary search half-width (mm) for first contact
for _ in range(40):
    mid = (lo + hi) / 2
    poly0 = leg_polygon_at(pivot0, point0, delta0, mid)
    poly1 = leg_polygon_at(pivot1, point1, delta1, mid)
    if poly0.intersects(poly1):
        hi = mid
    else:
        lo = mid

print(f"delta0={delta0}, delta1={delta1}")
print(f"calibrated half-width for first contact: {hi:.2f} mm (was 15.0 mm assumed)")
print(f"i.e. hinge/bracket adds ~{hi-15.0:.2f} mm beyond the bare leg's measured width")
