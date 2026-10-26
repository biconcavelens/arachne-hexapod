import numpy as np
import trimesh

LEG = r"C:\projects\arachne\vorpal-mujoco\meshes\leg.stl"
mesh = trimesh.load(LEG)
verts = mesh.vertices
x = verts[:, 0]
xmin = x.min()

# Tight band right at the very tip - this should be the hip-horn attachment boss
for band_mm in [0.5, 1.0, 2.0]:
    band = verts[x < xmin + band_mm]
    if len(band) < 3:
        continue
    centroid = band.mean(axis=0)
    print(f"band={band_mm}mm: n={len(band)}, centroid={centroid.round(2)}, "
          f"y_range=[{band[:,1].min():.2f},{band[:,1].max():.2f}], "
          f"z_range=[{band[:,2].min():.2f},{band[:,2].max():.2f}]")

print(f"\nxmin={xmin:.2f}")
print(f"mesh centroid: {mesh.centroid.round(2)}")
print(f"mesh bounds:\n{mesh.bounds.round(2)}")
