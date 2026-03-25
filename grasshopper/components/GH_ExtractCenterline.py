"""
GH_ExtractCenterline.py
=======================
Grasshopper / GhPython component – Extract Centerline from Branch Mesh.

Usage inside Grasshopper
------------------------
1.  Add a *GhPython Script* component to your canvas.
2.  Paste the contents of this file into the script editor.
3.  Rename the component to ``Extract Centerline``.

Inputs
------
mesh        : Mesh    – Rhino mesh of a single branch.
num_slices  : int     – Number of cross-sectional slices (default 50).
smooth_iter : int     – Laplacian smoothing passes (default 3).

Outputs
-------
centerline  : Curve   – Centerline as a PolylineCurve.
radii       : list    – Local radius at each centerline point.
frames      : list    – Frenet frames (Rhino Plane) at each point.
length      : float   – Arc-length of the centerline.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Allow the branchwork package to be imported inside Grasshopper/Rhino.
# Adjust the path below to match where you installed the package.
# ---------------------------------------------------------------------------
_PKG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _PKG_PATH not in sys.path:
    sys.path.insert(0, _PKG_PATH)

import rhinoscriptsyntax as rs
import Rhino.Geometry as rg

from branchwork.mesh_utils import mesh_from_vertices_faces
from branchwork.centerline import CenterlineExtractor

# ---------------------------------------------------------------------------
# Component inputs (Grasshopper injects these names automatically)
# ---------------------------------------------------------------------------
# mesh        – Rhino.Geometry.Mesh
# num_slices  – int  (optional, default 50)
# smooth_iter – int  (optional, default 3)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
if "num_slices" not in dir() or num_slices is None:  # noqa: F821
    num_slices = 50
if "smooth_iter" not in dir() or smooth_iter is None:  # noqa: F821
    smooth_iter = 3

# ---------------------------------------------------------------------------
# Convert Rhino mesh → COMPAS mesh
# ---------------------------------------------------------------------------
vertices = [[v.X, v.Y, v.Z] for v in mesh.Vertices]  # noqa: F821
faces = []
for fi in range(mesh.Faces.Count):  # noqa: F821
    f = mesh.Faces[fi]  # noqa: F821
    if f.IsTriangle:
        faces.append([f.A, f.B, f.C])
    else:
        faces.append([f.A, f.B, f.C, f.D])

compas_mesh = mesh_from_vertices_faces(vertices, faces)

# ---------------------------------------------------------------------------
# Extract centerline
# ---------------------------------------------------------------------------
extractor = CenterlineExtractor(compas_mesh, num_slices=int(num_slices), smooth_iterations=int(smooth_iter))
cl = extractor.compute()

# ---------------------------------------------------------------------------
# Convert COMPAS outputs → Rhino geometry
# ---------------------------------------------------------------------------
# Centerline polyline
rh_pts = [rg.Point3d(p.x, p.y, p.z) for p in cl.points]
centerline = rg.PolylineCurve(rh_pts)

# Radii
radii = extractor.radii

# Frames as Rhino Planes
frames = []
for fr in extractor.frames:
    origin = rg.Point3d(fr.point.x, fr.point.y, fr.point.z)
    xaxis  = rg.Vector3d(fr.xaxis.x, fr.xaxis.y, fr.xaxis.z)
    yaxis  = rg.Vector3d(fr.yaxis.x, fr.yaxis.y, fr.yaxis.z)
    frames.append(rg.Plane(origin, xaxis, yaxis))

# Length
length = extractor.length
