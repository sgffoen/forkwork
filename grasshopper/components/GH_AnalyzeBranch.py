"""
GH_AnalyzeBranch.py
===================
Grasshopper / GhPython component – Analyse a Branch Mesh.

Usage inside Grasshopper
------------------------
1.  Add a *GhPython Script* component to your canvas.
2.  Paste the contents of this file into the script editor.
3.  Rename the component to ``Analyze Branch``.

Inputs
------
mesh            : Mesh  – Rhino mesh of a single branch.
num_slices      : int   – Cross-section resolution (default 50).
smooth_iter     : int   – Laplacian smoothing passes (default 3).

Outputs
-------
centerline      : Curve          – Centerline PolylineCurve.
length          : float          – Arc-length of the centerline.
mean_radius     : float          – Mean local radius.
min_radius      : float          – Minimum local radius.
max_radius      : float          – Maximum local radius.
taper_ratio     : float          – min_radius / max_radius.
radii           : list of float  – Per-point local radii.
bifurcations    : list of Point  – Bifurcation locations.
bif_angles      : list of float  – Bifurcation angles (degrees).
curvatures      : list of float  – Local curvature at each centerline point.
frames          : list of Plane  – Frenet frames along the centerline.
"""

import sys
import os

_PKG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _PKG_PATH not in sys.path:
    sys.path.insert(0, _PKG_PATH)

import Rhino.Geometry as rg

from branchwork.mesh_utils import mesh_from_vertices_faces
from branchwork.branch_analysis import BranchAnalyzer

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
# Run analysis
# ---------------------------------------------------------------------------
analyzer = BranchAnalyzer(compas_mesh, num_slices=int(num_slices), smooth_iter=int(smooth_iter))
metrics = analyzer.analyze()

# ---------------------------------------------------------------------------
# Convert COMPAS outputs → Rhino geometry
# ---------------------------------------------------------------------------
# Centerline
rh_pts = [rg.Point3d(p.x, p.y, p.z) for p in metrics.centerline.points]
centerline = rg.PolylineCurve(rh_pts)

# Scalar metrics
length      = metrics.length
mean_radius = metrics.thickness.mean_radius
min_radius  = metrics.thickness.min_radius
max_radius  = metrics.thickness.max_radius
taper_ratio = metrics.thickness.taper_ratio
radii       = metrics.thickness.radii
curvatures  = metrics.curvatures

# Bifurcations
bifurcations = [rg.Point3d(b.location.x, b.location.y, b.location.z) for b in metrics.bifurcations]
bif_angles   = [b.angle_deg for b in metrics.bifurcations]

# Frenet frames
frames = []
for fr in metrics.frames:
    origin = rg.Point3d(fr.point.x, fr.point.y, fr.point.z)
    xaxis  = rg.Vector3d(fr.xaxis.x, fr.xaxis.y, fr.xaxis.z)
    yaxis  = rg.Vector3d(fr.yaxis.x, fr.yaxis.y, fr.yaxis.z)
    frames.append(rg.Plane(origin, xaxis, yaxis))
