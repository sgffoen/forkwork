"""
GH_MeshToSkeleton.py
====================
Grasshopper / GhPython component – Convert Branch Mesh to Skeleton Graph.

Usage inside Grasshopper
------------------------
1.  Add a *GhPython Script* component to your canvas.
2.  Paste the contents of this file into the script editor.
3.  Rename the component to ``Mesh To Skeleton``.

Inputs
------
mesh        : Mesh  – Rhino mesh of a single branch.
num_slices  : int   – Cross-section resolution (default 50).
smooth_iter : int   – Laplacian smoothing passes (default 3).

Outputs
-------
centerline      : Curve          – Skeleton polyline (raw centerline).
node_points     : list of Point  – All skeleton node positions.
edge_lines      : list of Line   – All skeleton edges as line segments.
edge_lengths    : list of float  – Length of each skeleton edge.
leaf_points     : list of Point  – Tip nodes (degree 1).
junction_points : list of Point  – Bifurcation nodes (degree > 2).
branch_curves   : list of Curve  – Each unbranched segment as a PolylineCurve.
total_length    : float          – Total arc-length of all skeleton edges.
"""

import sys
import os

_PKG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _PKG_PATH not in sys.path:
    sys.path.insert(0, _PKG_PATH)

import Rhino.Geometry as rg

from branchwork.mesh_utils import mesh_from_vertices_faces
from branchwork.skeleton import SkeletonGraph

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
# Build skeleton graph
# ---------------------------------------------------------------------------
sg = SkeletonGraph(compas_mesh, num_slices=int(num_slices), smooth_iter=int(smooth_iter))
graph = sg.build()

# ---------------------------------------------------------------------------
# Convert COMPAS outputs → Rhino geometry
# ---------------------------------------------------------------------------
# Centerline polyline
cl = sg.centerline
rh_cl_pts = [rg.Point3d(p.x, p.y, p.z) for p in cl.points]
centerline = rg.PolylineCurve(rh_cl_pts)

# Node points
node_points = [rg.Point3d(pt.x, pt.y, pt.z) for pt in sg.node_points.values()]

# Edge lines
edge_lines = []
for u, v in graph.edges():
    pu = sg.node_points[u]
    pv = sg.node_points[v]
    edge_lines.append(
        rg.Line(
            rg.Point3d(pu.x, pu.y, pu.z),
            rg.Point3d(pv.x, pv.y, pv.z),
        )
    )

# Edge lengths
edge_lengths = sg.edge_lengths()

# Leaf and junction nodes
leaf_keys     = sg.leaf_nodes()
junction_keys = sg.junction_nodes()
leaf_points     = [rg.Point3d(sg.node_points[k].x, sg.node_points[k].y, sg.node_points[k].z) for k in leaf_keys]
junction_points = [rg.Point3d(sg.node_points[k].x, sg.node_points[k].y, sg.node_points[k].z) for k in junction_keys]

# Branch curves
branch_curves = []
for branch_keys in sg.branches():
    pts = [sg.node_points[k] for k in branch_keys]
    rh_bpts = [rg.Point3d(p.x, p.y, p.z) for p in pts]
    if len(rh_bpts) >= 2:
        branch_curves.append(rg.PolylineCurve(rh_bpts))

# Total length
total_length = sg.total_length()
