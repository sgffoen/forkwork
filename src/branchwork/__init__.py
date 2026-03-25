"""
branchwork – Grasshopper/Rhino plugin for irregular and bifurcated branches.

All geometry processing is built on the COMPAS framework
(https://compas.dev).

Sub-modules
-----------
mesh_utils       Mesh loading, validation, and slicing helpers.
centerline       True centerline extraction from branch meshes.
branch_analysis  Bifurcation detection, thickness profiles, and Frenet frames.
skeleton         Topological skeletonisation helpers (graph-based).
"""

from .centerline import CenterlineExtractor
from .branch_analysis import BranchAnalyzer
from .skeleton import SkeletonGraph

__all__ = [
    "CenterlineExtractor",
    "BranchAnalyzer",
    "SkeletonGraph",
]
