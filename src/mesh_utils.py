"""
mesh_utils.py
=============
Mesh loading, validation, and slicing helpers built on COMPAS.

All public functions accept / return ``compas.datastructures.Mesh`` objects
unless stated otherwise.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from compas.datastructures import Mesh
from compas_cgal.skeletonization import mesh_skeleton
from compas.geometry import (
    Frame,
    Line,
    Plane,
    Point,
    Polyline,
    Vector,
    bounding_box,
    centroid_points,
    oriented_bounding_box_numpy,
    Box,
)



# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def mesh_bounding_box(mesh):
    """Compute the axis-aligned bounding box of a mesh.

    Parameters
    ----------
    mesh : :class:`compas.datastructures.Mesh`
        The input mesh.

    Returns
    -------
    list of Point
        The 8 corner points of the bounding box.
    """
    v, f = mesh.to_vertices_and_faces(triangulated=True)
    bbox = oriented_bounding_box_numpy(v)
    bbox = Box.from_bounding_box(bbox)
    return bbox


def principal_axes(mesh: Mesh) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the three principal axes of a mesh via PCA.

    Returns
    -------
    (axes, eigenvalues)
        *axes*  : (3, 3) array – each **row** is a unit axis (sorted by
                  descending variance, so row 0 is the longest axis).
        *eigenvalues* : (3,) array
    """
    pts = np.array([mesh.vertex_coordinates(v) for v in mesh.vertices()])
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # sort descending
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors.T[idx], eigenvalues[idx]


def skeletonize_mesh(mesh):
    """Skeletonizes a mesh and returns a list of polylines representing the skeleton.

    Parameters
    ----------
    mesh : compas.datastructures.Mesh
        The input mesh to be skeletonized.

    Returns
    -------
    list of compas.geometry.Polyline
        A list of polylines representing the skeleton of the mesh.
    """
    v, f = mesh.to_vertices_and_faces(triangulated=True)

    # =============================================================================
    # Skeleton
    # =============================================================================

    skeleton_edges = mesh_skeleton((v, f))

    polylines = []
    for start_point, end_point in skeleton_edges:
        polyline = Polyline([start_point, end_point])
        polylines.append(polyline)

    return polylines