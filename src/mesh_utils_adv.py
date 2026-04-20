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
from compas.geometry import (
    Frame,
    Line,
    Plane,
    Point,
    Polyline,
    Vector,
    bounding_box,
    centroid_points,
)
from compas.geometry import intersection_segment_plane


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def mesh_from_vertices_faces(
    vertices: List[List[float]],
    faces: List[List[int]],
) -> Mesh:
    """Create a :class:`compas.datastructures.Mesh` from raw data.

    Parameters
    ----------
    vertices : list of [x, y, z]
    faces    : list of vertex-index lists (triangles or quads)

    Returns
    -------
    Mesh
    """
    mesh = Mesh()
    for v in vertices:
        mesh.add_vertex(x=v[0], y=v[1], z=v[2])
    for f in faces:
        mesh.add_face(f)
    return mesh


def validate_mesh(mesh: Mesh) -> Tuple[bool, List[str]]:
    """Check basic mesh validity.

    Returns
    -------
    (is_valid, list_of_issues)
    """
    issues: List[str] = []
    if mesh.number_of_vertices() < 4:
        issues.append("Mesh has fewer than 4 vertices.")
    if mesh.number_of_faces() < 1:
        issues.append("Mesh has no faces.")
    if not mesh.is_manifold():
        issues.append("Mesh is not manifold.")
    return (len(issues) == 0, issues)


def mesh_bounding_box(mesh: Mesh) -> Tuple[Point, Point]:
    """Return (min_point, max_point) of the mesh AABB."""
    pts = [mesh.vertex_coordinates(v) for v in mesh.vertices()]
    bbox = bounding_box(pts)
    lo = Point(*bbox[0])
    hi = Point(*bbox[6])
    return lo, hi


def mesh_centroid(mesh: Mesh) -> Point:
    """Return the centroid of all mesh vertices."""
    pts = [mesh.vertex_coordinates(v) for v in mesh.vertices()]
    return Point(*centroid_points(pts))


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


def slice_mesh_with_plane(mesh: Mesh, plane: Plane) -> List[Polyline]:
    """Intersect *mesh* with *plane* and return contour polylines.

    Each contiguous loop of intersection segments is returned as a
    :class:`~compas.geometry.Polyline`.  Open chains (boundary cuts) are
    also returned.

    Parameters
    ----------
    mesh  : Mesh
    plane : Plane – origin + normal define the cutting plane

    Returns
    -------
    list of Polyline
        May be empty when the plane misses the mesh entirely.
    """
    segments: List[Tuple[Point, Point]] = []

    for fkey in mesh.faces():
        verts = mesh.face_vertices(fkey)
        n = len(verts)
        crossings: List[Point] = []
        for i in range(n):
            a = Point(*mesh.vertex_coordinates(verts[i]))
            b = Point(*mesh.vertex_coordinates(verts[(i + 1) % n]))
            seg = [a, b]
            pt = intersection_segment_plane(seg, plane)
            if pt is not None:
                crossings.append(Point(*pt))
        if len(crossings) == 2:
            segments.append((crossings[0], crossings[1]))

    if not segments:
        return []

    return _chain_segments(segments)


def contour_centroid(contour: Polyline) -> Optional[Point]:
    """Return the centroid of a closed contour polyline."""
    pts = list(contour.points)
    if not pts:
        return None
    # remove duplicate closing point if present
    if len(pts) > 1 and pts[0].x == pts[-1].x and pts[0].y == pts[-1].y and pts[0].z == pts[-1].z:
        pts = pts[:-1]
    return Point(*centroid_points([[p.x, p.y, p.z] for p in pts]))


def contour_radius(contour: Polyline, centre: Point) -> float:
    """Approximate the radius of a contour as the mean distance to its centroid."""
    pts = list(contour.points)
    if not pts:
        return 0.0
    cx, cy, cz = centre.x, centre.y, centre.z
    distances = [
        math.sqrt((p.x - cx) ** 2 + (p.y - cy) ** 2 + (p.z - cz) ** 2)
        for p in pts
    ]
    return float(np.mean(distances))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _chain_segments(
    segments: List[Tuple[Point, Point]],
    tol: float = 1e-6,
) -> List[Polyline]:
    """Chain unordered line segments into polylines."""
    # Convert to numpy for fast distance checks
    segs = [(np.array([a.x, a.y, a.z]), np.array([b.x, b.y, b.z])) for a, b in segments]
    used = [False] * len(segs)
    chains: List[List[np.ndarray]] = []

    for start_idx in range(len(segs)):
        if used[start_idx]:
            continue
        chain = [segs[start_idx][0], segs[start_idx][1]]
        used[start_idx] = True
        extended = True
        while extended:
            extended = False
            tail = chain[-1]
            for i, (a, b) in enumerate(segs):
                if used[i]:
                    continue
                if np.linalg.norm(tail - a) < tol:
                    chain.append(b)
                    used[i] = True
                    extended = True
                    break
                if np.linalg.norm(tail - b) < tol:
                    chain.append(a)
                    used[i] = True
                    extended = True
                    break
        chains.append(chain)

    return [Polyline([Point(*pt) for pt in ch]) for ch in chains if len(ch) >= 2]