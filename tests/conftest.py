"""
conftest.py – shared fixtures for branchwork tests.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import pytest

from compas.datastructures import Mesh
from branchwork.mesh_utils import mesh_from_vertices_faces


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def make_cylinder_mesh(
    radius: float = 1.0,
    height: float = 10.0,
    n_sides: int = 16,
    n_rings: int = 20,
) -> Mesh:
    """Create a closed (capped) cylinder mesh aligned with the Z-axis.

    The resulting mesh is a reasonable proxy for a straight branch.
    """
    vertices: List[List[float]] = []
    faces: List[List[int]] = []

    # Side rings
    for j in range(n_rings + 1):
        z = j * height / n_rings
        for i in range(n_sides):
            angle = 2 * math.pi * i / n_sides
            vertices.append([radius * math.cos(angle), radius * math.sin(angle), z])

    # Quads on the side
    for j in range(n_rings):
        for i in range(n_sides):
            a = j * n_sides + i
            b = j * n_sides + (i + 1) % n_sides
            c = (j + 1) * n_sides + (i + 1) % n_sides
            d = (j + 1) * n_sides + i
            faces.append([a, b, c, d])

    # Bottom cap
    bottom_centre = len(vertices)
    vertices.append([0.0, 0.0, 0.0])
    for i in range(n_sides):
        a = i
        b = (i + 1) % n_sides
        faces.append([bottom_centre, b, a])

    # Top cap
    top_centre = len(vertices)
    vertices.append([0.0, 0.0, height])
    top_ring_start = n_rings * n_sides
    for i in range(n_sides):
        a = top_ring_start + i
        b = top_ring_start + (i + 1) % n_sides
        faces.append([top_centre, a, b])

    return mesh_from_vertices_faces(vertices, faces)


def make_tapered_cylinder_mesh(
    radius_start: float = 1.5,
    radius_end: float = 0.5,
    height: float = 10.0,
    n_sides: int = 16,
    n_rings: int = 20,
) -> Mesh:
    """Create a tapered (cone-like) cylinder mesh aligned with the Z-axis."""
    vertices: List[List[float]] = []
    faces: List[List[int]] = []

    for j in range(n_rings + 1):
        t = j / n_rings
        r = radius_start + t * (radius_end - radius_start)
        z = t * height
        for i in range(n_sides):
            angle = 2 * math.pi * i / n_sides
            vertices.append([r * math.cos(angle), r * math.sin(angle), z])

    for j in range(n_rings):
        for i in range(n_sides):
            a = j * n_sides + i
            b = j * n_sides + (i + 1) % n_sides
            c = (j + 1) * n_sides + (i + 1) % n_sides
            d = (j + 1) * n_sides + i
            faces.append([a, b, c, d])

    return mesh_from_vertices_faces(vertices, faces)


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cylinder_mesh():
    return make_cylinder_mesh()


@pytest.fixture
def tapered_mesh():
    return make_tapered_cylinder_mesh()
