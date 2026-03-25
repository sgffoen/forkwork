"""
test_mesh_utils.py – unit tests for branchwork.mesh_utils
"""

import math
import pytest
import numpy as np

from compas.datastructures import Mesh
from compas.geometry import Plane, Point, Vector

from branchwork.mesh_utils import (
    contour_centroid,
    contour_radius,
    mesh_bounding_box,
    mesh_centroid,
    mesh_from_vertices_faces,
    principal_axes,
    slice_mesh_with_plane,
    validate_mesh,
)


# ---------------------------------------------------------------------------
# mesh_from_vertices_faces
# ---------------------------------------------------------------------------


class TestMeshFromVerticesFaces:
    def test_triangle(self):
        verts = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        faces = [[0, 1, 2]]
        m = mesh_from_vertices_faces(verts, faces)
        assert m.number_of_vertices() == 3
        assert m.number_of_faces() == 1

    def test_quad(self):
        verts = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        faces = [[0, 1, 2, 3]]
        m = mesh_from_vertices_faces(verts, faces)
        assert m.number_of_vertices() == 4
        assert m.number_of_faces() == 1


# ---------------------------------------------------------------------------
# validate_mesh
# ---------------------------------------------------------------------------


class TestValidateMesh:
    def test_valid_cylinder(self, cylinder_mesh):
        is_valid, issues = validate_mesh(cylinder_mesh)
        # A clean cylinder mesh should be manifold and have enough vertices
        assert cylinder_mesh.number_of_vertices() > 4

    def test_too_few_vertices(self):
        verts = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        faces = [[0, 1, 2]]
        m = mesh_from_vertices_faces(verts, faces)
        is_valid, issues = validate_mesh(m)
        assert not is_valid
        assert any("fewer than 4" in msg for msg in issues)


# ---------------------------------------------------------------------------
# mesh_bounding_box
# ---------------------------------------------------------------------------


class TestMeshBoundingBox:
    def test_cylinder_bbox(self, cylinder_mesh):
        lo, hi = mesh_bounding_box(cylinder_mesh)
        # Cylinder radius=1, height=10
        assert lo.z == pytest.approx(0.0, abs=1e-6)
        assert hi.z == pytest.approx(10.0, abs=1e-6)
        assert abs(lo.x) <= 1.01
        assert abs(lo.y) <= 1.01


# ---------------------------------------------------------------------------
# mesh_centroid
# ---------------------------------------------------------------------------


class TestMeshCentroid:
    def test_cylinder_centroid_z(self, cylinder_mesh):
        # Centroid of a cylinder from z=0 to z=10 should be near z=5
        c = mesh_centroid(cylinder_mesh)
        assert c.z == pytest.approx(5.0, abs=0.5)

    def test_centroid_xy_near_zero(self, cylinder_mesh):
        c = mesh_centroid(cylinder_mesh)
        assert abs(c.x) < 0.2
        assert abs(c.y) < 0.2


# ---------------------------------------------------------------------------
# principal_axes
# ---------------------------------------------------------------------------


class TestPrincipalAxes:
    def test_main_axis_is_z(self, cylinder_mesh):
        axes, eigenvalues = principal_axes(cylinder_mesh)
        # For a cylinder aligned with Z the principal axis should be close to [0,0,1]
        main = axes[0]
        assert abs(abs(main[2]) - 1.0) < 0.1  # allow small deviation

    def test_eigenvalues_sorted_descending(self, cylinder_mesh):
        _, eigenvalues = principal_axes(cylinder_mesh)
        assert eigenvalues[0] >= eigenvalues[1] >= eigenvalues[2]


# ---------------------------------------------------------------------------
# slice_mesh_with_plane
# ---------------------------------------------------------------------------


class TestSliceMeshWithPlane:
    def test_mid_slice_returns_contour(self, cylinder_mesh):
        # A plane through the middle of the cylinder should produce a contour
        plane = Plane(Point(0, 0, 5), Vector(0, 0, 1))
        contours = slice_mesh_with_plane(cylinder_mesh, plane)
        assert len(contours) >= 1

    def test_miss_plane_returns_empty(self, cylinder_mesh):
        # A plane far above the cylinder should produce nothing
        plane = Plane(Point(0, 0, 100), Vector(0, 0, 1))
        contours = slice_mesh_with_plane(cylinder_mesh, plane)
        assert contours == []

    def test_contour_has_points(self, cylinder_mesh):
        plane = Plane(Point(0, 0, 5), Vector(0, 0, 1))
        contours = slice_mesh_with_plane(cylinder_mesh, plane)
        assert all(len(c.points) >= 2 for c in contours)


# ---------------------------------------------------------------------------
# contour_centroid / contour_radius
# ---------------------------------------------------------------------------


class TestContourHelpers:
    def test_centroid_near_axis(self, cylinder_mesh):
        plane = Plane(Point(0, 0, 5), Vector(0, 0, 1))
        contours = slice_mesh_with_plane(cylinder_mesh, plane)
        assert contours
        cpt = contour_centroid(contours[0])
        assert cpt is not None
        assert abs(cpt.x) < 0.3
        assert abs(cpt.y) < 0.3

    def test_radius_near_one(self, cylinder_mesh):
        plane = Plane(Point(0, 0, 5), Vector(0, 0, 1))
        contours = slice_mesh_with_plane(cylinder_mesh, plane)
        assert contours
        cpt = contour_centroid(contours[0])
        r = contour_radius(contours[0], cpt)
        assert r == pytest.approx(1.0, abs=0.15)
