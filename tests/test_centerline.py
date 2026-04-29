"""
test_centerline.py – unit tests for branchwork.centerline
"""

import math
import pytest
import numpy as np

from compas.geometry import Polyline, Point

from branchwork.centerline import CenterlineExtractor, _laplacian_smooth, _compute_frames


# ---------------------------------------------------------------------------
# CenterlineExtractor
# ---------------------------------------------------------------------------


class TestCenterlineExtractor:
    def test_returns_polyline(self, cylinder_mesh):
        ext = CenterlineExtractor(cylinder_mesh, num_slices=20)
        cl = ext.compute()
        assert isinstance(cl, Polyline)

    def test_polyline_has_points(self, cylinder_mesh):
        ext = CenterlineExtractor(cylinder_mesh, num_slices=20)
        cl = ext.compute()
        assert len(cl.points) >= 2

    def test_centerline_length_approx(self, cylinder_mesh):
        """Centerline of a 10-unit cylinder should have length close to 10."""
        ext = CenterlineExtractor(cylinder_mesh, num_slices=30)
        ext.compute()
        assert ext.length == pytest.approx(10.0, rel=0.15)

    def test_mean_radius_approx(self, cylinder_mesh):
        """Mean radius of a unit cylinder should be close to 1."""
        ext = CenterlineExtractor(cylinder_mesh, num_slices=30)
        ext.compute()
        assert ext.mean_radius == pytest.approx(1.0, abs=0.25)

    def test_radii_count_matches_points(self, cylinder_mesh):
        ext = CenterlineExtractor(cylinder_mesh, num_slices=20)
        cl = ext.compute()
        assert len(ext.radii) == len(cl.points)

    def test_frames_count_matches_points(self, cylinder_mesh):
        ext = CenterlineExtractor(cylinder_mesh, num_slices=20)
        cl = ext.compute()
        assert len(ext.frames) == len(cl.points)

    def test_centerline_near_axis(self, cylinder_mesh):
        """All centerline points should be near the Z-axis (x≈0, y≈0)."""
        ext = CenterlineExtractor(cylinder_mesh, num_slices=30)
        cl = ext.compute()
        for pt in cl.points:
            assert abs(pt.x) < 0.3, f"x={pt.x} too far from axis"
            assert abs(pt.y) < 0.3, f"y={pt.y} too far from axis"

    def test_min_max_radius(self, cylinder_mesh):
        ext = CenterlineExtractor(cylinder_mesh, num_slices=20)
        ext.compute()
        assert ext.min_radius <= ext.mean_radius <= ext.max_radius

    def test_tapered_mesh_decreasing_radius(self, tapered_mesh):
        """The tapered mesh should show a difference between min and max radius."""
        ext = CenterlineExtractor(tapered_mesh, num_slices=30)
        ext.compute()
        assert ext.max_radius > ext.min_radius

    def test_few_slices_still_works(self, cylinder_mesh):
        ext = CenterlineExtractor(cylinder_mesh, num_slices=5)
        cl = ext.compute()
        assert len(cl.points) >= 2

    def test_no_smoothing(self, cylinder_mesh):
        ext = CenterlineExtractor(cylinder_mesh, num_slices=20, smooth_iterations=0)
        cl = ext.compute()
        assert cl is not None


# ---------------------------------------------------------------------------
# _laplacian_smooth
# ---------------------------------------------------------------------------


class TestLaplacianSmooth:
    def test_preserves_endpoints(self):
        pts = [np.array([float(i), 0.0, 0.0]) for i in range(10)]
        smoothed = _laplacian_smooth(pts, iterations=5)
        np.testing.assert_allclose(smoothed[0], pts[0])
        np.testing.assert_allclose(smoothed[-1], pts[-1])

    def test_reduces_jaggedness(self):
        # A zigzag chain should become smoother after Laplacian smoothing
        pts = [np.array([float(i), float((-1) ** i), 0.0]) for i in range(10)]
        smoothed = _laplacian_smooth(pts, iterations=10)
        # The y-variance should decrease
        y_before = np.std([p[1] for p in pts])
        y_after  = np.std([p[1] for p in smoothed])
        assert y_after < y_before

    def test_returns_same_count(self):
        pts = [np.array([float(i), 0.0, 0.0]) for i in range(8)]
        smoothed = _laplacian_smooth(pts, iterations=3)
        assert len(smoothed) == len(pts)

    def test_zero_iterations_unchanged(self):
        pts = [np.array([float(i), float(i), 0.0]) for i in range(5)]
        smoothed = _laplacian_smooth(pts, iterations=0)
        for a, b in zip(pts, smoothed):
            np.testing.assert_allclose(a, b)


# ---------------------------------------------------------------------------
# _compute_frames
# ---------------------------------------------------------------------------


class TestComputeFrames:
    def test_count_matches_points(self):
        pts = [np.array([0.0, 0.0, float(i)]) for i in range(10)]
        frames = _compute_frames(pts)
        assert len(frames) == 10

    def test_tangent_unit_length(self):
        pts = [np.array([0.0, 0.0, float(i)]) for i in range(10)]
        frames = _compute_frames(pts)
        for fr in frames:
            t = np.array([fr.xaxis.x, fr.xaxis.y, fr.xaxis.z])
            assert abs(np.linalg.norm(t) - 1.0) < 1e-6

    def test_straight_z_axis_tangent(self):
        pts = [np.array([0.0, 0.0, float(i)]) for i in range(10)]
        frames = _compute_frames(pts)
        for fr in frames:
            # Tangent should point along ±Z
            assert abs(abs(fr.xaxis.z) - 1.0) < 0.05
