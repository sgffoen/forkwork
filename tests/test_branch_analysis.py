"""
test_branch_analysis.py – unit tests for branchwork.branch_analysis
"""

import math
import pytest
import numpy as np

from branchwork.branch_analysis import (
    BranchAnalyzer,
    BranchMetrics,
    BifurcationPoint,
    ThicknessProfile,
    _compute_curvature,
)


# ---------------------------------------------------------------------------
# ThicknessProfile
# ---------------------------------------------------------------------------


class TestThicknessProfile:
    def _make_profile(self, radii):
        from compas.geometry import Point
        pts = [Point(float(i), 0, 0) for i in range(len(radii))]
        return ThicknessProfile(points=pts, radii=radii)

    def test_mean_radius(self):
        tp = self._make_profile([1.0, 2.0, 3.0])
        assert tp.mean_radius == pytest.approx(2.0)

    def test_min_max_radius(self):
        tp = self._make_profile([1.0, 2.0, 3.0])
        assert tp.min_radius == pytest.approx(1.0)
        assert tp.max_radius == pytest.approx(3.0)

    def test_taper_ratio(self):
        tp = self._make_profile([1.0, 2.0, 4.0])
        assert tp.taper_ratio == pytest.approx(0.25)

    def test_taper_ratio_uniform(self):
        tp = self._make_profile([2.0, 2.0, 2.0])
        assert tp.taper_ratio == pytest.approx(1.0)

    def test_empty_profile(self):
        tp = ThicknessProfile()
        assert tp.mean_radius == 0.0
        assert tp.taper_ratio == 0.0


# ---------------------------------------------------------------------------
# BranchAnalyzer
# ---------------------------------------------------------------------------


class TestBranchAnalyzer:
    def test_returns_branch_metrics(self, cylinder_mesh):
        analyzer = BranchAnalyzer(cylinder_mesh, num_slices=20)
        metrics = analyzer.analyze()
        assert isinstance(metrics, BranchMetrics)

    def test_centerline_not_none(self, cylinder_mesh):
        analyzer = BranchAnalyzer(cylinder_mesh, num_slices=20)
        metrics = analyzer.analyze()
        assert metrics.centerline is not None

    def test_length_approx(self, cylinder_mesh):
        analyzer = BranchAnalyzer(cylinder_mesh, num_slices=30)
        metrics = analyzer.analyze()
        assert metrics.length == pytest.approx(10.0, rel=0.15)

    def test_thickness_profile_populated(self, cylinder_mesh):
        analyzer = BranchAnalyzer(cylinder_mesh, num_slices=20)
        metrics = analyzer.analyze()
        assert len(metrics.thickness.radii) > 0
        assert len(metrics.thickness.points) > 0

    def test_mean_radius_approx(self, cylinder_mesh):
        analyzer = BranchAnalyzer(cylinder_mesh, num_slices=30)
        metrics = analyzer.analyze()
        assert metrics.thickness.mean_radius == pytest.approx(1.0, abs=0.25)

    def test_frames_count_matches_centerline(self, cylinder_mesh):
        analyzer = BranchAnalyzer(cylinder_mesh, num_slices=20)
        metrics = analyzer.analyze()
        assert len(metrics.frames) == len(metrics.centerline.points)

    def test_curvatures_count_matches_centerline(self, cylinder_mesh):
        analyzer = BranchAnalyzer(cylinder_mesh, num_slices=20)
        metrics = analyzer.analyze()
        assert len(metrics.curvatures) == len(metrics.centerline.points)

    def test_straight_cylinder_low_curvature(self, cylinder_mesh):
        """A straight cylinder should have near-zero curvature everywhere."""
        analyzer = BranchAnalyzer(cylinder_mesh, num_slices=30)
        metrics = analyzer.analyze()
        mean_curv = np.mean(metrics.curvatures)
        # Straight branch → curvature close to zero
        assert abs(mean_curv) < 0.5

    def test_tapered_mesh_taper_ratio_less_than_one(self, tapered_mesh):
        analyzer = BranchAnalyzer(tapered_mesh, num_slices=30)
        metrics = analyzer.analyze()
        assert metrics.thickness.taper_ratio < 1.0


# ---------------------------------------------------------------------------
# _compute_curvature
# ---------------------------------------------------------------------------


class TestComputeCurvature:
    def test_straight_line_zero_curvature(self):
        pts = [np.array([0.0, 0.0, float(i)]) for i in range(10)]
        curvs = _compute_curvature(pts)
        # Interior points should have zero curvature on a straight line
        for k in curvs[1:-1]:
            assert abs(k) < 1e-10

    def test_count_matches_input(self):
        pts = [np.array([float(i), float(i ** 2), 0.0]) for i in range(8)]
        curvs = _compute_curvature(pts)
        assert len(curvs) == len(pts)

    def test_curved_path_nonzero_curvature(self):
        # A circular arc should have non-zero curvature
        pts = [np.array([math.cos(t), math.sin(t), 0.0]) for t in np.linspace(0, math.pi, 20)]
        curvs = _compute_curvature(pts)
        mean_curv = np.mean(curvs[1:-1])
        assert mean_curv > 0.01

    def test_two_points_returns_zeros(self):
        pts = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
        curvs = _compute_curvature(pts)
        assert curvs == [0.0, 0.0]

    def test_end_points_equal_neighbours(self):
        pts = [np.array([float(i), float(i ** 2) * 0.1, 0.0]) for i in range(6)]
        curvs = _compute_curvature(pts)
        assert curvs[0] == curvs[1]
        assert curvs[-1] == curvs[-2]
