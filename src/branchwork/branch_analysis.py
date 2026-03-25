"""
branch_analysis.py
==================
Branch topology analysis: bifurcation detection, local thickness profiles,
Frenet frames, and curvature computation.

All geometry is based on COMPAS (https://compas.dev).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from compas.datastructures import Mesh
from compas.geometry import Frame, Line, Plane, Point, Polyline, Vector

from .centerline import CenterlineExtractor
from .mesh_utils import (
    contour_centroid,
    contour_radius,
    mesh_centroid,
    principal_axes,
    slice_mesh_with_plane,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class BifurcationPoint:
    """A point where a branch splits into two or more sub-branches.

    Attributes
    ----------
    location      : Point     – 3-D position of the bifurcation.
    centerline_index : int    – index into the parent centerline polyline.
    angle_deg     : float     – bifurcation angle in degrees (between child axes).
    child_directions : list   – unit vectors pointing along each child branch.
    """

    location: Point
    centerline_index: int = 0
    angle_deg: float = 0.0
    child_directions: List[Vector] = field(default_factory=list)


@dataclass
class ThicknessProfile:
    """Local radius / thickness sampled along a centerline.

    Attributes
    ----------
    points  : list of Point  – sample positions on the centerline.
    radii   : list of float  – local radius at each sample.
    """

    points: List[Point] = field(default_factory=list)
    radii: List[float] = field(default_factory=list)

    @property
    def mean_radius(self) -> float:
        return float(np.mean(self.radii)) if self.radii else 0.0

    @property
    def max_radius(self) -> float:
        return float(np.max(self.radii)) if self.radii else 0.0

    @property
    def min_radius(self) -> float:
        return float(np.min(self.radii)) if self.radii else 0.0

    @property
    def taper_ratio(self) -> float:
        """Ratio of minimum to maximum radius (1 = uniform, 0 = fully tapered)."""
        if self.max_radius == 0.0:
            return 0.0
        return self.min_radius / self.max_radius


@dataclass
class BranchMetrics:
    """All computed metrics for a single branch mesh.

    Attributes
    ----------
    centerline       : Polyline
    length           : float          – arc-length of centerline.
    thickness        : ThicknessProfile
    bifurcations     : list of BifurcationPoint
    frames           : list of Frame  – Frenet frames along centerline.
    curvatures       : list of float  – local curvature at each centerline point.
    """

    centerline: Optional[Polyline] = None
    length: float = 0.0
    thickness: ThicknessProfile = field(default_factory=ThicknessProfile)
    bifurcations: List[BifurcationPoint] = field(default_factory=list)
    frames: List[Frame] = field(default_factory=list)
    curvatures: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main analyser class
# ---------------------------------------------------------------------------


class BranchAnalyzer:
    """Analyse a branch mesh to extract topology and geometry metrics.

    Parameters
    ----------
    mesh        : :class:`compas.datastructures.Mesh`
    num_slices  : int   – cross-section resolution (default 50).
    smooth_iter : int   – Laplacian smoothing passes for centerline (default 3).
    bif_threshold : float
        Minimum ratio of contour count to total slices that triggers
        bifurcation detection at a given cross-section.  Default ``0.5``.
    """

    def __init__(
        self,
        mesh: Mesh,
        num_slices: int = 50,
        smooth_iter: int = 3,
        bif_threshold: float = 0.5,
    ) -> None:
        self.mesh = mesh
        self.num_slices = num_slices
        self.smooth_iter = smooth_iter
        self.bif_threshold = bif_threshold
        self._metrics: Optional[BranchMetrics] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self) -> BranchMetrics:
        """Run full analysis and return :class:`BranchMetrics`.

        Steps
        -----
        1. Extract the centerline via :class:`~branchwork.CenterlineExtractor`.
        2. Build the thickness profile from per-slice radii.
        3. Detect bifurcation candidates (slices with multiple contours).
        4. Compute curvature along the centerline.
        """
        extractor = CenterlineExtractor(
            self.mesh,
            num_slices=self.num_slices,
            smooth_iterations=self.smooth_iter,
        )
        centerline = extractor.compute()

        thickness = ThicknessProfile(
            points=list(centerline.points),
            radii=extractor.radii,
        )

        bifurcations = self._detect_bifurcations(centerline, extractor.frames)
        curvatures = _compute_curvature(
            [np.array([p.x, p.y, p.z]) for p in centerline.points]
        )

        self._metrics = BranchMetrics(
            centerline=centerline,
            length=extractor.length,
            thickness=thickness,
            bifurcations=bifurcations,
            frames=extractor.frames,
            curvatures=curvatures,
        )
        return self._metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_bifurcations(
        self,
        centerline: Polyline,
        frames: List[Frame],
    ) -> List[BifurcationPoint]:
        """Detect bifurcations by looking for slices with multiple contours.

        When a cross-section plane intersects a bifurcated region it will
        produce **two or more** separate contour loops.  Each such slice is a
        bifurcation candidate.
        """
        axes, _ = principal_axes(self.mesh)
        main_axis = axes[0]
        normal = Vector(*main_axis)

        pts_arr = np.array([self.mesh.vertex_coordinates(v) for v in self.mesh.vertices()])
        centroid = pts_arr.mean(axis=0)
        projections = pts_arr.dot(main_axis)
        t_min, t_max = projections.min(), projections.max()
        margin = (t_max - t_min) * 0.02
        t_min += margin
        t_max -= margin

        step = (t_max - t_min) / (self.num_slices - 1)

        bifurcations: List[BifurcationPoint] = []
        cl_pts = list(centerline.points)
        n_pts = len(cl_pts)

        for i in range(self.num_slices):
            t = t_min + i * step
            origin = Point(*(main_axis * t + (centroid - main_axis * centroid.dot(main_axis))))
            plane = Plane(origin, normal)

            contours = slice_mesh_with_plane(self.mesh, plane)
            if len(contours) < 2:
                continue

            # Map slice index to closest centerline index
            cl_idx = min(int(round(i * (n_pts - 1) / (self.num_slices - 1))), n_pts - 1)
            bif_pt = cl_pts[cl_idx]

            # Compute per-contour centroids and derive child directions
            child_centroids = []
            for c in contours:
                cpt = contour_centroid(c)
                if cpt is not None:
                    child_centroids.append(np.array([cpt.x, cpt.y, cpt.z]))

            child_dirs: List[Vector] = []
            ref = np.array([bif_pt.x, bif_pt.y, bif_pt.z])
            for cc in child_centroids:
                d = cc - ref
                norm = np.linalg.norm(d)
                if norm > 1e-9:
                    child_dirs.append(Vector(*(d / norm)))

            angle = 0.0
            if len(child_dirs) >= 2:
                angle = math.degrees(
                    math.acos(
                        max(-1.0, min(1.0, child_dirs[0].dot(child_dirs[1])))
                    )
                )

            bifurcations.append(
                BifurcationPoint(
                    location=bif_pt,
                    centerline_index=cl_idx,
                    angle_deg=angle,
                    child_directions=child_dirs,
                )
            )

        return bifurcations


# ---------------------------------------------------------------------------
# Pure geometry helpers
# ---------------------------------------------------------------------------


def _compute_curvature(pts: List[np.ndarray]) -> List[float]:
    """Estimate discrete curvature at each interior point of a polyline.

    Uses the Menger curvature formula: κ = 4·Area(ABC) / (|AB|·|BC|·|CA|).
    End-points are assigned the curvature of their immediate neighbour.
    """
    n = len(pts)
    if n < 3:
        return [0.0] * n

    curvatures = [0.0] * n
    for i in range(1, n - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        ab = np.linalg.norm(b - a)
        bc = np.linalg.norm(c - b)
        ca = np.linalg.norm(a - c)

        if ab < 1e-12 or bc < 1e-12 or ca < 1e-12:
            curvatures[i] = 0.0
            continue

        cross = np.cross(b - a, c - a)
        area = np.linalg.norm(cross) / 2.0
        denom = ab * bc * ca
        curvatures[i] = (4.0 * area) / denom if denom > 1e-12 else 0.0

    curvatures[0] = curvatures[1]
    curvatures[-1] = curvatures[-2]
    return curvatures
