"""
centerline.py
=============
True centerline extraction from branch meshes using COMPAS geometry.

Algorithm overview
------------------
1. Compute the principal axis of the mesh via PCA (the *longitudinal* axis).
2. Project all vertices onto the principal axis and find the extents.
3. Slice the mesh with equally-spaced planes perpendicular to the principal
   axis.
4. For each slice compute the centroid of the intersection contour.
5. Fit the sequence of centroids into a smooth polyline (the centerline).

The resulting centerline is a :class:`compas.geometry.Polyline`.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from compas.datastructures import Mesh
from compas.geometry import Frame, Plane, Point, Polyline, Vector

from .mesh_utils import (
    contour_centroid,
    contour_radius,
    mesh_centroid,
    principal_axes,
    slice_mesh_with_plane,
)


class CenterlineExtractor:
    """Extract the centerline of a branch mesh.

    Parameters
    ----------
    mesh : :class:`compas.datastructures.Mesh`
        Input mesh representing a single branch (tubular or irregular).
    num_slices : int
        Number of cross-sectional slices along the principal axis.
        More slices give higher resolution but increase computation time.
        Default is ``50``.
    smooth_iterations : int
        Number of Laplacian smoothing passes applied to the raw centroid
        sequence before returning the final polyline.  Default is ``3``.

    Attributes
    ----------
    centerline : Polyline or None
        Set after :meth:`compute` is called.
    radii : list of float
        Local radius (thickness / 2) at each centerline vertex.
    frames : list of Frame
        Frenet-like frame (tangent, normal, binormal) at each vertex.
    """

    def __init__(self, mesh: Mesh, num_slices: int = 50, smooth_iterations: int = 3) -> None:
        self.mesh = mesh
        self.num_slices = max(3, int(num_slices))
        self.smooth_iterations = max(0, int(smooth_iterations))

        self.centerline: Optional[Polyline] = None
        self.radii: List[float] = []
        self.frames: List[Frame] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self) -> Polyline:
        """Run the extraction and return the centerline polyline.

        Returns
        -------
        Polyline
            Ordered sequence of centroid points forming the centerline.

        Raises
        ------
        ValueError
            If the mesh produces no valid cross-section centroids.
        """
        axes, _ = principal_axes(self.mesh)
        main_axis = axes[0]  # longest variance direction

        pts_arr = np.array([self.mesh.vertex_coordinates(v) for v in self.mesh.vertices()])
        centroid = pts_arr.mean(axis=0)

        # Project all vertices onto main axis to find extent
        projections = pts_arr.dot(main_axis)
        t_min, t_max = projections.min(), projections.max()
        margin = (t_max - t_min) * 0.02  # 2 % inset to avoid cap artefacts
        t_min += margin
        t_max -= margin

        step = (t_max - t_min) / (self.num_slices - 1)
        normal = Vector(*main_axis)

        centroids: List[np.ndarray] = []
        raw_radii: List[float] = []

        for i in range(self.num_slices):
            t = t_min + i * step
            origin = Point(*(main_axis * t + (centroid - main_axis * centroid.dot(main_axis))))
            plane = Plane(origin, normal)

            contours = slice_mesh_with_plane(self.mesh, plane)
            if not contours:
                continue

            # Use the largest contour (by vertex count) as the representative
            main_contour = max(contours, key=lambda c: len(c.points))
            cpt = contour_centroid(main_contour)
            if cpt is None:
                continue

            r = contour_radius(main_contour, cpt)
            centroids.append(np.array([cpt.x, cpt.y, cpt.z]))
            raw_radii.append(r)

        if len(centroids) < 2:
            raise ValueError(
                "Centerline extraction produced fewer than 2 valid cross-sections. "
                "Check that the mesh is a closed, manifold branch."
            )

        # Smooth the centroid sequence
        smooth_pts = _laplacian_smooth(centroids, iterations=self.smooth_iterations)

        self.centerline = Polyline([Point(*p) for p in smooth_pts])
        self.radii = list(raw_radii[: len(smooth_pts)])
        # pad or trim radii to match point count
        while len(self.radii) < len(smooth_pts):
            self.radii.append(self.radii[-1])
        self.radii = self.radii[: len(smooth_pts)]

        self.frames = _compute_frames(smooth_pts)

        return self.centerline

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def length(self) -> float:
        """Approximate arc-length of the centerline polyline."""
        if self.centerline is None:
            return 0.0
        return self.centerline.length

    @property
    def mean_radius(self) -> float:
        """Mean local radius along the centerline."""
        if not self.radii:
            return 0.0
        return float(np.mean(self.radii))

    @property
    def min_radius(self) -> float:
        """Minimum local radius (narrowest cross-section)."""
        if not self.radii:
            return 0.0
        return float(np.min(self.radii))

    @property
    def max_radius(self) -> float:
        """Maximum local radius (widest cross-section)."""
        if not self.radii:
            return 0.0
        return float(np.max(self.radii))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _laplacian_smooth(
    pts: List[np.ndarray],
    iterations: int = 3,
    weight: float = 0.5,
) -> List[np.ndarray]:
    """Apply Laplacian smoothing to a 1-D chain of points.

    End-points are kept fixed to prevent the polyline from shrinking.
    """
    result = [p.copy() for p in pts]
    n = len(result)
    for _ in range(iterations):
        new_result = [result[0]]
        for i in range(1, n - 1):
            smoothed = result[i] + weight * ((result[i - 1] + result[i + 1]) / 2.0 - result[i])
            new_result.append(smoothed)
        new_result.append(result[-1])
        result = new_result
    return result


def _compute_frames(pts: List[np.ndarray]) -> List[Frame]:
    """Compute a Frenet-like frame at each point on the chain.

    Returns
    -------
    list of :class:`compas.geometry.Frame`
        Each frame has:
        - ``xaxis`` – tangent direction
        - ``yaxis`` – approximate normal (via cross-product with a reference)
    """
    frames: List[Frame] = []
    n = len(pts)

    # Reference vector for normal calculation (avoid degenerate cross-products)
    ref = np.array([0.0, 0.0, 1.0])

    for i in range(n):
        if i == 0:
            tangent = pts[1] - pts[0]
        elif i == n - 1:
            tangent = pts[-1] - pts[-2]
        else:
            tangent = pts[i + 1] - pts[i - 1]

        tangent_len = np.linalg.norm(tangent)
        if tangent_len < 1e-12:
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            tangent = tangent / tangent_len

        # Ensure ref is not parallel to tangent
        if abs(np.dot(tangent, ref)) > 0.99:
            ref = np.array([0.0, 1.0, 0.0])

        normal = np.cross(tangent, ref)
        normal_len = np.linalg.norm(normal)
        if normal_len < 1e-12:
            normal = np.array([0.0, 1.0, 0.0])
        else:
            normal = normal / normal_len

        frames.append(
            Frame(
                Point(*pts[i]),
                Vector(*tangent),
                Vector(*normal),
            )
        )

    return frames
