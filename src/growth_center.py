"""
growth_center.py
================
Compute the growth center of a polygon, defined as the center of the maximum inscribed circle.

Algorithm overview
------------------
1. Compute the straight skeleton of the polygon to get candidate interior points.
2. For each skeleton point, compute the minimum distance to the polygon boundary.
3. The skeleton point with the maximum minimum distance is the center of the maximum inscribed circle.
4. The radius of the maximum inscribed circle is that maximum minimum distance.
5. The growth center is the center of the maximum inscribed circle.
"""

from typing import List, Tuple
import numpy as np
from helpers import (
    get_transform_polygon_to_origin,
    get_transform_polygon_to_xy_plane,
    min_distance_points_to_poly,
)
from compas.geometry import (
    Circle,
    Frame,
    Vector,
)
from compas_cgal.straight_skeleton_2 import interior_straight_skeleton


def maximum_inscribed_circle(polygon):
    """Compute the maximum inscribed circle of a polygon.

    Parameters
    ----------
    polygon : list of Point
        The input polygon vertices (in order).

    Returns
    -------
    Circle
        The maximum inscribed circle.
    """
    T = get_transform_polygon_to_origin(polygon)
    R = get_transform_polygon_to_xy_plane(polygon)
    # T * R: rotate around centroid first, then translate centroid to origin.
    # R * T would apply T first (centroid already at origin) then rotate around
    # the original centroid position, which is now the wrong pivot.
    combined = T * R
    polygon_transformed = polygon.transformed(combined)
    sk_points, indices, edges, edge_types = interior_straight_skeleton(polygon_transformed, as_graph=False)
    # sk_points are in transformed space, so measure against the transformed polygon
    min_distances = min_distance_points_to_poly(sk_points, polygon_transformed)
    max_index = np.argmax(min_distances)
    center = sk_points[max_index]
    radius = min_distances[max_index]
    circle = Circle(radius, Frame(center, Vector(1, 0, 0), Vector(0, 1, 0)))
    inverse = combined.inverse()
    circle.transform(inverse)
    return circle