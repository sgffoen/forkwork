from __future__ import annotations

import math
from typing import List, Optional, Tuple

from compas.datastructures import Graph
from compas.geometry import (
    oriented_bounding_box_numpy,
    Box,
    normal_polygon,
    Vector,
    Rotation,
    distance_point_point_xy,
    Point,
    Translation,
)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def curve_bounding_box(curve, sample=50) -> Box:
    """Compute the oriented bounding box of a set of points.

    Parameters
    ----------
    points : list of Point
        The input points.

    Returns
    -------
    :class:`compas.geometry.Box`
        The oriented bounding box.
    """
    t, pts = curve.divide_by_count(sample, return_points=True)
    bbox = oriented_bounding_box_numpy(pts)
    bbox = Box.from_bounding_box(bbox)
    return bbox


def get_transform_polygon_to_origin(polygon):
    """Get a transform that moves the polygon centroid to the world origin
    """
    centroid = polygon.centroid
    translation = Vector.from_start_end(centroid, Point(0, 0, 0))
    return Translation.from_vector(translation)


def get_transform_polygon_to_xy_plane(polygon):
    normal = Vector(*normal_polygon(polygon.points, True))
    if normal.z == -1.0:
        # rotate 180 degrees around X axis
        return Rotation.from_axis_and_angle(Vector(1, 0, 0), math.pi, polygon.centroid)
    target = Vector(0, 0, 1)
    axis = normal.cross(target)
    angle = normal.angle(target)
    R = Rotation.from_axis_and_angle(axis, angle, polygon.centroid)
    return R


def min_distance_points_to_poly(points, polygon):
    """Compute the minimum distance between point and polygon boundary from a set of points.

    Parameters
    ----------
    points : list of Point
        The input points.
    polygon : list of Point
        The input polygon vertices (in order).

    Returns
    -------
    list of float
        The minimum distances from the points to the polygon.
    """
    min_distances = []
    for pt in points:
        min_dist = float("inf")
        for j in polygon.points:
            dist = distance_point_point_xy(pt, j)
            if dist < min_dist:
                min_dist = dist
        min_distances.append(min_dist)
    return min_distances
