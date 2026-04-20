from __future__ import annotations

import math

from compas.geometry import (
    oriented_bounding_box_numpy,
    Box,
    normal_polygon,
    Point,
    Vector,
    Rotation,
    transform_points,
    distance_point_point_xy,
    distance_point_line_xy,
    Frame,
    Circle,
)
from compas_cgal.straight_skeleton_2 import interior_straight_skeleton

GEOMETRY_TOLERANCE = 1e-6


def _polygon_normal_vector(polygon) -> Vector:
    normal = Vector(*normal_polygon(polygon.points, True))
    if normal.magnitude <= GEOMETRY_TOLERANCE:
        raise ValueError("Polygon normal is undefined for a degenerate polygon.")
    return normal

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def curve_bounding_box(curve, sample=50) -> Box:
    """Compute the oriented bounding box of a sampled curve.

    Parameters
    ----------
    curve : Curve
        The input curve.
    sample : int, optional
        Number of curve divisions used to sample the curve.

    Returns
    -------
    :class:`compas.geometry.Box`
        The oriented bounding box.
    """
    _, pts = curve.divide_by_count(sample, return_points=True)
    bbox = oriented_bounding_box_numpy(pts)
    bbox = Box.from_bounding_box(bbox)
    return bbox


def align_polygon_to_z_vector(polygon):
    """Create a rotation that aligns the polygon normal with the world Z axis."""
    target = Vector(0, 0, 1)

    normal = _polygon_normal_vector(polygon)

    if math.isclose(normal.z, 1.0, abs_tol=GEOMETRY_TOLERANCE):
        return None
    if math.isclose(normal.z, -1.0, abs_tol=GEOMETRY_TOLERANCE):
        return Rotation.from_axis_and_angle(Vector(1, 0, 0), math.pi, polygon.centroid)

    # rotation axis
    axis = normal.cross(target)
    if axis.magnitude <= GEOMETRY_TOLERANCE:
        return None

    # rotation angle
    angle = normal.angle(target)

    # rotation transform
    R = Rotation.from_axis_and_angle(axis, angle, polygon.centroid)
    return R


def min_distance_points_to_poly(points, polygon):
    """Compute the minimum XY distance from each point to the polygon boundary.

    Parameters
    ----------
    points : list of Point
        The input points.
    polygon : Polygon
        The input polygon.

    Returns
    -------
    list of float
        The minimum distances from the points to the polygon.
    """
    polygon_points = polygon.points
    if len(polygon_points) < 2:
        raise ValueError("Polygon boundary must contain at least two points.")

    segments = polygon.lines
    min_distances = []
    for pt in points:
        min_dist = float("inf")
        for seg in segments:
            dist = distance_point_line_xy(pt, seg)
            if dist < min_dist:
                min_dist = dist
        min_distances.append(min_dist)
    return min_distances


def skeleton_2D(polygon):
    """Compute straight-skeleton points for a planar polygon.

    Parameters
    ----------
    polygon : Polygon
        The input polygon.

    Returns
    -------
    list of Point
        Skeleton vertices in the polygon coordinate system.
    """
    polygon_pts = polygon.points
    normal = _polygon_normal_vector(polygon)
    R_inverse = None
    if not math.isclose(normal.z, 1.0, abs_tol=GEOMETRY_TOLERANCE):
        R = align_polygon_to_z_vector(polygon)
        if R is not None:
            polygon_pts = transform_points(polygon.points, R)
            R_inverse = R.inverse()
    points, indices, edges, edge_types = interior_straight_skeleton(polygon_pts, as_graph=False)
    if R_inverse is not None:
        points = transform_points(points, R_inverse)
    return [Point(*point) for point in points]


def circle_from_normal(center, normal, radius):
    """Create a circle with a given normal and radius.

    Parameters
    ----------
    center : Point
        The center of the circle.
    normal : Vector
        The normal vector of the circle's plane.
    radius : float
        The radius of the circle.

    Returns
    -------
    Circle
        The resulting circle.
    """
    center = Point(*center)
    if normal.magnitude <= GEOMETRY_TOLERANCE:
        raise ValueError("Circle normal cannot be zero-length.")

    ref = Vector(0, 0, 1)
    axis = normal.cross(ref)
    if axis.magnitude <= GEOMETRY_TOLERANCE:
        # normal is parallel to ref, choose a different reference vector
        ref = Vector(1, 0, 0)
        axis = normal.cross(ref)
    axis.unitize()
    yaxis = normal.cross(axis)
    yaxis.unitize()
    frame = Frame(center, axis, yaxis)
    return Circle(radius, frame)


def maximum_inscribed_circle(polygon):
    """Compute the maximum inscribed circle of a polygon.

    Parameters
    ----------
    polygon : Polygon
        The input polygon.

    Returns
    -------
    Circle
        The maximum inscribed circle.
    """
    skeleton_points = skeleton_2D(polygon)
    if not skeleton_points:
        raise ValueError("Straight skeleton produced no interior points.")

    min_distances = min_distance_points_to_poly(skeleton_points, polygon)
    max_index = max(range(len(min_distances)), key=min_distances.__getitem__)
    center = skeleton_points[max_index]
    radius = min_distances[max_index]
    normal = _polygon_normal_vector(polygon)
    return circle_from_normal(center, normal, radius)