import numpy as np
from compas.colors import Color
from compas.colors import ColorMap
from compas.datastructures import Mesh
from compas.geometry import Box
from compas.geometry import Point
from compas.geometry import Polyline
from compas.geometry import Translation

from compas_cgal.geodesics import HeatGeodesicSolver
from compas_cgal.geodesics import geodesic_isolines
from compas_cgal.geodesics import geodesic_isolines_split
from compas_cgal.geodesics import heat_geodesic_distances



def closest_vertex(mesh, point):
    closest = None
    min_dist = float("inf")
    for v in mesh.vertices():
        v_pt = Point(*mesh.vertex_coordinates(v))
        dist = point.distance_to_point(v_pt)
        if dist < min_dist:
            min_dist = dist
            closest = v
    return closest


def get_mesh_vertices_from_points(mesh, points):
    vertices = []
    for pt in points:
        closest_v = closest_vertex(mesh, pt)
        vertices.append(closest_v)
    return vertices


def geodesic_distances(mesh, sources):
    V, F = mesh.to_vertices_and_faces()
    V_np = np.array(V)
    solver = HeatGeodesicSolver((V, F))
    dist = solver.solve(sources)
    return dist


def make_isolines(mesh, sources, offset, ISOVALUES):
    V, F = mesh.to_vertices_and_faces()
    polylines = []
    for pts in geodesic_isolines((V, F), sources, ISOVALUES):
        points = [[pts[i, 0] + offset[0], pts[i, 1] + offset[1], pts[i, 2]] for i in range(len(pts))]
        polylines.append(Polyline(points))
    return polylines
