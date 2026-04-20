import numpy as np
from compas.datastructures import Mesh
from compas.geometry import Box
from compas.geometry import Point
from compas.geometry import Polyline
from compas.geometry import Translation

from compas_cgal.geodesics import HeatGeodesicSolver
from compas_cgal.geodesics import geodesic_isolines
from compas_cgal.geodesics import geodesic_isolines_split
from compas_cgal.geodesics import heat_geodesic_distances


def make_mesh(V, F, offset):
    m = Mesh.from_vertices_and_faces(V, F)
    m.transform(Translation.from_vector([offset[0], offset[1], 0]))
    return m


def make_isolines(sources, offset, V, F, ISOVALUES):
    polylines = []
    for pts in geodesic_isolines((V, F), sources, ISOVALUES):
        points = [[pts[i, 0] + offset[0], pts[i, 1] + offset[1], pts[i, 2]] for i in range(len(pts))]
        polylines.append(Polyline(points))
    return polylines


def iso_contours(mesh):
    V, F = mesh.to_vertices_and_faces()
    V_np = np.array(V)

    # =============================================================================
    # Config
    # =============================================================================

    X_OFF, Y_OFF = 0.75, 1.0
    ISOVALUES = [i / 50 for i in range(50)]
    SPLIT_ISOVALUES = [i / 20 for i in range(20)]

    # =============================================================================
    # Single sources
    # =============================================================================

    src1 = [0]
    dist1 = heat_geodesic_distances((V, F), src1)

    polylines = make_isolines(src1, (X_OFF, 0), V, F, ISOVALUES)

    return polylines