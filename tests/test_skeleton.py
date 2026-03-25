"""
test_skeleton.py – unit tests for branchwork.skeleton
"""

import pytest

from compas.datastructures import Graph

from branchwork.skeleton import SkeletonGraph, _dist
from compas.geometry import Point


# ---------------------------------------------------------------------------
# SkeletonGraph
# ---------------------------------------------------------------------------


class TestSkeletonGraph:
    def test_build_returns_graph(self, cylinder_mesh):
        sg = SkeletonGraph(cylinder_mesh, num_slices=20)
        g = sg.build()
        assert isinstance(g, Graph)

    def test_graph_has_nodes(self, cylinder_mesh):
        sg = SkeletonGraph(cylinder_mesh, num_slices=20)
        sg.build()
        assert sg.graph.number_of_nodes() >= 2

    def test_graph_has_edges(self, cylinder_mesh):
        sg = SkeletonGraph(cylinder_mesh, num_slices=20)
        sg.build()
        assert sg.graph.number_of_edges() >= 1

    def test_node_points_populated(self, cylinder_mesh):
        sg = SkeletonGraph(cylinder_mesh, num_slices=20)
        sg.build()
        assert len(sg.node_points) == sg.graph.number_of_nodes()

    def test_centerline_set_after_build(self, cylinder_mesh):
        sg = SkeletonGraph(cylinder_mesh, num_slices=20)
        sg.build()
        assert sg.centerline is not None

    def test_edge_lengths_count(self, cylinder_mesh):
        sg = SkeletonGraph(cylinder_mesh, num_slices=20)
        sg.build()
        assert len(sg.edge_lengths()) == sg.graph.number_of_edges()

    def test_total_length_approx(self, cylinder_mesh):
        sg = SkeletonGraph(cylinder_mesh, num_slices=30)
        sg.build()
        assert sg.total_length() == pytest.approx(10.0, rel=0.15)

    def test_leaf_nodes_for_open_chain(self, cylinder_mesh):
        """A simple chain should have exactly 2 leaf nodes (the two ends)."""
        sg = SkeletonGraph(cylinder_mesh, num_slices=20)
        sg.build()
        leaves = sg.leaf_nodes()
        assert len(leaves) == 2

    def test_no_junction_nodes_on_simple_cylinder(self, cylinder_mesh):
        """A simple, non-bifurcated cylinder should have no junction nodes."""
        sg = SkeletonGraph(cylinder_mesh, num_slices=20)
        sg.build()
        junctions = sg.junction_nodes()
        assert len(junctions) == 0

    def test_branches_returns_one_branch_for_cylinder(self, cylinder_mesh):
        sg = SkeletonGraph(cylinder_mesh, num_slices=20)
        sg.build()
        branches = sg.branches()
        assert len(branches) >= 1

    def test_all_edge_lengths_positive(self, cylinder_mesh):
        sg = SkeletonGraph(cylinder_mesh, num_slices=20)
        sg.build()
        for el in sg.edge_lengths():
            assert el > 0.0


# ---------------------------------------------------------------------------
# _dist helper
# ---------------------------------------------------------------------------


class TestDist:
    def test_dist_origin(self):
        a = Point(0, 0, 0)
        b = Point(0, 0, 0)
        assert _dist(a, b) == pytest.approx(0.0)

    def test_dist_along_x(self):
        a = Point(0, 0, 0)
        b = Point(3, 0, 0)
        assert _dist(a, b) == pytest.approx(3.0)

    def test_dist_3d(self):
        a = Point(1, 2, 3)
        b = Point(4, 6, 3)
        assert _dist(a, b) == pytest.approx(5.0)
