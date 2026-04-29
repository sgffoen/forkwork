"""
skeleton.py
===========
Topological skeletonisation of branch meshes using a graph-based approach.

The skeleton graph represents the connectivity of branches and their
bifurcation points. Each node is a :class:`compas.geometry.Point` and
edges connect adjacent skeleton nodes.

This module uses ``compas.datastructures.Graph`` as the underlying
data structure.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from compas.datastructures import Graph, Mesh
from compas.geometry import Point, Polyline, Vector

from .centerline import CenterlineExtractor
from .mesh_utils import principal_axes


class SkeletonGraph:
    """Build a skeleton graph from a branch mesh.

    The graph is constructed from the centerline points of the mesh plus
    any bifurcation nodes detected during cross-section analysis.

    Parameters
    ----------
    mesh        : :class:`compas.datastructures.Mesh`
    num_slices  : int   – cross-section resolution passed to
                          :class:`~branchwork.CenterlineExtractor`.
    smooth_iter : int   – Laplacian smoothing passes for centerline.

    Attributes
    ----------
    graph : :class:`compas.datastructures.Graph`
        The computed skeleton graph (after :meth:`build` is called).
    node_points : dict[int, Point]
        Mapping from graph node key to 3-D position.
    """

    def __init__(self, mesh: Mesh, num_slices: int = 50, smooth_iter: int = 3) -> None:
        self.mesh = mesh
        self.num_slices = num_slices
        self.smooth_iter = smooth_iter

        self.graph: Optional[Graph] = None
        self.node_points: Dict[int, Point] = {}
        self._centerline: Optional[Polyline] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> Graph:
        """Build and return the skeleton graph.

        Returns
        -------
        :class:`compas.datastructures.Graph`
        """
        extractor = CenterlineExtractor(
            self.mesh,
            num_slices=self.num_slices,
            smooth_iterations=self.smooth_iter,
        )
        centerline = extractor.compute()
        self._centerline = centerline
        pts = list(centerline.points)

        g = Graph()
        prev_key: Optional[int] = None

        for i, pt in enumerate(pts):
            node_attr = {"x": pt.x, "y": pt.y, "z": pt.z, "radius": extractor.radii[i]}
            key = g.add_node(**node_attr)
            self.node_points[key] = pt

            if prev_key is not None:
                g.add_edge(prev_key, key, weight=_dist(pts[i - 1], pt))

            prev_key = key

        self.graph = g
        return g

    def edge_lengths(self) -> List[float]:
        """Return the length of every edge in the skeleton graph."""
        if self.graph is None:
            return []
        lengths = []
        for u, v in self.graph.edges():
            pu = self.node_points[u]
            pv = self.node_points[v]
            lengths.append(_dist(pu, pv))
        return lengths

    def total_length(self) -> float:
        """Sum of all edge lengths in the skeleton graph."""
        return sum(self.edge_lengths())

    def leaf_nodes(self) -> List[int]:
        """Return node keys with degree 1 (tips of the branch)."""
        if self.graph is None:
            return []
        return [n for n in self.graph.nodes() if self.graph.degree(n) == 1]

    def junction_nodes(self) -> List[int]:
        """Return node keys with degree > 2 (bifurcation / junction points)."""
        if self.graph is None:
            return []
        return [n for n in self.graph.nodes() if self.graph.degree(n) > 2]

    def branches(self) -> List[List[int]]:
        """Decompose the skeleton into unbranched segments (chains).

        A *branch* is a maximal path between two nodes that each have
        degree ≠ 2 (i.e. tips or junctions).  Interior nodes with degree 2
        are part of the chain but not endpoints.

        Returns
        -------
        list of lists of node keys
        """
        if self.graph is None:
            return []

        special: Set[int] = set(self.leaf_nodes() + self.junction_nodes())
        if not special:
            # Circular path – return all nodes as one branch
            return [list(self.graph.nodes())]

        visited_edges: Set[Tuple[int, int]] = set()
        result: List[List[int]] = []

        for start in special:
            for nb in self.graph.neighbors(start):
                edge = (min(start, nb), max(start, nb))
                if edge in visited_edges:
                    continue
                chain = [start, nb]
                visited_edges.add(edge)
                current = nb
                prev = start
                while current not in special:
                    nbs = [n for n in self.graph.neighbors(current) if n != prev]
                    if not nbs:
                        break
                    nxt = nbs[0]
                    e2 = (min(current, nxt), max(current, nxt))
                    if e2 in visited_edges:
                        break
                    visited_edges.add(e2)
                    chain.append(nxt)
                    prev, current = current, nxt
                result.append(chain)

        return result

    @property
    def centerline(self) -> Optional[Polyline]:
        """The raw centerline polyline used to build the graph."""
        return self._centerline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dist(a: Point, b: Point) -> float:
    return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2)
