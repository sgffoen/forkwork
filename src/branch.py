from __future__ import annotations

from typing import Tuple

import numpy as np

from compas.datastructures import Mesh
from mesh_utils import mesh_bounding_box, principal_axes, skeletonize_mesh, mesh_bbox_to_world_xyz_transform, flip_mesh_top_bottom
from graph_utils import get_number_of_end_points_above_threshold, get_average_z_from_end_points


class Branch:
    """A single branch mesh.

    Parameters
    ----------
    mesh : :class:`compas.datastructures.Mesh`
        The mesh geometry of the branch.
    """

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self._skeleton_graph = None  # cached on first call to skeleton()
        self._is_bifurcation = None  # cached on first call to is_bifurcation()
        self._centerline = None

    def bounding_box(self):
        """Compute the oriented bounding box of the branch.

        Returns
        -------
        :class:`compas.geometry.Box`
        """
        return mesh_bounding_box(self.mesh)

    def pca(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the principal axes of the branch via PCA.

        Returns
        -------
        (axes, eigenvalues)
            *axes* : (3, 3) array – each row is a unit axis sorted by
            descending variance (row 0 is the longest direction).
            *eigenvalues* : (3,) array
        """
        return principal_axes(self.mesh)

    def skeleton_graph(self):
        """Compute and cache the skeleton of the branch.

        The result is computed once and stored on ``self._skeleton``.
        Subsequent calls return the cached result immediately.

        Parameters
        ----------

        Returns
        -------
        list of :class:`compas.geometry.Polyline` or graph
        """
        if self._skeleton_graph is None:
            self._skeleton_graph = skeletonize_mesh(self.mesh, graph=True)
        return self._skeleton_graph
    
    def number_of_end_points(self):
        """Count the number of end points (degree 1 nodes) in the skeleton graph.

        Returns
        -------
        int
            The number of end points in the skeleton graph.
        """
        graph = self.skeleton_graph()
        return len(list(graph.nodes_where({"degree": 1})))
    
    @property
    def is_bifurcation(self):
        if self._is_bifurcation is None:
            end_points = list(self.skeleton_graph().nodes_where({"degree": 1}))
            self._is_bifurcation = len(end_points) > 2
        return self._is_bifurcation
    
    def update_skeleton_graph(self):
        """Clear the cached skeleton graph so it will be recomputed on next call to skeleton_graph()
        """
        self._skeleton_graph = None
        self._is_bifurcation = None
        self._centerline = None

    def preprocess(self):
        """Align the branch to world XYZ and flip it if bifurcations point down.

        The mesh is first transformed so its bounding-box principal directions
        align with world axes. The transformed skeleton graph is then inspected:
        if at most one end point lies above ``threshold`` in Z, the branch is
        flipped top-to-bottom.

        Returns
        -------
        :class:`compas.datastructures.Mesh`
            The preprocessed mesh.
        """
        transform = mesh_bbox_to_world_xyz_transform(self.mesh)
        self.mesh = self.mesh.transformed(transform)

        graph = self.skeleton_graph()
        end_points_above_threshold = get_number_of_end_points_above_threshold(
            graph,
            get_average_z_from_end_points(graph)
        )

        if self.is_bifurcation:
            # only bifurcations have a top and bottom, regular branch has no top or bottom
            if end_points_above_threshold <= 1:
                # fork is upside down, flip it
                self.mesh = flip_mesh_top_bottom(self.mesh)
                self.update_skeleton_graph()

    def centerline(self):
        """Compute the centerline of the branch.

        Returns
        -------
        list of :class:`compas.geometry.Polyline`
            The centerline polylines of the branch.
        """
        if self._centerline is None:
            self._centerline = skeletonize_mesh(self.mesh, graph=False)
        return self._centerline
