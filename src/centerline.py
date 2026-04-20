from compas.geometry import Polyline
from compas.datastructures import Mesh
import math
import numpy as np

from mesh_utils import principal_axes


def branch_direction(mesh):
    """Computes the principal direction of a branch mesh.

    Parameters
    ----------
    mesh : :class:`compas.datastructures.Mesh`
        Input mesh representing a single branch (tubular or irregular).

    Returns
    -------
    Vector
        The principal direction vector of the branch.
    """
    axes, _ = principal_axes(mesh)
    main_axis = axes[0]  # longest variance direction
    return main_axis


