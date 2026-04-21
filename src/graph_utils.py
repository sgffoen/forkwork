from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import statistics

from compas.datastructures import Graph


def graph_from_polylines(polylines):
    """Construct a graph from a list of polylines.

    Parameters
    ----------
    polylines : list of compas.geometry.Polyline
        The input polylines.

    Returns
    -------
    dict
        A graph represented as an adjacency list, where keys are points and values are lists of connected points.
    """
    graph = Graph.from_edges([(tuple(line.start), tuple(line.end)) for line in polylines])
    return graph


def get_average_z_from_end_points(graph):
    """Compute the average Z value of the end points (degree 1 nodes) in the graph.
    Parameters
    ----------
    graph : Graph
        The input graph.
    Returns
    -------
    float
        The average Z value of the end points.
    """
    return statistics.mean([n[2] for n in graph.nodes_where({"degree":1})])


def get_number_of_end_points_above_threshold(graph, threshold):
    """Count the number of end points (degree 1 nodes) in the graph that have a Z value above a given threshold.

    Parameters
    ----------
    graph : Graph
        The input graph.
    threshold : float
        The Z value threshold.

    Returns
    -------
    int
        The number of end points above the threshold.
    """
    return sum(1 for _, _, z in graph.nodes_where({"degree":1}) if z > threshold)