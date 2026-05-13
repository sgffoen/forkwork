from __future__ import annotations

import itertools
import math
from typing import Iterable, List, Sequence, Tuple

from compas.datastructures import Mesh
from compas.geometry import Point, Polyline, Vector


EPS = 1e-9


def reconstruct_y_mesh_from_loops_and_axes(
	boundary_loops: Sequence[Polyline],
	skeleton_axes: Sequence[Polyline],
	ring_vertex_count: int = 32,
	sections_per_axis: int = 7,
	junction_axis_fraction: float = 0.12,
	junction_scale: float = 0.35,
) -> Mesh:
	"""Reconstruct a Y-topology mesh from 3 boundary loops and 3 skeleton axes.

	Parameters
	----------
	boundary_loops : sequence[:class:`compas.geometry.Polyline`]
		Exactly 3 closed boundary loops (one per branch opening).
	skeleton_axes : sequence[:class:`compas.geometry.Polyline`]
		Exactly 3 skeleton axis polylines that form one Y junction.
	ring_vertex_count : int, optional
		Number of vertices used per ring after resampling.
	sections_per_axis : int, optional
		Number of sweep sections from each boundary loop to the junction ring.
	junction_axis_fraction : float, optional
		Fraction of axis length kept near the junction (0..0.49).
	junction_scale : float, optional
		Section scale at the junction end (0..1). Smaller values reduce overlap.

	Returns
	-------
	:class:`compas.datastructures.Mesh`
		A manifold triangle/quad mesh with all 3 boundary loops kept intact.
	"""
	loops = _validate_three_polylines(boundary_loops, "boundary_loops")
	axes = _validate_three_polylines(skeleton_axes, "skeleton_axes")

	ring_vertex_count = max(8, int(ring_vertex_count))
	sections_per_axis = max(2, int(sections_per_axis))
	junction_axis_fraction = max(0.02, min(0.49, float(junction_axis_fraction)))
	junction_scale = max(0.08, min(0.95, float(junction_scale)))

	axis_point_lists = [_polyline_to_points(a) for a in axes]
	axis_point_lists, junction_point = _orient_axes_from_shared_junction(axis_point_lists)

	axis_leaf_points = [pts[-1] for pts in axis_point_lists]
	loops = _match_loops_to_axes(loops, axis_leaf_points)

	resampled_loops = [
		_resample_closed_ring(_polyline_to_points(lp, closed=True), ring_vertex_count)
		for lp in loops
	]

	for i in range(3):
		resampled_loops[i] = _orient_ring_to_axis_leaf(
			ring=resampled_loops[i],
			axis_leaf=axis_leaf_points[i],
			axis_neighbor=axis_point_lists[i][-2],
			target=junction_point,
		)

	branch_sections = []
	join_rings = []
	for axis_idx in range(3):
		sections = _build_branch_sections(
			ring=resampled_loops[axis_idx],
			axis_points=axis_point_lists[axis_idx],
			sections_per_axis=sections_per_axis,
			junction_axis_fraction=junction_axis_fraction,
			junction_scale=junction_scale,
		)
		branch_sections.append(sections)
		join_rings.append(sections[-1])

	center_ring = _build_center_ring(join_rings)

	vertices: List[List[float]] = []
	faces: List[List[int]] = []

	branch_vertex_ids = []
	for sections in branch_sections:
		section_ids = []
		for ring in sections:
			ids = [
				_append_vertex(vertices, p)
				for p in ring
			]
			section_ids.append(ids)
		branch_vertex_ids.append(section_ids)

	for axis_idx in range(3):
		section_ids = branch_vertex_ids[axis_idx]
		for a, b in zip(section_ids[:-1], section_ids[1:]):
			faces.extend(_quad_strip_faces(a, b))

	center_ring_ids = [_append_vertex(vertices, p) for p in center_ring]

	for axis_idx in range(3):
		join_ids = branch_vertex_ids[axis_idx][-1]
		faces.extend(_quad_strip_faces(join_ids, center_ring_ids))

	center_point = _centroid(center_ring)
	center_id = _append_vertex(vertices, center_point)
	for i in range(ring_vertex_count):
		j = (i + 1) % ring_vertex_count
		faces.append([center_id, center_ring_ids[i], center_ring_ids[j]])

	mesh = Mesh.from_vertices_and_faces(vertices, faces)
	mesh.remove_duplicate_vertices()
	mesh.remove_unused_vertices()
	return mesh


def _validate_three_polylines(data: Sequence[Polyline], name: str) -> List[Polyline]:
	if len(data) != 3:
		raise ValueError("{} must contain exactly 3 polylines.".format(name))

	result = []
	for idx, item in enumerate(data):
		if not isinstance(item, Polyline):
			raise TypeError("{}[{}] must be a compas Polyline.".format(name, idx))
		points = list(item.points)
		if len(points) < 2:
			raise ValueError("{}[{}] must have at least 2 points.".format(name, idx))
		result.append(item)
	return result


def _polyline_to_points(polyline: Polyline, closed: bool = False) -> List[Point]:
	points = [Point(*p) for p in polyline.points]
	if closed and len(points) >= 2 and points[0].distance_to_point(points[-1]) <= EPS:
		points = points[:-1]
	return points


def _distance(a: Point, b: Point) -> float:
	return a.distance_to_point(b)


def _centroid(points: Sequence[Point]) -> Point:
	n = float(len(points))
	return Point(
		sum(p.x for p in points) / n,
		sum(p.y for p in points) / n,
		sum(p.z for p in points) / n,
	)


def _normalize(v: Vector, fallback: Vector | None = None) -> Vector:
	l = v.length
	if l <= EPS:
		if fallback is None:
			return Vector(1.0, 0.0, 0.0)
		return _normalize(fallback)
	return Vector(v.x / l, v.y / l, v.z / l)


def _project_perpendicular(v: Vector, axis: Vector) -> Vector:
	a = _normalize(axis, Vector(0.0, 0.0, 1.0))
	d = v.dot(a)
	return Vector(v.x - d * a.x, v.y - d * a.y, v.z - d * a.z)


def _fallback_perp(axis: Vector) -> Vector:
	a = _normalize(axis, Vector(0.0, 0.0, 1.0))
	ref = Vector(0.0, 0.0, 1.0)
	if abs(a.dot(ref)) > 0.95:
		ref = Vector(1.0, 0.0, 0.0)
	return _normalize(a.cross(ref), Vector(1.0, 0.0, 0.0))


def _ring_normal(ring: Sequence[Point]) -> Vector:
	nx = ny = nz = 0.0
	n = len(ring)
	for i in range(n):
		p = ring[i]
		q = ring[(i + 1) % n]
		nx += (p.y - q.y) * (p.z + q.z)
		ny += (p.z - q.z) * (p.x + q.x)
		nz += (p.x - q.x) * (p.y + q.y)
	return _normalize(Vector(nx, ny, nz), Vector(0.0, 0.0, 1.0))


def _polyline_length(points: Sequence[Point]) -> float:
	return sum(_distance(a, b) for a, b in zip(points[:-1], points[1:]))


def _sample_path(points: Sequence[Point], distance: float) -> Tuple[Point, Vector]:
	if len(points) < 2:
		raise ValueError("Axis polyline must have at least 2 points.")

	d = max(0.0, float(distance))
	for a, b in zip(points[:-1], points[1:]):
		seg = b - a
		seg_len = seg.length
		if seg_len <= EPS:
			continue
		if d <= seg_len:
			t = d / seg_len
			pt = Point(
				a.x + t * (b.x - a.x),
				a.y + t * (b.y - a.y),
				a.z + t * (b.z - a.z),
			)
			return pt, _normalize(seg, Vector(0.0, 0.0, 1.0))
		d -= seg_len

	end = points[-1]
	tan = points[-1] - points[-2]
	return Point(*end), _normalize(tan, Vector(0.0, 0.0, 1.0))


def _resample_closed_ring(ring: Sequence[Point], count: int) -> List[Point]:
	if len(ring) < 3:
		raise ValueError("Boundary loop must contain at least 3 unique points.")

	ring = list(ring)
	n = len(ring)
	seg_lengths = []
	total = 0.0
	for i in range(n):
		a = ring[i]
		b = ring[(i + 1) % n]
		l = _distance(a, b)
		seg_lengths.append(l)
		total += l

	if total <= EPS:
		raise ValueError("Boundary loop has near-zero perimeter.")

	distances = [k * total / float(count) for k in range(count)]

	result = []
	cursor = 0
	walked = 0.0
	for target in distances:
		while True:
			l = seg_lengths[cursor]
			if walked + l + EPS >= target:
				break
			walked += l
			cursor = (cursor + 1) % n

		a = ring[cursor]
		b = ring[(cursor + 1) % n]
		l = max(seg_lengths[cursor], EPS)
		t = max(0.0, min(1.0, (target - walked) / l))
		result.append(
			Point(
				a.x + t * (b.x - a.x),
				a.y + t * (b.y - a.y),
				a.z + t * (b.z - a.z),
			)
		)

	return result


def _orient_axes_from_shared_junction(
	axis_lists: Sequence[Sequence[Point]],
) -> Tuple[List[List[Point]], Point]:
	endpoints = [
		(axis[0], axis[-1])
		for axis in axis_lists
	]

	best_choice = None
	best_score = float("inf")
	for picks in itertools.product([0, 1], repeat=3):
		chosen = [endpoints[i][picks[i]] for i in range(3)]
		c = _centroid(chosen)
		score = sum(_distance(p, c) for p in chosen)
		if score < best_score:
			best_score = score
			best_choice = picks

	if best_choice is None:
		raise ValueError("Could not infer shared Y-junction from skeleton axes.")

	oriented = []
	junction_samples = []
	for i, axis in enumerate(axis_lists):
		pick = best_choice[i]
		if pick == 0:
			oriented.append(list(axis))
			junction_samples.append(axis[0])
		else:
			oriented.append(list(reversed(axis)))
			junction_samples.append(axis[-1])

	return oriented, _centroid(junction_samples)


def _match_loops_to_axes(loops: Sequence[Polyline], axis_leaf_points: Sequence[Point]) -> List[Polyline]:
	loop_centers = [_centroid(_polyline_to_points(lp, closed=True)) for lp in loops]
	best_perm = None
	best_score = float("inf")
	for perm in itertools.permutations(range(3)):
		score = 0.0
		for axis_i, loop_i in enumerate(perm):
			score += _distance(loop_centers[loop_i], axis_leaf_points[axis_i])
		if score < best_score:
			best_score = score
			best_perm = perm

	if best_perm is None:
		return list(loops)
	return [loops[i] for i in best_perm]


def _orient_ring_to_axis_leaf(
	ring: Sequence[Point],
	axis_leaf: Point,
	axis_neighbor: Point,
	target: Point,
) -> List[Point]:
	ring = list(ring)
	center = _centroid(ring)
	tangent = _normalize(axis_leaf - axis_neighbor, Vector(0.0, 0.0, 1.0))

	normal = _ring_normal(ring)
	if normal.dot(tangent) < 0.0:
		ring.reverse()

	aim = target - center
	aim = _project_perpendicular(aim, tangent)
	if aim.length <= EPS:
		aim = _fallback_perp(tangent)
	aim = _normalize(aim)

	best_idx = min(
		range(len(ring)),
		key=lambda i: _project_perpendicular(ring[i] - center, tangent).angle(aim),
	)
	return ring[best_idx:] + ring[:best_idx]


def _build_branch_sections(
	ring: Sequence[Point],
	axis_points: Sequence[Point],
	sections_per_axis: int,
	junction_axis_fraction: float,
	junction_scale: float,
) -> List[List[Point]]:
	ring = list(ring)
	leaf_center = _centroid(ring)

	total = _polyline_length(axis_points)
	if total <= EPS:
		raise ValueError("Skeleton axis has near-zero length.")

	leaf_tangent = _normalize(axis_points[-1] - axis_points[-2], Vector(0.0, 0.0, 1.0))
	ring_normal = _ring_normal(ring)
	x0 = _project_perpendicular(ring_normal, leaf_tangent)
	if x0.length <= EPS:
		x0 = _fallback_perp(leaf_tangent)
	x0 = _normalize(x0)
	y0 = _normalize(leaf_tangent.cross(x0), _fallback_perp(leaf_tangent))

	local_uv = []
	for p in ring:
		d = p - leaf_center
		local_uv.append((d.dot(x0), d.dot(y0)))

	min_s = total * junction_axis_fraction
	params = [
		k / float(sections_per_axis - 1)
		for k in range(sections_per_axis)
	]

	sections = []
	x_prev = x0
	for t in params:
		# travel from leaf (s=total) to junction-side cutoff (s=min_s)
		s = total - (total - min_s) * t
		center, tangent = _sample_path(axis_points, s)

		x = _project_perpendicular(x_prev, tangent)
		if x.length <= EPS:
			x = _project_perpendicular(x0, tangent)
		if x.length <= EPS:
			x = _fallback_perp(tangent)
		x = _normalize(x)
		y = _normalize(tangent.cross(x), _fallback_perp(tangent))
		x_prev = x

		alpha = t * t * (3.0 - 2.0 * t)
		scale = (1.0 - alpha) + alpha * junction_scale

		section_ring = []
		for u, v in local_uv:
			section_ring.append(
				Point(
					center.x + scale * (u * x.x + v * y.x),
					center.y + scale * (u * x.y + v * y.y),
					center.z + scale * (u * x.z + v * y.z),
				)
			)
		sections.append(section_ring)

	# Keep the original boundary loop exactly as input at section 0.
	sections[0] = ring
	return sections


def _build_center_ring(join_rings: Sequence[Sequence[Point]]) -> List[Point]:
	n = len(join_rings[0])
	center_ring = []
	for i in range(n):
		center_ring.append(
			Point(
				(join_rings[0][i].x + join_rings[1][i].x + join_rings[2][i].x) / 3.0,
				(join_rings[0][i].y + join_rings[1][i].y + join_rings[2][i].y) / 3.0,
				(join_rings[0][i].z + join_rings[1][i].z + join_rings[2][i].z) / 3.0,
			)
		)
	return center_ring


def _append_vertex(vertices: List[List[float]], p: Point) -> int:
	vertices.append([float(p.x), float(p.y), float(p.z)])
	return len(vertices) - 1


def _quad_strip_faces(ring_a: Sequence[int], ring_b: Sequence[int]) -> List[List[int]]:
	n = len(ring_a)
	faces = []
	for i in range(n):
		j = (i + 1) % n
		faces.append([ring_a[i], ring_a[j], ring_b[j], ring_b[i]])
	return faces

