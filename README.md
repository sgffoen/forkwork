# forkwork

**Grasshopper/Rhino plugin for irregular and bifurcated branches.**

`forkwork` extracts true centerlines and analyses branch topology from mesh
inputs. All geometry processing is built on the
[COMPAS](https://compas.dev) framework.

---

## Features

| Feature | Description |
|---|---|
| **Centerline extraction** | PCA-guided mesh sectioning → Laplacian-smoothed centerline polyline |
| **Thickness profile** | Per-point local radius, mean / min / max radius, taper ratio |
| **Bifurcation detection** | Finds cross-sections with multiple contours; reports location & angle |
| **Frenet frames** | Tangent–normal–binormal frame at every centerline vertex |
| **Curvature** | Menger discrete curvature along the centerline |
| **Skeleton graph** | Graph of centerline nodes/edges; leaf & junction node queries; branch decomposition |
| **Grasshopper components** | Plug-and-play GhPython scripts for Rhino/Grasshopper |

---

## Repository layout

```
src/
  branchwork/
    __init__.py          – package entry point
    mesh_utils.py        – mesh loading, validation, slicing (COMPAS)
    centerline.py        – CenterlineExtractor class
    branch_analysis.py   – BranchAnalyzer, BranchMetrics, bifurcation data
    skeleton.py          – SkeletonGraph class
grasshopper/
  components/
    GH_ExtractCenterline.py  – GhPython node: mesh → centerline
    GH_AnalyzeBranch.py      – GhPython node: full branch metrics
    GH_MeshToSkeleton.py     – GhPython node: mesh → skeleton graph
tests/
  conftest.py                – shared fixtures (cylinder & tapered meshes)
  test_mesh_utils.py
  test_centerline.py
  test_branch_analysis.py
  test_skeleton.py
pyproject.toml
requirements.txt
```

---

## Installation

```bash
pip install -e ".[dev]"
```

Or just install the runtime dependencies:

```bash
pip install compas numpy
```

---

## Quick start (Python)

```python
from branchwork.mesh_utils import mesh_from_vertices_faces
from branchwork.centerline import CenterlineExtractor
from branchwork.branch_analysis import BranchAnalyzer
from branchwork.skeleton import SkeletonGraph

# Build a COMPAS mesh from raw data
mesh = mesh_from_vertices_faces(vertices, faces)

# --- Centerline ---
extractor = CenterlineExtractor(mesh, num_slices=50)
centerline = extractor.compute()          # compas.geometry.Polyline
print("length:", extractor.length)
print("mean radius:", extractor.mean_radius)

# --- Full analysis ---
analyzer = BranchAnalyzer(mesh, num_slices=50)
metrics = analyzer.analyze()
print("bifurcations:", len(metrics.bifurcations))
print("taper ratio:", metrics.thickness.taper_ratio)

# --- Skeleton graph ---
sg = SkeletonGraph(mesh, num_slices=50)
graph = sg.build()                        # compas.datastructures.Graph
print("leaf nodes:", sg.leaf_nodes())
print("junction nodes:", sg.junction_nodes())
print("branches:", len(sg.branches()))
```

---

## Grasshopper usage

1. Open a GhPython Script component in Grasshopper.
2. Paste the contents of one of the scripts from
   `grasshopper/components/` into the editor.
3. Wire your Rhino mesh into the `mesh` input.
4. Optionally adjust `num_slices` (default 50) and `smooth_iter` (default 3).

### Components

| Script | Inputs | Outputs |
|---|---|---|
| `GH_ExtractCenterline.py` | mesh, num_slices, smooth_iter | centerline, radii, frames, length |
| `GH_AnalyzeBranch.py` | mesh, num_slices, smooth_iter | centerline, length, mean/min/max/taper radius, radii, bifurcations, bif_angles, curvatures, frames |
| `GH_MeshToSkeleton.py` | mesh, num_slices, smooth_iter | centerline, node_points, edge_lines, edge_lengths, leaf_points, junction_points, branch_curves, total_length |

---

## Running the tests

```bash
pytest
```

All 65 tests should pass.

---

## Dependencies

- [COMPAS](https://compas.dev) ≥ 2.0
- [NumPy](https://numpy.org) ≥ 1.21

