# EdgeMender: A Topology Repair Algorithm for Voxel Boundary Meshes

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Actions status](https://github.com/MattTheCuber/edge-mender/actions/workflows/tests.yml/badge.svg)](https://github.com/MattTheCuber/edge-mender/actions)
<!-- Pytest Coverage Comment:Begin -->
<a href="https://github.com/MattTheCuber/edge-mender/blob/master/README.md"><img alt="Coverage" src="https://img.shields.io/badge/Coverage-100%25-brightgreen.svg" /></a><details><summary>Coverage Report </summary><table><tr><th>File</th><th>Stmts</th><th>Miss</th><th>Cover</th><th>Missing</th></tr><tbody><tr><td colspan="5"><b>edge_mender</b></td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/MattTheCuber/edge-mender/blob/master/edge_mender/__init__.py">__init__.py</a></td><td>6</td><td>0</td><td>100%</td><td>&nbsp;</td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/MattTheCuber/edge-mender/blob/master/edge_mender/data_factory.py">data_factory.py</a></td><td>135</td><td>0</td><td>100%</td><td>&nbsp;</td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/MattTheCuber/edge-mender/blob/master/edge_mender/edge_mender.py">edge_mender.py</a></td><td>192</td><td>0</td><td>100%</td><td>&nbsp;</td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/MattTheCuber/edge-mender/blob/master/edge_mender/geometry_helper.py">geometry_helper.py</a></td><td>43</td><td>0</td><td>100%</td><td>&nbsp;</td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/MattTheCuber/edge-mender/blob/master/edge_mender/mesh_generator.py">mesh_generator.py</a></td><td>74</td><td>0</td><td>100%</td><td>&nbsp;</td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/MattTheCuber/edge-mender/blob/master/edge_mender/visualizer.py">visualizer.py</a></td><td>46</td><td>0</td><td>100%</td><td>&nbsp;</td></tr><tr><td><b>TOTAL</b></td><td><b>496</b></td><td><b>0</b></td><td><b>100%</b></td><td>&nbsp;</td></tr></tbody></table></details>
<!-- Pytest Coverage Comment:End -->

This tool repairs non-manifold edges in voxel boundary meshes to fix downstream operations such as smoothing. Non-manifold edges are defined as edges shared by four faces. Voxel boundary meshes are 3D surface representations of a data set where the faces and vertices perfectly snap to cells in a structured voxel grid. Common uses for this algorithm are repairing output from [Cuberille](https://github.com/InsightSoftwareConsortium/ITKCuberille), [Surface Nets](https://vtk.org/doc/nightly/html/classvtkSurfaceNets3D.html) before smoothing, and non-adaptive [Dual Contouring](https://github.com/Kryolyz/Dual_Contouring_Voxel) implementation without vertex adjustment. The algorithm procedure works for quad meshes, but this implementation is currently limited to triangular meshes (see #3). This algorithm requires input meshes to have proper and consistent winding order.

The three algorithms listed above are provided in the [`mesh_generator`](edge_mender/mesh_generator.py) module. A set of test data sets are provided in the [`data_factory`](edge_mender/data_factory.py) module. A visualization tool is also provided in the [`visualizer`](edge_mender/visualizer.py) module. The [`results`](results) folder contains the [evaluation notebook](results/evaluate.ipynb) and the [example notebook](example.ipynb) demonstrates the tool in use.

## Usage Instructions

`pip install git+https://github.com/MattTheCuber/edge-mender`

```py
from edge_mender import EdgeMender

mesh: trimesh.Trimesh = ...

mender = EdgeMender(mesh)
mender.repair()
```

For a walkthrough example, see the [example notebook](example.ipynb).

## Contributor Instructions

1. Clone the repository using `https://github.com/MattTheCuber/edge-mender.git`
2. [Install uv](https://docs.astral.sh/uv/getting-started/installation)
3. Create a virtual environment using `uv venv`
4. Install all development dependencies using `uv sync --all-extras`
5. Create a branch and start writing code!
