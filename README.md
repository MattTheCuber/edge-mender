# EdgeMender: A Topology Repair Algorithm for Voxel Boundary Meshes

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Actions status](https://github.com/MattTheCuber/edge-mender/actions/workflows/tests.yml/badge.svg)](https://github.com/MattTheCuber/edge-mender/actions)
<!-- Pytest Coverage Comment:Begin -->
<a href="https://github.com/MattTheCuber/edge-mender/blob/master/README.md"><img alt="Coverage" src="https://img.shields.io/badge/Coverage-100%25-brightgreen.svg" /></a><details><summary>Coverage Report </summary><table><tr><th>File</th><th>Stmts</th><th>Miss</th><th>Cover</th><th>Missing</th></tr><tbody><tr><td colspan="5"><b>edge_mender</b></td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/MattTheCuber/edge-mender/blob/master/edge_mender/__init__.py">__init__.py</a></td><td>6</td><td>0</td><td>100%</td><td>&nbsp;</td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/MattTheCuber/edge-mender/blob/master/edge_mender/data_factory.py">data_factory.py</a></td><td>135</td><td>0</td><td>100%</td><td>&nbsp;</td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/MattTheCuber/edge-mender/blob/master/edge_mender/edge_mender.py">edge_mender.py</a></td><td>202</td><td>0</td><td>100%</td><td>&nbsp;</td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/MattTheCuber/edge-mender/blob/master/edge_mender/geometry_helper.py">geometry_helper.py</a></td><td>32</td><td>0</td><td>100%</td><td>&nbsp;</td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/MattTheCuber/edge-mender/blob/master/edge_mender/mesh_generator.py">mesh_generator.py</a></td><td>73</td><td>0</td><td>100%</td><td>&nbsp;</td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/MattTheCuber/edge-mender/blob/master/edge_mender/visualizer.py">visualizer.py</a></td><td>46</td><td>0</td><td>100%</td><td>&nbsp;</td></tr><tr><td><b>TOTAL</b></td><td><b>494</b></td><td><b>0</b></td><td><b>100%</b></td><td>&nbsp;</td></tr></tbody></table></details>
<!-- Pytest Coverage Comment:End -->

## Usage Instructions

```py
from edge_mender import EdgeMender

mesh: trimesh.Trimesh = ...

mender = EdgeMender(mesh)
mender.repair()
```
