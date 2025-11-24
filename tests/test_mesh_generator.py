"""Test the DataFactory class."""

import pytest
from numpy.typing import NDArray

from edge_mender.data_factory import DataFactory
from edge_mender.mesh_generator import MeshGenerator


@pytest.mark.slow
@pytest.mark.parametrize(
    "data",
    [
        DataFactory.simple_extrusion(),
        DataFactory.double_extrusion(),
        DataFactory.triple_extrusion(),
        DataFactory.stairs(),
        DataFactory.ceiling(),
        # These two cases fail due to the Cuberille implementation in ITK
        # DataFactory.double_tower_ceiling(.),
        DataFactory.hanging_points(),
        # DataFactory.checkerboard(.),
        DataFactory.hole(),
        DataFactory.kill_you(),
        DataFactory.random(size=8, seed=0),
    ],
)
def test_to_mesh_cuberille(data: NDArray) -> None:
    """Test MeshGenerator.to_mesh_cuberille."""
    mesh = MeshGenerator.to_mesh_cuberille(data)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    assert mesh.volume > 0


@pytest.mark.parametrize(
    "data",
    [
        DataFactory.simple_extrusion(),
        DataFactory.double_extrusion(),
        DataFactory.triple_extrusion(),
        DataFactory.stairs(),
        DataFactory.ceiling(),
        DataFactory.double_tower_ceiling(),
        DataFactory.hanging_points(),
        DataFactory.checkerboard(),
        DataFactory.hole(),
        DataFactory.kill_you(),
        DataFactory.random(size=8, seed=0),
        # SurfaceNets makes this have negative volume until inverted
        DataFactory.random(size=3, seed=55),
    ],
)
def test_to_mesh_surface_nets(data: NDArray) -> None:
    """Test MeshGenerator.to_mesh_surface_nets."""
    mesh = MeshGenerator.to_mesh_surface_nets(data)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    assert mesh.volume > 0


@pytest.mark.parametrize(
    "data",
    [
        DataFactory.simple_extrusion(),
        DataFactory.double_extrusion(),
        DataFactory.triple_extrusion(),
        DataFactory.stairs(),
        DataFactory.ceiling(),
        DataFactory.double_tower_ceiling(),
        DataFactory.hanging_points(),
        DataFactory.checkerboard(),
        DataFactory.hole(),
        DataFactory.kill_you(),
        DataFactory.random(size=8, seed=0),
    ],
)
def test_to_mesh_dual_contouring(data: NDArray) -> None:
    """Test MeshGenerator.to_mesh_dual_contouring."""
    mesh = MeshGenerator.to_mesh_dual_contouring(data)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    assert mesh.volume > 0
