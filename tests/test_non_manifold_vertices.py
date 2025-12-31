"""Test the non-manifold vertices repair algorithm."""

import numpy as np
import pytest
from numpy.typing import NDArray

from edge_mender.data_factory import DataFactory
from edge_mender.edge_mender import EdgeMender
from edge_mender.mesh_generator import MeshGenerator
from edge_mender.non_manifold_vertices import find_num_non_manifold_vertices


def test_has_non_manifold_vertices_basic() -> None:
    """Test that the repair algorithm finds non-manifold vertices."""
    data = np.zeros((4, 4, 4))
    data[1, 1, 1] = 1
    data[2, 2, 2] = 1
    mesh = MeshGenerator.to_mesh_surface_nets(data)
    mender = EdgeMender(mesh)

    assert (
        find_num_non_manifold_vertices(mender.mesh.faces, mender.mesh.vertex_faces) == 1
    )


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (DataFactory.simple_extrusion(), 0),
        (DataFactory.double_extrusion(), 0),
        (DataFactory.triple_extrusion(), 0),
        (DataFactory.stairs(), 0),
        (DataFactory.ceiling(), 0),
        (DataFactory.double_tower_ceiling(), 0),
        (DataFactory.hanging_points(), 0),
        (DataFactory.checkerboard(), 1),
        # NOTE: This test case fails due to a bug with SurfaceNets from VTK
        # https://gitlab.kitware.com/vtk/vtk/-/issues/19156, fixed, but not released yet
        # (DataFactory.hole(), False),  # noqa: ERA001
        (DataFactory.kill_you(), 1),
    ],
)
def test_has_non_manifold_vertices(data: NDArray, expected: int) -> None:
    """Test that the repair algorithm finds non-manifold vertices."""
    mesh = MeshGenerator.to_mesh_surface_nets(data)
    mender = EdgeMender(mesh)
    mender.repair()

    assert (
        find_num_non_manifold_vertices(mender.mesh.faces, mender.mesh.vertex_faces)
        == expected
    )


@pytest.mark.parametrize("shift_distance", [0.0, 0.1])
def test_repair_non_manifold_vertices_basic(shift_distance: float) -> None:
    """Test that the repair function works for a basic test case."""
    data = np.zeros((4, 4, 4))
    data[1, 1, 1] = 1
    data[2, 2, 2] = 1
    mesh = MeshGenerator.to_mesh_surface_nets(data)
    mender = EdgeMender(mesh)

    mender.repair_vertices(shift_distance=shift_distance)

    assert (
        find_num_non_manifold_vertices(mender.mesh.faces, mender.mesh.vertex_faces) == 0
    )


@pytest.mark.parametrize("shift_distance", [0.0, 0.1])
@pytest.mark.parametrize(
    "data",
    [
        DataFactory.simple_extrusion(),  # This has no non-manifold vertices
        DataFactory.double_extrusion(),  # This has no non-manifold vertices
        DataFactory.triple_extrusion(),  # This has no non-manifold vertices
        DataFactory.stairs(),  # This has no non-manifold vertices
        DataFactory.ceiling(),  # This has no non-manifold vertices
        DataFactory.double_tower_ceiling(),  # This has no non-manifold vertices
        DataFactory.hanging_points(),  # This has no non-manifold vertices
        DataFactory.checkerboard(),
        # NOTE: This test case fails due to a bug with SurfaceNets from VTK
        # https://gitlab.kitware.com/vtk/vtk/-/issues/19156, fixed, but not released yet
        # DataFactory.hole(),  # noqa: ERA001
        DataFactory.kill_you(),
    ],
)
def test_repair_non_manifold_vertices(data: NDArray, shift_distance: float) -> None:
    """Test that the repair function works for the test cases."""
    mesh = MeshGenerator.to_mesh_surface_nets(data)
    mender = EdgeMender(mesh)
    mender.repair()

    mender.repair_vertices(shift_distance=shift_distance)

    assert (
        find_num_non_manifold_vertices(mender.mesh.faces, mender.mesh.vertex_faces) == 0
    )


@pytest.mark.parametrize("shift_distance", [0.0, 0.1])
def test_repair_non_manifold_vertices_case_1(shift_distance: float) -> None:
    """Test that the repair function works for a basic test case."""
    data = np.zeros((4, 4, 4))
    data[1, 1, 1] = 1
    data[1, 1, 2] = 1
    data[2, 1, 2] = 1
    data[2, 2, 1] = 1
    mesh = MeshGenerator.to_mesh_surface_nets(data)
    mender = EdgeMender(mesh)
    mender.repair()

    mender.repair_vertices(shift_distance=shift_distance)

    assert (
        find_num_non_manifold_vertices(mender.mesh.faces, mender.mesh.vertex_faces) == 0
    )
