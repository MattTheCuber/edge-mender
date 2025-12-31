"""Test the non-manifold vertices repair algorithm."""

import numpy as np
import pytest
from numpy.typing import NDArray

from edge_mender.data_factory import DataFactory
from edge_mender.edge_mender import EdgeMender
from edge_mender.mesh_generator import MeshGenerator
from edge_mender.non_manifold_vertices import (
    has_non_manifold_vertices,  # pyright: ignore[reportAttributeAccessIssue]
)


def test_has_non_manifold_vertices_basic() -> None:
    """Test that the repair algorithm finds non-manifold vertices."""
    data = np.zeros((4, 4, 4))
    data[1, 1, 1] = 1
    data[2, 2, 2] = 1
    mesh = MeshGenerator.to_mesh_surface_nets(data)
    mender = EdgeMender(mesh)

    assert has_non_manifold_vertices(mender.mesh.faces, mender.mesh.vertex_faces)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (DataFactory.simple_extrusion(), False),
        (DataFactory.double_extrusion(), False),
        (DataFactory.triple_extrusion(), False),
        (DataFactory.stairs(), False),
        (DataFactory.ceiling(), False),
        (DataFactory.double_tower_ceiling(), False),
        (DataFactory.hanging_points(), False),
        (DataFactory.checkerboard(), True),
        # NOTE: This test case fails due to a bug with SurfaceNets from VTK
        # https://gitlab.kitware.com/vtk/vtk/-/issues/19156, fixed, but not released yet
        # (DataFactory.hole(), False),  # noqa: ERA001
        (DataFactory.kill_you(), True),
    ],
)
def test_has_non_manifold_vertices(data: NDArray, *, expected: bool) -> None:
    """Test that the repair algorithm finds non-manifold vertices."""
    mesh = MeshGenerator.to_mesh_surface_nets(data)
    mender = EdgeMender(mesh)
    mender.repair()

    assert (
        has_non_manifold_vertices(mender.mesh.faces, mender.mesh.vertex_faces)
        is expected
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

    assert not has_non_manifold_vertices(mender.mesh.faces, mender.mesh.vertex_faces)


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

    assert not has_non_manifold_vertices(mender.mesh.faces, mender.mesh.vertex_faces)
