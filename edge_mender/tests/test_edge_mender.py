"""Test major functions in the EdgeMender class."""

import numpy as np
import pytest
import trimesh
from numpy.typing import NDArray

from edge_mender.data_factory import DataFactory
from edge_mender.edge_mender import EdgeMender
from edge_mender.mesh_generator import MeshGenerator


@pytest.mark.parametrize("spacing", [(1, 1, 1), (1.25, 0.5, 0.25)])
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
    ],
)
def test_validate(data: NDArray, spacing: tuple[float, float, float]) -> None:
    """Test that the validate function works for valid meshes."""
    mesh = MeshGenerator.to_mesh_surface_nets(data)
    mesh.vertices *= spacing
    mender = EdgeMender(mesh)
    mender.validate(spacing=spacing)


def test_validate_fail_normals() -> None:
    """Test that the validate function fails for non-axis-aligned face normals."""
    # Pyramid with non-axis-aligned face normals
    mesh = trimesh.creation.cone(1, 1, sections=3)
    mender = EdgeMender(mesh)
    with pytest.raises(ValueError, match="non-axis-aligned face normals"):
        mender.validate(spacing=(1, 1, 1))


def test_validate_fail_angles() -> None:
    """Test that the validate function fails for non-standard degree angles."""
    mesh = trimesh.creation.box()
    # Stretch the box to create bad angles
    mesh.vertices *= [1, 1, 1.25]
    mender = EdgeMender(mesh)
    with pytest.raises(ValueError, match="bad angles"):
        mender.validate(spacing=(1, 1, 1))


def test_validate_fail_areas() -> None:
    """Test that the validate function fails for non-uniform face areas."""
    # Subdivide everything except one face to make the faces larger
    mesh = trimesh.creation.box().subdivide(list(range(10)))
    mender = EdgeMender(mesh)
    with pytest.raises(ValueError, match="non-uniform face areas"):
        mender.validate(spacing=(1, 1, 1))


@pytest.mark.parametrize(
    ("data", "expected_faces", "expected_vertices", "expected_edges"),
    [
        (
            np.array(
                [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                ],
            ),
            np.empty((0, 4)),
            np.empty((0, 2)),
            [],
        ),
        (
            DataFactory.simple_extrusion(),
            [[25, 41, 18, 22]],
            [[12, 15]],
            [37],
        ),
        (
            DataFactory.double_extrusion(),
            [[51, 20, 22, 25], [35, 32, 57, 28]],
            [[14, 17], [17, 20]],
            [42, 52],
        ),
        (
            DataFactory.triple_extrusion(),
            [[27, 61, 24, 22], [35, 67, 32, 30], [73, 38, 45, 42]],
            [[16, 19], [19, 22], [22, 25]],
            [46, 57, 67],
        ),
        (
            DataFactory.stairs(),
            [[28, 30, 65, 33]],
            [[19, 22]],
            [55],
        ),
        (
            DataFactory.ceiling(),
            [[24, 27, 55, 22]],
            [[16, 19]],
            [46],
        ),
        (
            DataFactory.double_tower_ceiling(),
            [[26, 61, 29, 24], [37, 32, 34, 69]],
            [[18, 21], [21, 24]],
            [50, 61],
        ),
        (
            DataFactory.hanging_points(),
            [[38, 45, 40, 59]],
            [[10, 25]],
            [68],
        ),
        (
            DataFactory.checkerboard(),
            [
                [26, 28, 33, 39],
                [24, 67, 31, 28],
                [65, 35, 24, 26],
                [41, 32, 30, 71],
                [79, 38, 41, 34],
                [66, 71, 64, 79],
            ],
            [[7, 22], [19, 22], [21, 22], [22, 23], [22, 25], [22, 37]],
            [49, 52, 54, 58, 65, 105],
        ),
        (
            DataFactory.hole(),
            [
                [18, 51, 21, 16],
                [20, 57, 27, 24],
                [20, 22, 35, 55],
                [36, 34, 71, 43],
                [55, 67, 50, 48],
                [50, 46, 53, 95],
                [54, 56, 61, 73],
                [99, 52, 59, 56],
                [52, 69, 97, 54],
                [64, 79, 66, 71],
                [85, 70, 75, 72],
                [70, 113, 81, 68],
            ],
            [
                [13, 17],
                [14, 18],
                [17, 18],
                [21, 22],
                [17, 31],
                [27, 31],
                [18, 32],
                [28, 32],
                [31, 32],
                [21, 35],
                [22, 36],
                [35, 36],
            ],
            [40, 44, 46, 62, 83, 86, 89, 92, 94, 102, 108, 112],
        ),
    ],
)
def test_find_non_manifold_edges(
    data: NDArray,
    expected_faces: NDArray | list[list[int]],
    expected_vertices: NDArray | list[list[int]],
    expected_edges: NDArray | list[int],
) -> None:
    """Test that the validate function works for valid meshes."""
    mesh = MeshGenerator.to_mesh_surface_nets(data)
    faces, vertices, edges = EdgeMender(mesh).find_non_manifold_edges()
    np.testing.assert_array_equal(faces, expected_faces)
    np.testing.assert_array_equal(vertices, expected_vertices)
    np.testing.assert_array_equal(edges, expected_edges)
