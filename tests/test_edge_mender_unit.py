"""Test private functions in the EdgeMender class."""

import numpy as np
import pytest
import trimesh

from edge_mender.data_factory import DataFactory
from edge_mender.edge_mender import EdgeMender
from edge_mender.mesh_generator import MeshGenerator


def test_edge_mender_init() -> None:
    """Test that the EdgeMender class can be initialized."""
    mesh = trimesh.creation.box()
    EdgeMender(mesh)
    EdgeMender(mesh, debug=True)


@pytest.mark.parametrize(
    ("mesh", "edge_vertices", "expected_faces"),
    [
        (trimesh.creation.box(), [0, 1], [0, 1]),
        (trimesh.creation.box(), [0, 2], [2, 3]),
        (trimesh.creation.box(), [1, 3], [0, 4]),
        (trimesh.creation.box(), [2, 3], [2, 7]),
        (trimesh.creation.box(), [0, 4], [1, 3]),
        (trimesh.creation.box(), [1, 5], [5, 6]),
        (trimesh.creation.box(), [3, 7], [4, 7]),
        (trimesh.creation.box(), [2, 6], [8, 9]),
        (trimesh.creation.box(), [4, 5], [5, 10]),
        (trimesh.creation.box(), [5, 7], [6, 11]),
        (trimesh.creation.box(), [6, 7], [9, 11]),
        (trimesh.creation.box(), [4, 6], [8, 10]),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.simple_extrusion()),
            [12, 15],
            [22, 25, 18, 41],
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.double_extrusion()),
            [14, 17],
            [22, 25, 20, 51],
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.double_extrusion()),
            [17, 20],
            [28, 57, 32, 35],
        ),
    ],
)
def test_get_faces_at_edge(
    mesh: trimesh.Trimesh,
    edge_vertices: list[int],
    expected_faces: list[int],
) -> None:
    """Test that the get_faces_at_edge function returns the correct faces."""
    edge_mender = EdgeMender(mesh)
    faces = edge_mender._get_faces_at_edge(np.array(edge_vertices))
    faces.sort()
    expected_faces = np.array(expected_faces)
    expected_faces.sort()
    np.testing.assert_array_equal(faces, expected_faces)


@pytest.mark.parametrize(
    ("mesh", "vertex", "expected_faces"),
    [
        (trimesh.creation.box(), 0, [0, 1, 2, 3]),
        (trimesh.creation.box(), 1, [0, 1, 4, 5, 6]),
        (trimesh.creation.box(), 2, [2, 3, 7, 8, 9]),
        (trimesh.creation.box(), 3, [0, 2, 4, 7]),
        (trimesh.creation.box(), 4, [1, 3, 5, 8, 10]),
        (trimesh.creation.box(), 5, [5, 6, 10, 11]),
        (trimesh.creation.box(), 6, [8, 9, 10, 11]),
        (trimesh.creation.box(), 7, [4, 6, 7, 9, 11]),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.simple_extrusion()),
            12,
            [12, 13, 18, 19, 22, 23, 25, 34, 35, 41],
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.simple_extrusion()),
            15,
            [18, 20, 22, 24, 25, 40, 41, 45],
        ),
    ],
)
def test_get_faces_at_vertex(
    mesh: trimesh.Trimesh,
    vertex: int,
    expected_faces: list[int],
) -> None:
    """Test that the get_faces_at_vertex function returns the correct faces."""
    edge_mender = EdgeMender(mesh)
    # Cache the vertex faces property
    edge_mender._vertex_faces = mesh.vertex_faces
    faces = edge_mender._get_faces_at_vertex(vertex)
    faces.sort()
    expected_faces = np.array(expected_faces)
    expected_faces.sort()
    np.testing.assert_array_equal(faces, expected_faces)


@pytest.mark.parametrize(
    ("mesh", "faces", "expected_centers"),
    [
        (
            trimesh.creation.box(),
            [0],
            [[-0.5, -0.5 / 3, 0.5 / 3]],
        ),
        (
            trimesh.creation.box(),
            [1, 3],
            [[-0.5 / 3, -0.5, -0.5 / 3], [-0.5 / 3, -0.5 / 3, -0.5]],
        ),
    ],
)
def test_get_face_centers(
    mesh: trimesh.Trimesh,
    faces: list[int],
    expected_centers: list[list[int]],
) -> None:
    """Test that the get_face_centers function returns the correct centers."""
    edge_mender = EdgeMender(mesh)
    centers = edge_mender._get_face_centers(faces)
    np.testing.assert_array_equal(centers, expected_centers)


@pytest.mark.parametrize(
    ("mesh", "edge_vertex_index", "edge_direction", "expected"),
    [
        (trimesh.creation.box(), 0, [1, 0, 0], False),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.simple_extrusion()),
            12,
            [0, 1, 0],
            True,
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.simple_extrusion()),
            15,
            [0, -1, 0],
            False,
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.ceiling()),
            16,
            [0, 1, 0],
            True,
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.ceiling()),
            19,
            [0, -1, 0],
            True,
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.checkerboard()),
            19,
            [0, 1, 0],
            True,
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.checkerboard()),
            22,
            [0, -1, 0],
            True,
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.checkerboard()),
            19,
            [0, 0, -1],
            True,
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.checkerboard()),
            7,
            [0, 0, 1],
            False,
        ),
    ],
)
def test_has_normals_matching_edge(
    mesh: trimesh.Trimesh,
    edge_vertex_index: int,
    edge_direction: list[int],
    *,
    expected: bool,
) -> None:
    """Test that the has_normals_matching_edge function returns the correct result."""
    edge_mender = EdgeMender(mesh)
    # Cache the vertex faces property
    edge_mender._vertex_faces = mesh.vertex_faces
    # Cache face normals
    edge_mender._face_normals = mesh.face_normals
    assert (
        edge_mender._has_normals_matching_edge(
            edge_vertex_index,
            mesh.vertices[edge_vertex_index],
            np.array(edge_direction),
        )
        == expected
    )


@pytest.mark.parametrize(
    (
        "mesh",
        "edge_direction",
        "edge_face_indices",
        "edge_vertices",
        "expected_ray_1",
        "expected_ray_2",
    ),
    [
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.simple_extrusion()),
            [0, 1, 0],
            [22, 25, 18, 41],
            [12, 15],
            [1, 0, 1],
            [-1, 0, -1],
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.ceiling()),
            [0, 1, 0],
            [22, 24, 27, 55],
            [16, 19],
            [1, 0, 1],
            [-1, 0, -1],
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.checkerboard()),
            [0, 1, 0],
            [34, 38, 41, 79],
            [19, 22],
            [-1, 0, 1],
            [1, 0, -1],
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.checkerboard()),
            [0, 0, 1],
            [26, 28, 33, 39],
            [7, 19],
            [1, 1, 0],
            [-1, -1, 0],
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.hanging_points()),
            [0, 0, 1],
            [38, 40, 45, 59],
            [10, 25],
            [-1, 1, 0],
            [1, -1, 0],
        ),
    ],
)
def test_get_split_direction_rays(
    mesh: trimesh.Trimesh,
    edge_direction: list[int],
    edge_face_indices: list[int],
    edge_vertices: list[int],
    expected_ray_1: list[int],
    expected_ray_2: list[int],
) -> None:
    """Test that the get_split_direction_rays function returns the correct rays."""
    edge_mender = EdgeMender(mesh)
    # Cache face normals
    edge_mender._face_normals = mesh.face_normals
    ray_1, ray_2 = edge_mender._get_split_direction_rays(
        np.array(edge_direction),
        np.array(edge_face_indices),
        mesh.vertices[edge_vertices],
    )
    assert ray_1.tolist() == expected_ray_1
    assert ray_2.tolist() == expected_ray_2


@pytest.mark.parametrize(
    ("mesh", "vertex_to_split"),
    [
        (trimesh.creation.box(), 0),
        (MeshGenerator.to_mesh_surface_nets(DataFactory.simple_extrusion()), 15),
    ],
)
def test_split_point(mesh: trimesh.Trimesh, vertex_to_split: int) -> None:
    """Test that the split_point function properly creates the new vertex."""
    edge_mender = EdgeMender(mesh)
    before_point = mesh.vertices[vertex_to_split].copy()
    before_vertex_count = len(mesh.vertices)
    new_point, new_vertex = edge_mender._split_point(before_point, vertex_to_split)
    assert mesh.vertices[vertex_to_split].tolist() == before_point.tolist()
    assert new_vertex == before_vertex_count
    assert new_point.tolist() == before_point.tolist()
    assert len(mesh.vertices) == before_vertex_count + 1


@pytest.mark.parametrize(
    ("mesh", "edge_vertices_to_split", "expected_point"),
    [
        (
            trimesh.creation.box(),
            [0, 1],
            [-0.5, -0.5, 0],
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.simple_extrusion()),
            [12, 15],
            [1.5, 2.0, 1.5],
        ),
    ],
)
def test_split_edge(
    mesh: trimesh.Trimesh,
    edge_vertices_to_split: list[int],
    expected_point: list[int],
) -> None:
    """Test that the split_edge function properly creates the new vertices."""
    edge_mender = EdgeMender(mesh)
    before_vertex_count = len(mesh.vertices)
    new_point_0, new_vertex_0, new_point_1, new_vertex_1 = edge_mender._split_edge(
        mesh.vertices[edge_vertices_to_split],
    )
    assert new_vertex_0 == before_vertex_count
    assert new_vertex_1 == before_vertex_count + 1
    assert new_point_0.tolist() == expected_point
    assert new_point_1.tolist() == expected_point
    assert len(mesh.vertices) == before_vertex_count + 2
