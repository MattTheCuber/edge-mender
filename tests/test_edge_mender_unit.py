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
    e = np.array(expected_faces)
    e.sort()
    np.testing.assert_array_equal(faces, e)


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
    e = np.array(expected_faces)
    e.sort()
    np.testing.assert_array_equal(faces, e)


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

    centers = edge_mender._get_face_centers(np.array(faces))

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
            [22, 25, 18, 41],  # See test_get_faces_at_edge
            [12, 15],
            [1, 0, 1],
            [-1, 0, -1],
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.ceiling()),
            [0, 1, 0],
            [22, 24, 27, 55],  # See test_get_faces_at_edge
            [16, 19],
            [1, 0, 1],
            [-1, 0, -1],
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.checkerboard()),
            [0, 1, 0],
            [34, 38, 41, 79],  # See test_get_faces_at_edge
            [19, 22],
            [-1, 0, 1],
            [1, 0, -1],
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.checkerboard()),
            [0, 0, 1],
            [26, 28, 33, 39],  # See test_get_faces_at_edge
            [7, 19],
            [1, 1, 0],
            [-1, -1, 0],
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.hanging_points()),
            [0, 0, 1],
            [38, 40, 45, 59],  # See test_get_faces_at_edge
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

    assert len(mesh.vertices) == before_vertex_count + 1
    assert mesh.vertices[vertex_to_split].tolist() == before_point.tolist()
    assert new_vertex == before_vertex_count
    assert new_vertex != vertex_to_split
    assert new_point.tolist() == before_point.tolist()


@pytest.mark.parametrize(
    ("mesh", "vertex_to_split", "ray_1", "ray_2", "face_indices"),
    [
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.simple_extrusion()),
            15,
            [1, 0, 1],  # See test_get_split_direction_rays
            [-1, 0, -1],  # See test_get_split_direction_rays
            [22, 25, 18, 41],  # See test_get_faces_at_edge
        ),
    ],
)
def test_reassign_face(
    mesh: trimesh.Trimesh,
    vertex_to_split: int,
    ray_1: list[int],
    ray_2: list[int],
    face_indices: list[int],
) -> None:
    """Test that the split_face function properly creates and updates the faces."""
    edge_mender = EdgeMender(mesh)
    new_point, new_vertex = edge_mender._split_point(
        mesh.vertices[vertex_to_split],
        vertex_to_split,
    )

    face_centers = edge_mender._get_face_centers(np.array(face_indices))
    for i, face_index in enumerate(face_indices):
        edge_mender._reassign_face(
            face_index,
            face_centers[i],
            vertex_to_split,
            new_point,
            new_vertex,
            np.array(ray_1),
            np.array(ray_2),
        )

    assert np.any(mesh.faces[face_indices] == vertex_to_split, axis=1).sum() == 2  # noqa: PLR2004
    assert np.any(mesh.faces[face_indices] == new_vertex, axis=1).sum() == 2  # noqa: PLR2004


@pytest.mark.parametrize(
    ("mesh", "vertex_to_split", "ray_1", "ray_2", "face_indices"),
    [
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.simple_extrusion()),
            15,
            [1, 0, 1],  # See test_get_split_direction_rays
            [-1, 0, -1],  # See test_get_split_direction_rays
            [22, 25, 18, 41],  # See test_get_faces_at_edge
        ),
    ],
)
def test_reassign_face_fail(
    mesh: trimesh.Trimesh,
    vertex_to_split: int,
    ray_1: list[int],
    ray_2: list[int],
    face_indices: list[int],
) -> None:
    """Test that the split_face function properly creates and updates the faces."""
    edge_mender = EdgeMender(mesh)
    new_point, new_vertex = edge_mender._split_point(
        mesh.vertices[vertex_to_split],
        vertex_to_split,
    )

    for face_index in face_indices:
        with pytest.raises(ValueError, match="Angles are the same"):
            edge_mender._reassign_face(
                face_index,
                new_point,  # This will cause both angles to be the same
                vertex_to_split,
                new_point,
                new_vertex,
                np.array(ray_1),
                np.array(ray_2),
            )


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

    assert len(mesh.vertices) == before_vertex_count + 2
    assert new_vertex_0 != new_vertex_1
    assert new_vertex_0 == before_vertex_count
    assert new_vertex_1 == before_vertex_count + 1
    assert new_point_0.tolist() == expected_point
    assert new_point_1.tolist() == expected_point


@pytest.mark.parametrize(
    ("mesh", "edge_vertices_to_split", "ray_1", "ray_2", "face_index"),
    [
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.ceiling()),
            [16, 19],
            [1, 0, 1],  # See test_get_split_direction_rays
            [-1, 0, -1],  # See test_get_split_direction_rays
            22,  # See test_get_faces_at_edge
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.ceiling()),
            [16, 19],
            [1, 0, 1],  # See test_get_split_direction_rays
            [-1, 0, -1],  # See test_get_split_direction_rays
            24,  # See test_get_faces_at_edge
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.ceiling()),
            [16, 19],
            [1, 0, 1],  # See test_get_split_direction_rays
            [-1, 0, -1],  # See test_get_split_direction_rays
            27,  # See test_get_faces_at_edge
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.ceiling()),
            [16, 19],
            [1, 0, 1],  # See test_get_split_direction_rays
            [-1, 0, -1],  # See test_get_split_direction_rays
            55,  # See test_get_faces_at_edge
        ),
    ],
)
def test_split_face(
    mesh: trimesh.Trimesh,
    edge_vertices_to_split: list[int],
    ray_1: list[int],
    ray_2: list[int],
    face_index: int,
) -> None:
    """Test that the split_face function properly creates and updates the faces."""
    edge_mender = EdgeMender(mesh)
    new_point_0, new_vertex_0, _, new_vertex_1 = edge_mender._split_edge(
        mesh.vertices[edge_vertices_to_split],
    )
    before_face_indices = set(mesh.faces[face_index].copy().tolist())
    before_face_count = len(mesh.faces)

    new_face_index = edge_mender._split_face(
        np.array(edge_vertices_to_split),
        face_index,
        edge_mender._get_face_centers(np.array([face_index]))[0],
        new_point_0,
        new_vertex_0,
        new_vertex_1,
        np.array(ray_1),
        np.array(ray_2),
    )

    assert len(mesh.faces) == before_face_count + 1
    assert new_face_index == before_face_count
    assert new_face_index != face_index
    all_new_vertices = np.concatenate(
        [mesh.faces[face_index], mesh.faces[new_face_index]],
    )
    assert before_face_indices.issubset(set(all_new_vertices.tolist()))
    selected_vertex = (
        new_vertex_0 if new_vertex_0 in mesh.faces[face_index] else new_vertex_1
    )
    assert selected_vertex in mesh.faces[face_index]
    assert selected_vertex in mesh.faces[new_face_index]


@pytest.mark.parametrize(
    ("mesh", "edge_vertices_to_split", "ray_1", "ray_2", "face_index"),
    [
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.ceiling()),
            [16, 19],
            [1, 0, 1],  # See test_get_split_direction_rays
            [-1, 0, -1],  # See test_get_split_direction_rays
            22,  # See test_get_faces_at_edge
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.ceiling()),
            [16, 19],
            [1, 0, 1],  # See test_get_split_direction_rays
            [-1, 0, -1],  # See test_get_split_direction_rays
            24,  # See test_get_faces_at_edge
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.ceiling()),
            [16, 19],
            [1, 0, 1],  # See test_get_split_direction_rays
            [-1, 0, -1],  # See test_get_split_direction_rays
            27,  # See test_get_faces_at_edge
        ),
        (
            MeshGenerator.to_mesh_surface_nets(DataFactory.ceiling()),
            [16, 19],
            [1, 0, 1],  # See test_get_split_direction_rays
            [-1, 0, -1],  # See test_get_split_direction_rays
            55,  # See test_get_faces_at_edge
        ),
    ],
)
def test_split_face_fail(
    mesh: trimesh.Trimesh,
    edge_vertices_to_split: list[int],
    ray_1: list[int],
    ray_2: list[int],
    face_index: int,
) -> None:
    """Test that the split_face function properly creates and updates the faces."""
    edge_mender = EdgeMender(mesh)
    new_point_0, new_vertex_0, _, new_vertex_1 = edge_mender._split_edge(
        mesh.vertices[edge_vertices_to_split],
    )

    with pytest.raises(ValueError, match="Angles are the same"):
        edge_mender._split_face(
            np.array(edge_vertices_to_split),
            face_index,
            new_point_0,  # This will cause both angles to be the same
            new_point_0,
            new_vertex_0,
            new_vertex_1,
            np.array(ray_1),
            np.array(ray_2),
        )
