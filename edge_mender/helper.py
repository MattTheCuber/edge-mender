import fill_voids
import numpy as np
import trimesh


def has_holes(data: np.ndarray) -> bool:
    return not np.array_equal(data, fill_voids.fill(data, in_place=False))


def find_all_non_manifold_edges(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    unique_edges, counts = np.unique(
        mesh.faces_unique_edges.flatten(), return_counts=True
    )
    edges = unique_edges[counts != 2]
    vertices = mesh.edges_unique[edges]
    return vertices, edges


def find_non_manifold_edges(
    mesh: trimesh.Trimesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_edges, counts = np.unique(
        mesh.faces_unique_edges.flatten(), return_counts=True
    )
    edges = unique_edges[counts == 4]
    vertices = mesh.edges_unique[edges]

    distance_check, edge_index = mesh.edges_sorted_tree.query(vertices, k=4)
    if np.any(distance_check):
        raise ValueError("Problem with edge face lookup")
    faces = edge_index // 3

    return faces, vertices, edges


def faces_at_edges(mesh: trimesh.Trimesh, edge_vertices: np.ndarray) -> np.ndarray:
    distance_check, edge_index = mesh.edges_sorted_tree.query(edge_vertices, k=4)
    return edge_index[distance_check == 0] // 3


def faces_at_vertex(mesh: trimesh.Trimesh, vertex: int) -> np.ndarray:
    # Get connected faces
    faces = mesh.vertex_faces[vertex]
    faces = faces[faces != -1]  # remove padding (-1 entries)
    return faces


def face_centers(mesh: trimesh.Trimesh, face_indices: np.ndarray) -> np.ndarray:
    return np.array(np.mean(mesh.vertices[mesh.faces[face_indices]], axis=1))


def point_in_line(
    test_point: np.ndarray, line_point: np.ndarray, direction: np.ndarray
) -> bool:
    diff = test_point - line_point

    # If the direction vector is zero, line is undefined
    if np.allclose(direction, 0):
        return np.allclose(test_point, line_point)

    # Handle division by zero by checking ratios only where direction ≠ 0
    t_values = []
    for i in range(3):
        if abs(direction[i]) > 1e-9:
            t_values.append(diff[i] / direction[i])
        elif abs(diff[i]) > 1e-9:
            # If direction is zero but diff isn't, point can't lie on line
            return False

    # All non-zero ratios must be the same
    return np.allclose(t_values, t_values[0])


def rays_intersect(
    point_1: np.ndarray,
    normal_1: np.ndarray,
    point_2: np.ndarray,
    normal_2: np.ndarray,
    tol: float = 1e-9,
) -> bool:
    A = np.array([[normal_1[0], -normal_2[0]], [normal_1[1], -normal_2[1]]])
    b = point_2 - point_1

    det = np.linalg.det(A)
    if abs(det) < tol:
        if np.linalg.norm(np.cross(np.append(normal_1, 0), np.append(b, 0))) < tol:
            raise ValueError("Colinear")
        raise ValueError("Parallel")

    t, s = np.linalg.solve(A, b)
    if t >= -tol and s >= -tol:
        return True
    return False


def angle_between_point_and_ray(
    point: np.ndarray, ray_point: np.ndarray, ray_dir: np.ndarray
) -> float:
    # Vector from ray point to external point
    v = point - ray_point

    # Normalize direction and v
    d_norm = ray_dir / np.linalg.norm(ray_dir)
    v_norm = v / np.linalg.norm(v)

    # Clamp to handle floating point precision issues
    dot = np.clip(np.dot(d_norm, v_norm), -1.0, 1.0)

    return np.arccos(dot)


def get_direction(
    mesh: trimesh.Trimesh, edge: tuple[int, int], points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Find faces at the first vertex of the edge
    faces1 = faces_at_vertex(mesh, edge[0])
    print(
        f"Vertex {edge[0]} at {points[0]} is connected to {len(faces1)} faces: {faces1}"
    )

    # Find faces at the second vertex of the edge
    faces2 = faces_at_vertex(mesh, edge[1])
    print(
        f"Vertex {edge[1]} at {points[1]} is connected to {len(faces2)} faces: {faces2}"
    )

    # Find all faces connected to the edge
    all_faces = np.unique(np.concatenate([faces1, faces2]))
    print(f"Edge {edge} is connected to {len(all_faces)} faces: {all_faces}")

    # Find all normals for the connected faces
    all_normals = mesh.face_normals[all_faces]
    unique_normals = np.unique(all_normals, axis=0)
    direction = unique_normals.sum(axis=0)
    print(f"Edge {edge} has normals sum: {direction}")

    return direction, unique_normals


def any_normals_point_towards_edge_direction(
    mesh: trimesh.Trimesh,
    edge_vertex_index: int,
    point: np.ndarray,
    edge_direction: np.ndarray,
) -> bool:
    # Find faces at the vertex
    faces = faces_at_vertex(mesh, edge_vertex_index)
    print(
        f"Vertex {edge_vertex_index} at {point} is connected to {len(faces)} faces: {faces}"
    )

    # Find all normals for the connected faces
    all_normals = mesh.face_normals[faces]
    print(f"Vertex {edge_vertex_index} faces have the following normals: {all_normals}")
    colinear_normals = all_normals[np.all(all_normals == edge_direction, axis=1)]
    unique_normals = np.unique(colinear_normals, axis=0)
    print(
        f"Vertex {edge_vertex_index} faces have the following unique colinear normals: {unique_normals}"
    )

    # Check if the normals point towards the edge direction
    dot = np.dot(unique_normals, edge_direction)
    return np.any(dot == 1).item()


def is_left(
    line_point: np.ndarray,
    direction: np.ndarray,
    test_point: np.ndarray,
) -> bool:
    print(
        f"Testing if point {test_point} is left of line at {line_point} with direction {direction}"
    )
    vx, vy = test_point[0] - line_point[0], test_point[1] - line_point[1]
    cross = direction[0] * vy - direction[1] * vx
    if cross == 0:
        raise ValueError("Point is on the line")
    return cross > 0


def split_point(
    mesh: trimesh.Trimesh,
    point_to_move: np.ndarray,
    vertex_to_move: int,
) -> tuple[np.ndarray, int]:
    new_point = point_to_move.copy()
    mesh.vertices = np.vstack([mesh.vertices, new_point])
    split_vertex_index_2 = mesh.vertices.shape[0] - 1
    print(
        f"Split points: {vertex_to_move} at {point_to_move} and {split_vertex_index_2} at {new_point}"
    )

    return new_point, split_vertex_index_2


def split_edge(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
) -> tuple[np.ndarray, int, np.ndarray, int]:
    new_point_0 = np.mean(points, axis=0)
    new_point_1 = np.mean(points, axis=0)
    mesh.vertices = np.vstack([mesh.vertices, new_point_0, new_point_1])
    new_vertex_index_0 = mesh.vertices.shape[0] - 2
    new_vertex_index_1 = mesh.vertices.shape[0] - 1
    print(
        f"Split edge producing 2 points: {new_vertex_index_0} at {new_point_0} and {new_vertex_index_1} at {new_point_1}"
    )

    return new_point_0, new_vertex_index_0, new_point_1, new_vertex_index_1


def fix_face(
    mesh: trimesh.Trimesh,
    face_index: np.ndarray,
    face_center: np.ndarray,
    vertex_to_move: int,
    new_point: np.ndarray,
    new_vertex: int,
    ray_1: np.ndarray,
    ray_2: np.ndarray,
):
    face_points = mesh.faces[face_index]
    angle_1 = angle_between_point_and_ray(face_center, new_point, ray_1)
    angle_2 = angle_between_point_and_ray(face_center, new_point, ray_2)

    if angle_1 < angle_2:
        face_points[face_points == vertex_to_move] = new_vertex
    elif angle_1 > angle_2:
        pass  # No change
    else:
        raise ValueError(
            "Angles are the same, this is impossible. Are any of your face normals inverted?"
        )


def split_face(
    mesh: trimesh.Trimesh,
    edge: np.ndarray,
    face_index: np.ndarray,
    face_center: np.ndarray,
    new_point: np.ndarray,
    new_vertex_0: int,
    new_vertex_1: int,
    ray_1: np.ndarray,
    ray_2: np.ndarray,
) -> int:
    face_points = mesh.faces[face_index]
    angle_1 = angle_between_point_and_ray(face_center, new_point, ray_1)
    angle_2 = angle_between_point_and_ray(face_center, new_point, ray_2)

    new_face_points = face_points.copy()
    if angle_1 < angle_2:
        face_points[face_points == edge[0]] = new_vertex_0
        new_face_points[face_points == edge[1]] = new_vertex_0
    elif angle_1 > angle_2:
        face_points[face_points == edge[0]] = new_vertex_1
        new_face_points[face_points == edge[1]] = new_vertex_1
    else:
        raise ValueError("Angles are the same, this is impossible. Something broke!")

    mesh.faces = np.vstack([mesh.faces, new_face_points])

    return mesh.faces.shape[0] - 1


def get_new_edges(
    mesh: trimesh.Trimesh, non_manifold_vertices: np.ndarray, new_vertices: list[int]
) -> np.ndarray:
    check = np.hstack([non_manifold_vertices.flatten(), new_vertices])
    mask = np.isin(mesh.edges_sorted[:, 0], check) & np.isin(
        mesh.edges_sorted[:, 1], check
    )
    return mesh.edges_sorted[mask]
