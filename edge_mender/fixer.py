import numpy as np
import trimesh

from edge_mender.helper import (
    any_normals_point_towards_edge_direction,
    face_centers,
    faces_at_edges,
    faces_at_vertex,
    find_non_manifold_edges,
    fix_face,
    get_new_edges,
    is_left,
    rays_intersect,
    split_edge,
    split_face,
    split_point,
)


def get_split_rays_via_grouping(
    mesh: trimesh.Trimesh,
    edge_direction: np.ndarray,
    edge_face_indices: np.ndarray,
    edge_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    i = ~edge_direction.astype(bool)

    # Get two orthagonal directions to the edge direction
    line_direction_1 = [1, 1, 1] - np.abs(edge_direction)
    line_direction_2 = line_direction_1.copy()
    line_direction_2[np.argmax(line_direction_2)] = -1
    print(f"Line directions: {line_direction_1} and {line_direction_2}")

    edge_face_centers = face_centers(mesh, edge_face_indices)
    print(f"Edge face centers: {edge_face_centers}")

    group_1_centers = []
    group_2_centers = []
    group_1_faces = []
    group_2_faces = []
    for center, face_index in zip(edge_face_centers, edge_face_indices):
        if is_left(edge_points[0][i], line_direction_1[i], center[i]):
            group_1_centers.append(center)
            group_1_faces.append(face_index)
        else:
            group_2_centers.append(center)
            group_2_faces.append(face_index)
    print(
        f"Group 1 has faces {group_1_faces} with centers {group_1_centers}, Group 2 has faces {group_2_faces} with centers {group_2_centers}"
    )

    group_1_normals = mesh.face_normals[np.array(group_1_faces)]
    group_2_normals = mesh.face_normals[np.array(group_2_faces)]
    print(
        f"Group 1 has normals {list(group_1_normals)}, Group 2 has normals {list(group_2_normals)}"
    )

    groups_intersect = rays_intersect(
        point_1=group_1_centers[0][i],
        normal_1=group_1_normals[0][i],
        point_2=group_1_centers[1][i],
        normal_2=group_1_normals[1][i],
    )
    print(f"Groups intersect? {groups_intersect}")
    print(f"Groups correct? {not groups_intersect}")

    ray_1 = line_direction_1 if groups_intersect else line_direction_2
    ray_2 = ray_1 * -1
    print(f"Ray directions: {ray_1} and {ray_2}")

    return ray_1, ray_2


def fix(
    mesh: trimesh.Trimesh,
    *,
    move_distance: float = 0.0,  # should be less than 25% of the voxel size
    skip_edges: list[int] | None = None,
) -> tuple[list[int], list[int], list[list[int]]]:
    non_manifold_faces, non_manifold_vertices, non_manifold_edges = (
        find_non_manifold_edges(mesh)
    )
    print(f"Found {len(non_manifold_edges)} non-manifold edges\n")

    # Track vertices to move and the amount + direction to move them
    move_vertices: dict[int, np.ndarray] = {}
    # Track split vertices to avoid double processing
    split_vertices = set()

    new_faces = []
    new_vertices = []

    for original_edge_faces, edge_vertex_indices, edge in zip(
        non_manifold_faces, non_manifold_vertices, non_manifold_edges, strict=True
    ):
        if skip_edges and edge in skip_edges:
            print(f"Skipping edge {edge} as requested")
            continue
        print(f"Processing edge {edge}")

        edge_vertex_indices: np.ndarray
        points = mesh.vertices[edge_vertex_indices]
        print(
            f"Edge {edge} connects vertices {edge_vertex_indices} at {points[0]} and {points[1]}"
        )

        current_edge_faces = faces_at_edges(mesh, edge_vertex_indices)
        print(f"Edge {edge} was shared by faces {original_edge_faces}")
        print(f"Edge {edge} is now shared by faces {current_edge_faces}")

        for point, edge_vertex_index in zip(points, edge_vertex_indices):
            print(f"Processing vertex {edge_vertex_index} at {point}")
            if edge_vertex_index in split_vertices:
                print(f"Skipping already handled vertex {edge_vertex_index} at {point}")
                continue

            # Get the edge direction
            edge_direction = (
                points[1] - points[0]
                if edge_vertex_index == edge_vertex_indices[0]
                else points[0] - points[1]
            )
            print(f"Edge direction: {edge_direction}")

            # No floor
            if not any_normals_point_towards_edge_direction(
                mesh, edge_vertex_index, point, edge_direction
            ):
                print("No floor detected")

                other_edge_vertex_index = (
                    edge_vertex_indices[1]
                    if edge_vertex_index == edge_vertex_indices[0]
                    else edge_vertex_indices[0]
                )
                other_point = points[1] if point is points[0] else points[0]
                print(f"Splitting vertex {edge_vertex_index} at {point}")
                print(f"Other vertex {other_edge_vertex_index} at {other_point}")

                ray_1, ray_2 = get_split_rays_via_grouping(
                    mesh, edge_direction, original_edge_faces, points
                )

                new_point, new_vertex = split_point(mesh, point, edge_vertex_index)
                new_vertices.append(edge_vertex_index)
                new_vertices.append(new_vertex)
                if move_distance:
                    move_vertices[edge_vertex_index] = point + (ray_2 * move_distance)
                    move_vertices[new_vertex] = new_point + (ray_1 * move_distance)

                faces_to_reconnect = faces_at_vertex(mesh, edge_vertex_index)
                faces_to_reconnect_centers = face_centers(mesh, faces_to_reconnect)
                for face_index, face_center in zip(
                    faces_to_reconnect, faces_to_reconnect_centers
                ):
                    fix_face(
                        mesh,
                        face_index,
                        face_center,
                        edge_vertex_index,
                        new_point,
                        new_vertex,
                        ray_1,
                        ray_2,
                    )
                    new_faces.append(face_index)

                split_vertices.add(edge_vertex_index)
            else:
                print("Floor detected, skipping vertex split")

        # Floor and ceiling case
        if (
            edge_vertex_indices[0] not in split_vertices
            and edge_vertex_indices[1] not in split_vertices
        ):
            print("No vertices split, floor and ceiling case")

            ray_1, ray_2 = get_split_rays_via_grouping(
                mesh, edge_direction, original_edge_faces, points
            )

            new_point_0, new_vertex_0, new_point_1, new_vertex_1 = split_edge(
                mesh, points
            )
            new_vertices.append(new_vertex_0)
            new_vertices.append(new_vertex_1)
            if move_distance:
                move_vertices[new_vertex_0] = new_point_0 + (ray_1 * move_distance)
                move_vertices[new_vertex_1] = new_point_1 + (ray_2 * move_distance)

            faces_to_reconnect = np.array(list(current_edge_faces))
            faces_to_reconnect_centers = face_centers(mesh, faces_to_reconnect)
            for face_index, face_center in zip(
                faces_to_reconnect, faces_to_reconnect_centers
            ):
                new_face_index = split_face(
                    mesh,
                    edge_vertex_indices,
                    face_index,
                    face_center,
                    new_point_0,
                    new_vertex_0,
                    new_vertex_1,
                    ray_1,
                    ray_2,
                )
                new_faces.append(face_index)
                new_faces.append(new_face_index)

        print()

    # Move after splitting since movements will make normals that aren't orthogonal
    for vertex_to_move, new_point in move_vertices.items():
        print(
            f"Moving vertex {vertex_to_move} from {mesh.vertices[vertex_to_move]} to {new_point}"
        )
        mesh.vertices[vertex_to_move] = new_point

    return (
        new_faces,
        new_vertices,
        get_new_edges(mesh, non_manifold_vertices, new_vertices).tolist(),
    )
