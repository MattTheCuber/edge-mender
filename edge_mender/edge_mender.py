"""Provides the class for repairing non-manifold edges in voxel boundary meshes."""

import numpy as np
import trimesh
from numpy.typing import NDArray

from edge_mender.geometry_helper import GeometryHelper


class EdgeMender:
    """The class for repairing non-manifold edges in voxel boundary meshes."""

    def __init__(self, mesh: trimesh.Trimesh) -> None:
        self.mesh = mesh

    def _get_split_rays_via_grouping(
        self,
        edge_direction: NDArray,
        edge_face_indices: NDArray,
        edge_points: NDArray,
    ) -> tuple[NDArray, NDArray]:
        i = ~edge_direction.astype(bool)

        # Get two orthagonal directions to the edge direction
        line_direction_1 = [1, 1, 1] - np.abs(edge_direction)
        line_direction_2 = line_direction_1.copy()
        line_direction_2[np.argmax(line_direction_2)] = -1
        print(f"Line directions: {line_direction_1} and {line_direction_2}")

        edge_face_centers = self._get_face_centers(edge_face_indices)
        print(f"Edge face centers: {edge_face_centers}")

        group_1_centers = []
        group_2_centers = []
        group_1_faces = []
        group_2_faces = []
        for center, face_index in zip(edge_face_centers, edge_face_indices):
            if GeometryHelper.is_left(
                edge_points[0][i], line_direction_1[i], center[i]
            ):
                group_1_centers.append(center)
                group_1_faces.append(face_index)
            else:
                group_2_centers.append(center)
                group_2_faces.append(face_index)
        print(
            f"Group 1 has faces {group_1_faces} with centers {group_1_centers}, Group 2 has faces {group_2_faces} with centers {group_2_centers}"
        )

        group_1_normals = self.mesh.face_normals[np.array(group_1_faces)]
        group_2_normals = self.mesh.face_normals[np.array(group_2_faces)]
        print(
            f"Group 1 has normals {list(group_1_normals)}, Group 2 has normals {list(group_2_normals)}"
        )

        groups_intersect = GeometryHelper.rays_intersect(
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

    def repair(
        self,
        *,
        move_distance: float = 0.0,  # should be less than 25% of the voxel size
        skip_edges: list[int] | None = None,
    ) -> tuple[list[int], list[int], list[list[int]]]:
        non_manifold_faces, non_manifold_vertices, non_manifold_edges = (
            self.find_non_manifold_edges()
        )
        print(f"Found {len(non_manifold_edges)} non-manifold edges\n")

        # Track vertices to move and the amount + direction to move them
        move_vertices: dict[int, NDArray] = {}
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

            edge_vertex_indices: NDArray
            points = self.mesh.vertices[edge_vertex_indices]
            print(
                f"Edge {edge} connects vertices {edge_vertex_indices} at {points[0]} and {points[1]}"
            )

            current_edge_faces = self._get_faces_at_edges(edge_vertex_indices)
            print(f"Edge {edge} was shared by faces {original_edge_faces}")
            print(f"Edge {edge} is now shared by faces {current_edge_faces}")

            for point, edge_vertex_index in zip(points, edge_vertex_indices):
                print(f"Processing vertex {edge_vertex_index} at {point}")
                if edge_vertex_index in split_vertices:
                    print(
                        f"Skipping already handled vertex {edge_vertex_index} at {point}"
                    )
                    continue

                # Get the edge direction
                edge_direction = (
                    points[1] - points[0]
                    if edge_vertex_index == edge_vertex_indices[0]
                    else points[0] - points[1]
                )
                print(f"Edge direction: {edge_direction}")

                # No floor
                if not self._has_normals_matching_edge(
                    edge_vertex_index, point, edge_direction
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

                    ray_1, ray_2 = self._get_split_rays_via_grouping(
                        edge_direction, original_edge_faces, points
                    )

                    new_point, new_vertex = self._split_point(point, edge_vertex_index)
                    new_vertices.append(edge_vertex_index)
                    new_vertices.append(new_vertex)
                    if move_distance:
                        move_vertices[edge_vertex_index] = point + (
                            ray_2 * move_distance
                        )
                        move_vertices[new_vertex] = new_point + (ray_1 * move_distance)

                    faces_to_reconnect = self._get_faces_at_vertex(edge_vertex_index)
                    faces_to_reconnect_centers = self._get_face_centers(
                        faces_to_reconnect
                    )
                    for face_index, face_center in zip(
                        faces_to_reconnect, faces_to_reconnect_centers
                    ):
                        self._reassign_face(
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

                ray_1, ray_2 = self._get_split_rays_via_grouping(
                    edge_direction, original_edge_faces, points
                )

                new_point_0, new_vertex_0, new_point_1, new_vertex_1 = self._split_edge(
                    points
                )
                new_vertices.append(new_vertex_0)
                new_vertices.append(new_vertex_1)
                if move_distance:
                    move_vertices[new_vertex_0] = new_point_0 + (ray_1 * move_distance)
                    move_vertices[new_vertex_1] = new_point_1 + (ray_2 * move_distance)

                faces_to_reconnect = np.array(list(current_edge_faces))
                faces_to_reconnect_centers = self._get_face_centers(faces_to_reconnect)
                for face_index, face_center in zip(
                    faces_to_reconnect, faces_to_reconnect_centers
                ):
                    new_face_index = self._split_face(
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
                f"Moving vertex {vertex_to_move} from {self.mesh.vertices[vertex_to_move]} to {new_point}"
            )
            self.mesh.vertices[vertex_to_move] = new_point

        return (
            new_faces,
            new_vertices,
            self._get_new_edges(non_manifold_vertices, new_vertices).tolist(),
        )

    def find_non_manifold_edges(self) -> tuple[NDArray, NDArray, NDArray]:
        unique_edges, counts = np.unique(
            self.mesh.faces_unique_edges.flatten(), return_counts=True
        )
        edges = unique_edges[counts == 4]
        vertices = self.mesh.edges_unique[edges]

        distance_check, edge_index = self.mesh.edges_sorted_tree.query(vertices, k=4)
        if np.any(distance_check):
            raise ValueError("Problem with edge face lookup")
        faces = edge_index // 3

        return faces, vertices, edges

    def _get_faces_at_edges(self, edge_vertices: NDArray) -> NDArray:
        distance_check, edge_index = self.mesh.edges_sorted_tree.query(
            edge_vertices, k=4
        )
        return edge_index[distance_check == 0] // 3

    def _get_faces_at_vertex(self, vertex: int) -> NDArray:
        # Get connected faces
        faces = self.mesh.vertex_faces[vertex]
        faces = faces[faces != -1]  # remove padding (-1 entries)
        return faces

    def _get_face_centers(self, face_indices: NDArray) -> NDArray:
        return np.array(
            np.mean(self.mesh.vertices[self.mesh.faces[face_indices]], axis=1)
        )

    def _get_split_direction_rays(
        self,
        edge: tuple[int, int],
        points: NDArray,
    ) -> tuple[NDArray, NDArray]:
        # Find faces at the first vertex of the edge
        faces1 = self._get_faces_at_vertex(edge[0])
        print(
            f"Vertex {edge[0]} at {points[0]} is connected to {len(faces1)} faces: {faces1}"
        )

        # Find faces at the second vertex of the edge
        faces2 = self._get_faces_at_vertex(edge[1])
        print(
            f"Vertex {edge[1]} at {points[1]} is connected to {len(faces2)} faces: {faces2}"
        )

        # Find all faces connected to the edge
        all_faces = np.unique(np.concatenate([faces1, faces2]))
        print(f"Edge {edge} is connected to {len(all_faces)} faces: {all_faces}")

        # Find all normals for the connected faces
        all_normals = self.mesh.face_normals[all_faces]
        unique_normals = np.unique(all_normals, axis=0)
        direction = unique_normals.sum(axis=0)
        print(f"Edge {edge} has normals sum: {direction}")

        return direction, unique_normals

    def _has_normals_matching_edge(
        self,
        edge_vertex_index: int,
        point: NDArray,
        edge_direction: NDArray,
    ) -> bool:
        # Find faces at the vertex
        faces = self._get_faces_at_vertex(edge_vertex_index)
        print(
            f"Vertex {edge_vertex_index} at {point} is connected to {len(faces)} faces: {faces}"
        )

        # Find all normals for the connected faces
        all_normals = self.mesh.face_normals[faces]
        print(
            f"Vertex {edge_vertex_index} faces have the following normals: {all_normals}"
        )
        colinear_normals = all_normals[np.all(all_normals == edge_direction, axis=1)]
        unique_normals = np.unique(colinear_normals, axis=0)
        print(
            f"Vertex {edge_vertex_index} faces have the following unique colinear normals: {unique_normals}"
        )

        # Check if the normals point towards the edge direction
        dot = np.dot(unique_normals, edge_direction)
        return np.any(dot == 1).item()

    def _split_point(
        self,
        point_to_move: NDArray,
        vertex_to_move: int,
    ) -> tuple[NDArray, int]:
        new_point = point_to_move.copy()
        self.mesh.vertices = np.vstack([self.mesh.vertices, new_point])
        split_vertex_index_2 = self.mesh.vertices.shape[0] - 1
        print(
            f"Split points: {vertex_to_move} at {point_to_move} and {split_vertex_index_2} at {new_point}"
        )

        return new_point, split_vertex_index_2

    def _split_edge(self, points: NDArray) -> tuple[NDArray, int, NDArray, int]:
        new_point_0 = np.mean(points, axis=0)
        new_point_1 = np.mean(points, axis=0)
        self.mesh.vertices = np.vstack([self.mesh.vertices, new_point_0, new_point_1])
        new_vertex_index_0 = self.mesh.vertices.shape[0] - 2
        new_vertex_index_1 = self.mesh.vertices.shape[0] - 1
        print(
            f"Split edge producing 2 points: {new_vertex_index_0} at {new_point_0} and {new_vertex_index_1} at {new_point_1}"
        )

        return new_point_0, new_vertex_index_0, new_point_1, new_vertex_index_1

    def _split_face(
        self,
        edge: NDArray,
        face_index: NDArray,
        face_center: NDArray,
        new_point: NDArray,
        new_vertex_0: int,
        new_vertex_1: int,
        ray_1: NDArray,
        ray_2: NDArray,
    ) -> int:
        face_points = self.mesh.faces[face_index]
        angle_1 = GeometryHelper.angle_between_point_and_ray(
            face_center, new_point, ray_1
        )
        angle_2 = GeometryHelper.angle_between_point_and_ray(
            face_center, new_point, ray_2
        )

        new_face_points = face_points.copy()
        if angle_1 < angle_2:
            face_points[face_points == edge[0]] = new_vertex_0
            new_face_points[face_points == edge[1]] = new_vertex_0
        elif angle_1 > angle_2:
            face_points[face_points == edge[0]] = new_vertex_1
            new_face_points[face_points == edge[1]] = new_vertex_1
        else:
            raise ValueError(
                "Angles are the same, this is impossible. Something broke!"
            )

        self.mesh.faces = np.vstack([self.mesh.faces, new_face_points])

        return self.mesh.faces.shape[0] - 1

    def _reassign_face(
        self,
        face_index: NDArray,
        face_center: NDArray,
        vertex_to_move: int,
        new_point: NDArray,
        new_vertex: int,
        ray_1: NDArray,
        ray_2: NDArray,
    ):
        face_points = self.mesh.faces[face_index]
        angle_1 = GeometryHelper.angle_between_point_and_ray(
            face_center, new_point, ray_1
        )
        angle_2 = GeometryHelper.angle_between_point_and_ray(
            face_center, new_point, ray_2
        )

        if angle_1 < angle_2:
            face_points[face_points == vertex_to_move] = new_vertex
        elif angle_1 > angle_2:
            pass  # No change
        else:
            raise ValueError(
                "Angles are the same, this is impossible. Are any of your face normals inverted?"
            )

    def _get_new_edges(
        self,
        non_manifold_vertices: NDArray,
        new_vertices: list[int],
    ) -> NDArray:
        check = np.hstack([non_manifold_vertices.flatten(), new_vertices])
        mask = np.isin(self.mesh.edges_sorted[:, 0], check) & np.isin(
            self.mesh.edges_sorted[:, 1], check
        )
        return self.mesh.edges_sorted[mask]
