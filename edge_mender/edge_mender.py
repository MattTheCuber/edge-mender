"""Provides the class for repairing non-manifold edges in voxel boundary meshes."""

import logging
import math

import numpy as np
import trimesh
from numpy.typing import NDArray

from edge_mender.geometry_helper import GeometryHelper

logging.basicConfig(format="%(message)s")

NON_MANIFOLD_EDGE_FACE_COUNT = 4


class EdgeMender:
    """The class for repairing non-manifold edges in voxel boundary meshes."""

    def __init__(self, mesh: trimesh.Trimesh, *, debug: bool = False) -> None:
        self.mesh = mesh
        self._face_normals: NDArray[np.float64] | None = None
        """Return the unit normal vector for each face.

        If a face is degenerate and a normal can't be generated a zero magnitude unit
        vector will be returned for that face.

        (len(self.faces), 3) float64

        Normal vectors of each face
        """
        self._vertex_faces: NDArray[np.int64] | None = None
        """A representation of the face indices that correspond to each vertex.

        (n,m) int

        Each row contains the face indices that correspond to the given vertex,
        padded with -1 up to the max number of faces corresponding to any one vertex
        Where n == len(self.vertices), m == max number of faces for a single vertex.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if debug else logging.WARNING)

    def validate(self, *, spacing: tuple[float, float, float]) -> None:
        """Validate that the mesh is a valid voxel boundary mesh before repair.

        Raises
        ------
        ValueError
            If the mesh has non-axis-aligned face normals.
        ValueError
            If the mesh has faces with angles that aren't 90° or 45°
        ValueError
            If the mesh has non-uniform face areas.
        """
        test_mesh = self.mesh.copy()
        test_mesh.vertices /= spacing

        # Ensure normals are axis-aligned
        axes = [-1, 0, 1]
        non_axis_aligned_count = np.sum(
            ~np.isin(np.round(test_mesh.face_normals, 8), axes),
        )
        if non_axis_aligned_count > 0:
            msg = (
                f"WARNING: Mesh has {non_axis_aligned_count} "
                "non-axis-aligned face normals."
            )
            raise ValueError(msg)

        # Ensure all faces have 90° or 45° angles
        angles_dist = np.abs(
            test_mesh.face_angles[..., None] - [math.pi / 2, math.pi / 4],
        ).min(axis=2)
        bad_angle_faces = np.sum((angles_dist > 1e-8).any(axis=1))  # noqa: PLR2004
        if bad_angle_faces > 0:
            msg = f"WARNING: Mesh has {bad_angle_faces} faces with bad angles."
            raise ValueError(msg)

        # Ensure all faces have the same area
        unique_areas = len(np.unique(test_mesh.area_faces))
        if unique_areas > 1:
            msg = f"WARNING: Mesh has {unique_areas} unique non-uniform face areas."
            raise ValueError(msg)

    def find_non_manifold_edges(self) -> tuple[NDArray, NDArray, NDArray]:
        unique_edges, counts = np.unique(
            self.mesh.faces_unique_edges.flatten(),
            return_counts=True,
        )
        edges = unique_edges[counts == NON_MANIFOLD_EDGE_FACE_COUNT]
        vertices = self.mesh.edges_unique[edges]

        distance_check, edge_index = self.mesh.edges_sorted_tree.query(
            vertices, k=NON_MANIFOLD_EDGE_FACE_COUNT
        )
        if np.any(distance_check):
            msg = "Problem with edge face lookup"
            raise ValueError(msg)
        faces = edge_index // 3

        return faces, vertices, edges

    def repair(
        self,
        *,
        move_distance: float = 0.0,  # should be less than 25% of the voxel size
        skip_edges: list[int] | None = None,
        only_edges: list[int] | None = None,
    ) -> tuple[list[int], list[int], list[list[int]]]:
        non_manifold_faces, non_manifold_vertices, non_manifold_edges = (
            self.find_non_manifold_edges()
        )
        self.logger.debug("Found %d non-manifold edges\n", len(non_manifold_edges))

        # Cache face normals
        self._face_normals = self.mesh.face_normals

        # Track vertices to move and the amount + direction to move them
        move_vertices: dict[int, NDArray] = {}
        # Track split vertices to avoid double processing
        split_vertices = set()

        new_faces = []
        new_vertices = []

        for original_edge_faces, edge_vertex_indices, edge in zip(
            non_manifold_faces,
            non_manifold_vertices,
            non_manifold_edges,
            strict=True,
        ):
            if skip_edges and edge in skip_edges:
                self.logger.debug("Skipping edge %d as requested", edge)
                continue
            if only_edges and edge not in only_edges:
                self.logger.debug("Skipping edge %d as requested", edge)
                continue
            self.logger.debug("Processing edge %d", edge)

            # Cache vertex faces
            self.mesh._cache.delete("vertex_faces")  # noqa: SLF001
            self._vertex_faces = self.mesh.vertex_faces

            edge_vertex_indices: NDArray
            points = self.mesh.vertices[edge_vertex_indices]
            self.logger.debug(
                "Edge %d connects vertices %s at %s and %s",
                edge,
                edge_vertex_indices,
                points[0],
                points[1],
            )

            current_edge_faces = self._get_faces_at_edge(edge_vertex_indices)
            self.logger.debug(
                "Edge %d was shared by faces %s",
                edge,
                original_edge_faces,
            )
            self.logger.debug(
                "Edge %d is now shared by faces %s",
                edge,
                current_edge_faces,
            )

            for point, edge_vertex_index in zip(
                points,
                edge_vertex_indices,
                strict=True,
            ):
                self.logger.debug(
                    "Processing vertex %d at %s",
                    edge_vertex_index,
                    point,
                )
                if edge_vertex_index in split_vertices:
                    self.logger.debug(
                        "Skipping already handled vertex %d at %s",
                        edge_vertex_index,
                        point,
                    )
                    continue

                # Get the edge direction
                edge_direction = (
                    points[1] - points[0]
                    if edge_vertex_index == edge_vertex_indices[0]
                    else points[0] - points[1]
                )
                self.logger.debug("Edge direction: %s", edge_direction)

                # No floor
                if not self._has_normals_matching_edge(
                    edge_vertex_index,
                    point,
                    edge_direction,
                ):
                    self.logger.debug("No floor detected")

                    other_edge_vertex_index = (
                        edge_vertex_indices[1]
                        if edge_vertex_index == edge_vertex_indices[0]
                        else edge_vertex_indices[0]
                    )
                    other_point = points[1] if point is points[0] else points[0]
                    self.logger.debug(
                        "Splitting vertex %d at %s",
                        edge_vertex_index,
                        point,
                    )
                    self.logger.debug(
                        "Other vertex %d at %s",
                        other_edge_vertex_index,
                        other_point,
                    )

                    ray_1, ray_2 = self._get_split_direction_rays(
                        edge_direction,
                        original_edge_faces,
                        points,
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
                        faces_to_reconnect,
                    )
                    for face_index, face_center in zip(
                        faces_to_reconnect,
                        faces_to_reconnect_centers,
                        strict=True,
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
                    self.logger.debug("Floor detected, skipping vertex split")

            # Floor and ceiling case
            if (
                edge_vertex_indices[0] not in split_vertices
                and edge_vertex_indices[1] not in split_vertices
            ):
                self.logger.debug("No vertices split, floor and ceiling case")

                ray_1, ray_2 = self._get_split_direction_rays(
                    edge_direction,
                    original_edge_faces,
                    points,
                )

                new_point_0, new_vertex_0, new_point_1, new_vertex_1 = self._split_edge(
                    points,
                )
                new_vertices.append(new_vertex_0)
                new_vertices.append(new_vertex_1)
                if move_distance:
                    move_vertices[new_vertex_0] = new_point_0 + (ray_1 * move_distance)
                    move_vertices[new_vertex_1] = new_point_1 + (ray_2 * move_distance)

                faces_to_reconnect = np.array(list(current_edge_faces))
                faces_to_reconnect_centers = self._get_face_centers(faces_to_reconnect)
                for face_index, face_center in zip(
                    faces_to_reconnect,
                    faces_to_reconnect_centers,
                    strict=True,
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
                    self._face_normals = np.vstack(
                        [self._face_normals, self._face_normals[face_index]],
                    )
                    new_faces.append(face_index)
                    new_faces.append(new_face_index)

            self.logger.debug("")

        # Move after splitting since movements will make normals that aren't orthogonal
        for vertex_to_move, new_point in move_vertices.items():
            self.logger.debug(
                "Moving vertex %d from %s to %s",
                vertex_to_move,
                self.mesh.vertices[vertex_to_move],
                new_point,
            )
            self.mesh.vertices[vertex_to_move] = new_point

        return (
            new_faces,
            new_vertices,
            self._get_new_edges(non_manifold_vertices, new_vertices).tolist(),
        )

    def _get_faces_at_edge(self, edge_vertices: NDArray) -> NDArray:
        return self.mesh.edges_face[
            np.concatenate(
                (
                    np.where(
                        (self.mesh.edges[:, 0] == edge_vertices[0])
                        & (self.mesh.edges[:, 1] == edge_vertices[1]),
                    )[0],
                    np.where(
                        (self.mesh.edges[:, 0] == edge_vertices[1])
                        & (self.mesh.edges[:, 1] == edge_vertices[0]),
                    )[0],
                ),
            )
        ]

    def _get_faces_at_vertex(self, vertex: int) -> NDArray:
        # Get connected faces
        faces = self._vertex_faces[vertex]
        return faces[faces != -1]  # remove padding (-1 entries)

    def _get_face_centers(self, face_indices: NDArray) -> NDArray:
        return np.array(
            np.mean(self.mesh.vertices[self.mesh.faces[face_indices]], axis=1),
        )

    def _has_normals_matching_edge(
        self,
        edge_vertex_index: int,
        point: NDArray,
        edge_direction: NDArray,
    ) -> bool:
        # Find faces at the vertex
        faces = self._get_faces_at_vertex(edge_vertex_index)
        self.logger.debug(
            "Vertex %d at %s is connected to %d faces: %s",
            edge_vertex_index,
            point,
            len(faces),
            faces,
        )

        # Find all normals for the connected faces
        all_normals = self._face_normals[faces]
        self.logger.debug(
            "Vertex %d faces have the following normals: %s",
            edge_vertex_index,
            all_normals,
        )
        colinear_normals = all_normals[np.all(all_normals == edge_direction, axis=1)]
        unique_normals = np.unique(colinear_normals, axis=0)
        self.logger.debug(
            "Vertex %d faces have the following unique colinear normals: %s",
            edge_vertex_index,
            unique_normals,
        )

        # Check if the normals point towards the edge direction
        dot = np.dot(unique_normals, edge_direction)
        return np.any(dot == 1).item()

    def _get_split_direction_rays(
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
        self.logger.debug(
            "Line directions: %s and %s",
            line_direction_1,
            line_direction_2,
        )

        edge_face_centers = self._get_face_centers(edge_face_indices)
        self.logger.debug("Edge face centers: %s", edge_face_centers)

        group_1_centers = []
        group_2_centers = []
        group_1_faces = []
        group_2_faces = []
        for center, face_index in zip(
            edge_face_centers,
            edge_face_indices,
            strict=True,
        ):
            self.logger.debug(
                "Testing if point %s is left of line at %s with direction %s",
                center[i],
                edge_points[0][i],
                line_direction_1[i],
            )
            if GeometryHelper.is_left(
                edge_points[0][i],
                line_direction_1[i],
                center[i],
            ):
                group_1_centers.append(center)
                group_1_faces.append(face_index)
            else:
                group_2_centers.append(center)
                group_2_faces.append(face_index)
        self.logger.debug(
            "Group 1 has faces %s with centers %s, "
            "Group 2 has faces %s with centers %s",
            group_1_faces,
            group_1_centers,
            group_2_faces,
            group_2_centers,
        )

        group_1_normals = self._face_normals[np.array(group_1_faces)]
        group_2_normals = self._face_normals[np.array(group_2_faces)]
        self.logger.debug(
            "Group 1 has normals %s, Group 2 has normals %s",
            list(group_1_normals),
            list(group_2_normals),
        )

        groups_intersect = GeometryHelper.rays_intersect(
            point_1=group_1_centers[0][i],
            normal_1=group_1_normals[0][i],
            point_2=group_1_centers[1][i],
            normal_2=group_1_normals[1][i],
        )
        self.logger.debug("Groups intersect? %s", groups_intersect)
        self.logger.debug("Groups correct? %s", not groups_intersect)

        ray_1 = line_direction_1 if groups_intersect else line_direction_2
        ray_2 = ray_1 * -1
        self.logger.debug("Ray directions: %s and %s", ray_1, ray_2)

        return ray_1, ray_2

    def _split_point(
        self,
        point_to_move: NDArray,
        vertex_to_move: int,
    ) -> tuple[NDArray, int]:
        new_point = point_to_move.copy()
        self.mesh.vertices = np.vstack([self.mesh.vertices, new_point])
        split_vertex_index_2 = self.mesh.vertices.shape[0] - 1
        self.logger.debug(
            "Split points: %d at %s and %d at %s",
            vertex_to_move,
            point_to_move,
            split_vertex_index_2,
            new_point,
        )

        return new_point, split_vertex_index_2

    def _split_edge(self, points: NDArray) -> tuple[NDArray, int, NDArray, int]:
        new_point_0 = np.mean(points, axis=0)
        new_point_1 = np.mean(points, axis=0)
        self.mesh.vertices = np.vstack([self.mesh.vertices, new_point_0, new_point_1])
        new_vertex_index_0 = self.mesh.vertices.shape[0] - 2
        new_vertex_index_1 = self.mesh.vertices.shape[0] - 1
        self.logger.debug(
            "Split edge producing 2 points: %d at %s and %d at %s",
            new_vertex_index_0,
            new_point_0,
            new_vertex_index_1,
            new_point_1,
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
            face_center,
            new_point,
            ray_1,
        )
        angle_2 = GeometryHelper.angle_between_point_and_ray(
            face_center,
            new_point,
            ray_2,
        )

        new_face_points = face_points.copy()
        if angle_1 < angle_2:
            face_points[face_points == edge[0]] = new_vertex_0
            new_face_points[face_points == edge[1]] = new_vertex_0
        elif angle_1 > angle_2:
            face_points[face_points == edge[0]] = new_vertex_1
            new_face_points[face_points == edge[1]] = new_vertex_1
        else:
            msg = "Angles are the same, this is impossible. Something broke!"
            raise ValueError(msg)

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
            face_center,
            new_point,
            ray_1,
        )
        angle_2 = GeometryHelper.angle_between_point_and_ray(
            face_center,
            new_point,
            ray_2,
        )

        if angle_1 < angle_2:
            face_points[face_points == vertex_to_move] = new_vertex
        elif angle_1 > angle_2:
            pass  # No change
        else:
            msg = (
                "Angles are the same, this is impossible. "
                "Are any of your face normals inverted?"
            )
            raise ValueError(msg)

    def _get_new_edges(
        self,
        non_manifold_vertices: NDArray,
        new_vertices: list[int],
    ) -> NDArray:
        check = np.hstack([non_manifold_vertices.flatten(), new_vertices])
        mask = np.isin(self.mesh.edges_sorted[:, 0], check) & np.isin(
            self.mesh.edges_sorted[:, 1],
            check,
        )
        return self.mesh.edges_sorted[mask]
