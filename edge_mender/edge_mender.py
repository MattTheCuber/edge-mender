"""Provides the class for repairing non-manifold edges in voxel boundary meshes."""

import logging
import math

import numpy as np
import trimesh
from numpy.typing import NDArray

from edge_mender.geometry_helper import GeometryHelper
from edge_mender.non_manifold_edges import (
    find_non_manifold_edges,
    get_faces_at_edge,
    get_faces_at_vertex,
)
from edge_mender.non_manifold_vertices import repair_vertices

logging.basicConfig(format="%(message)s")

NON_MANIFOLD_EDGE_FACE_COUNT = 4


class EdgeMender:
    """The class for repairing non-manifold edges in voxel boundary meshes."""

    def __init__(self, mesh: trimesh.Trimesh, *, debug: bool = False) -> None:
        self.mesh = mesh
        self._face_normals: NDArray[np.float64] = np.empty((0, 3), dtype=np.float64)
        """Return the unit normal vector for each face.

        If a face is degenerate and a normal can't be generated a zero magnitude unit
        vector will be returned for that face.

        (len(self.faces), 3) float64

        Normal vectors of each face
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if debug else logging.WARNING)

    def validate(self, *, spacing: tuple[float, float, float]) -> None:
        """Validate that the mesh is a valid voxel boundary mesh before repair.

        Parameters
        ----------
        spacing : tuple[float, float, float]
            The spacing of the mesh in each dimension. This is used to check that the
            face areas are uniform and that the angles are correct.

        Raises
        ------
        ValueError
            If the mesh has non-axis-aligned face normals.
        ValueError
            If the mesh has faces with angles that aren't 90° or 45°
        ValueError
            If the mesh has non-uniform face areas.

        References
        ----------
        .. [1] https://trimesh.org/trimesh.html#trimesh.Trimesh.face_angles
        .. [2] https://trimesh.org/trimesh.html#trimesh.Trimesh.area_faces
        """
        test_mesh = self.mesh.copy()
        test_mesh.vertices /= spacing

        # Check if empty
        if test_mesh.is_empty:
            msg = "Mesh is empty."
            raise ValueError(msg)

        # Check if volume is positive
        if test_mesh.volume <= 0:
            msg = "Mesh has non-positive volume."
            raise ValueError(msg)

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

    def find_non_manifold_edges_2(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Find non-manifold edges within the mesh.

        Non-manifold edges are defined as edges shared by 4 faces.

        Returns
        -------
        non_manifold_vertices : NDArray[np.int64]
            An (n, 2) array of the two vertex indices for each non-manifold edge.
        non_manifold_faces : NDArray[np.int64]
            An (n, 4) array of the four face indices for each non-manifold edge.
        """
        return find_non_manifold_edges(self.mesh.edges)

    def find_non_manifold_edges(
        self,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        """Find non-manifold edges within the mesh.

        Non-manifold edges are defined as edges shared by 4 faces.

        Returns
        -------
        non_manifold_faces : NDArray[np.int64]
            An (n, 4) array of the four face indices for each non-manifold edge.
        non_manifold_vertices : NDArray[np.int64]
            An (n, 2) array of the two vertex indices for each non-manifold edge.
        non_manifold_edges : NDArray[np.int64]
            An (n,) array of the edge indices for each non-manifold edge.

        Raises
        ------
        ValueError
            If there is a problem with the edge face lookup.

        References
        ----------
        .. [1] https://github.com/mikedh/trimesh/issues/2469
        """
        # Find all unique edges and their face counts
        unique_edges, counts = np.unique(
            self.mesh.faces_unique_edges.flatten(),
            return_counts=True,
        )
        # Find the edges that are shared by 4 faces
        edges = unique_edges[counts == NON_MANIFOLD_EDGE_FACE_COUNT]

        # Get the vertices for each edge
        vertices = self.mesh.edges_unique[edges]

        # Get the faces for each edge
        edge_index = self.mesh.edges_sorted_tree.query(
            vertices,
            k=NON_MANIFOLD_EDGE_FACE_COUNT,
        )[1]
        faces = edge_index // 3

        return faces, vertices, edges

    def repair(self, *, shift_distance: float = 0.0) -> None:
        """Repair non-manifold edges in the mesh.

        Non-manifold edges are defined as edges shared by 4 faces.

        It accomplishes this by iterating through each non-manifold edge and
        splitting the vertices or faces as needed to restore manifoldness.

        Parameters
        ----------
        shift_distance : float, optional
            The distance to shift vertices when repairing, by default 0.0

            This will typically only be used for visualization purposes. The
            value should be less than 25% of the voxel size to avoid creating
            intersecting faces.
        """
        nme_vertices, nme_faces = self.find_non_manifold_edges_2()
        self.logger.debug("Found %d non-manifold edges\n", len(nme_vertices))

        # Cache face normals
        self._face_normals = self.mesh.face_normals

        # Track vertices to shift and the amount + direction to shift them
        shift_vertices: dict[int, NDArray] = {}
        # Track split vertices to avoid double processing
        split_vertices = set()

        for original_edge_faces, edge_vertex_indices in zip(
            nme_faces,
            nme_vertices,
            strict=True,
        ):
            edge_vertex_indices: NDArray
            points = self.mesh.vertices[edge_vertex_indices]
            self.logger.debug(
                "Processing edge %s at %s and %s",
                edge_vertex_indices,
                points[0],
                points[1],
            )

            current_edge_faces = get_faces_at_edge(
                edge_vertex_indices[0],
                edge_vertex_indices[1],
                self.mesh.faces,
            )
            self.logger.debug(
                "Edge %s was shared by faces %s",
                edge_vertex_indices,
                original_edge_faces,
            )
            self.logger.debug(
                "Edge %s is now shared by faces %s",
                edge_vertex_indices,
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

                # Find all faces at this vertex
                faces_at_vertex = get_faces_at_vertex(
                    edge_vertex_index,
                    self.mesh.faces,
                )
                self.logger.debug(
                    "Vertex %d at %s is connected to %d faces: %s",
                    edge_vertex_index,
                    point,
                    len(faces_at_vertex),
                    faces_at_vertex,
                )

                # No floor
                if not self._has_normals_matching_edge(
                    edge_vertex_index,
                    faces_at_vertex,
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

                    split_direction = self._get_split_direction(
                        edge_direction,
                        original_edge_faces,
                        points,
                    )

                    new_point, new_vertex = self._split_point(point, edge_vertex_index)
                    if shift_distance:
                        shift_vertices[edge_vertex_index] = point - (
                            split_direction * shift_distance
                        )
                        shift_vertices[new_vertex] = new_point + (
                            split_direction * shift_distance
                        )

                    faces_centers_at_vertex = self._get_face_centers(
                        faces_at_vertex,
                    )
                    for face_index, face_center in zip(
                        faces_at_vertex,
                        faces_centers_at_vertex,
                        strict=True,
                    ):
                        self._reassign_face(
                            face_index,
                            face_center,
                            edge_vertex_index,
                            new_point,
                            new_vertex,
                            split_direction,
                        )

                    split_vertices.add(edge_vertex_index)
                else:
                    self.logger.debug("Floor detected, skipping vertex split")

            # Floor and ceiling case
            if (
                edge_vertex_indices[0] not in split_vertices
                and edge_vertex_indices[1] not in split_vertices
            ):
                self.logger.debug("No vertices split, floor and ceiling case")

                split_direction = self._get_split_direction(
                    edge_direction,
                    original_edge_faces,
                    points,
                )

                new_point_left, new_vertex_left, new_point_right, new_vertex_right = (
                    self._split_edge(points)
                )
                if shift_distance:
                    shift_vertices[new_vertex_left] = new_point_left + (
                        split_direction * shift_distance
                    )
                    shift_vertices[new_vertex_right] = new_point_right - (
                        split_direction * shift_distance
                    )

                faces_to_reconnect = np.array(list(current_edge_faces))
                faces_to_reconnect_centers = self._get_face_centers(faces_to_reconnect)
                for face_index, face_center in zip(
                    faces_to_reconnect,
                    faces_to_reconnect_centers,
                    strict=True,
                ):
                    self._split_face(
                        edge_vertex_indices,
                        face_index,
                        face_center,
                        new_point_left,
                        new_vertex_left,
                        new_vertex_right,
                        split_direction,
                    )
                    self._face_normals = np.vstack(
                        [self._face_normals, self._face_normals[face_index]],
                    )

            self.logger.debug("")

        # Shift after splitting since shifting will make normals that aren't orthogonal
        for vertex_to_shift, new_point in shift_vertices.items():
            self.logger.debug(
                "Shifting vertex %d from %s to %s",
                vertex_to_shift,
                self.mesh.vertices[vertex_to_shift],
                new_point,
            )
            self.mesh.vertices[vertex_to_shift] = new_point

    def repair_vertices(self, *, shift_distance: float = 0.0) -> None:
        """Repair non-manifold vertices in the mesh.

        Requires all non-manifold edges to have already been repaired.

        Non-manifold vertices are defined as vertices where more than one
        contiguous group of faces originates.

        It accomplishes this by finding and iterating through each non-manifold
        vertex, splitting them to restore manifoldness.

        Parameters
        ----------
        shift_distance : float, optional
            The distance to shift vertices when repairing, by default 0.0

            This will typically only be used for visualization purposes. The
            value should be less than 25% of the voxel size to avoid creating
            intersecting faces.
        """
        self.mesh.vertices = repair_vertices(
            self.mesh.faces,
            self.mesh.vertices,
            self.mesh.vertex_faces,
            self.mesh.triangles_center,
            shift_distance=shift_distance,
        )

    def _get_face_centers(self, face_indices: NDArray) -> NDArray:
        """Get the centers of the given face indices.

        Parameters
        ----------
        face_indices : NDArray
            The face indices.

        Returns
        -------
        NDArray
            The centers of the faces.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Triangle_center
        """
        return np.array(
            np.mean(self.mesh.vertices[self.mesh.faces[face_indices]], axis=1),
        )

    def _has_normals_matching_edge(
        self,
        edge_vertex_index: int,
        faces_at_vertex: NDArray,
        edge_direction: NDArray,
    ) -> bool:
        """Check if any face normals at the given vertex match the edge direction.

        Parameters
        ----------
        edge_vertex_index : int
            The index of the vertex at the edge.
        faces_at_vertex : NDArray
            The face indices at the vertex.
        edge_direction : NDArray
            The direction vector of the edge.

        Returns
        -------
        bool
            Whether any face normals at the vertex match the edge direction.
        """
        # Find all normals for the connected faces
        normals = self._face_normals[faces_at_vertex]
        self.logger.debug(
            "Vertex %d faces have the following normals: %s",
            edge_vertex_index,
            normals,
        )
        return GeometryHelper.any_directions_match(edge_direction, normals)

    def _get_split_direction(
        self,
        edge_direction: NDArray,
        edge_face_indices: NDArray,
        edge_points: NDArray,
    ) -> NDArray:
        """Get the direction vector to use for splitting.

        Parameters
        ----------
        edge_direction : NDArray
            The direction vector of the edge.
        edge_face_indices : NDArray
            The indices of the faces adjacent to the edge.
        edge_points : NDArray
            The (x, y, z) coordinates of the points defining the edge.

        Returns
        -------
        split_direction : NDArray
            The direction vector.
        """
        i = ~edge_direction.astype(bool)

        # Get two orthagonal directions to the edge direction
        line_direction_a, line_direction_b = (
            GeometryHelper.get_diagonal_orthogonal_directions(edge_direction)
        )
        self.logger.debug(
            "Line directions: %s and %s",
            line_direction_a,
            line_direction_b,
        )

        edge_face_centers = self._get_face_centers(edge_face_indices)
        self.logger.debug("Edge face centers: %s", edge_face_centers)

        group_a_centers = []
        group_a_faces = []
        group_b_centers = []
        group_b_faces = []
        for center, face_index in zip(
            edge_face_centers,
            edge_face_indices,
            strict=True,
        ):
            self.logger.debug(
                "Testing if point %s is left of line at %s with direction %s",
                center[i],
                edge_points[0][i],
                line_direction_a[i],
            )
            if GeometryHelper.is_left(
                edge_points[0][i],
                line_direction_a[i],
                center[i],
            ):
                group_a_centers.append(center)
                group_a_faces.append(face_index)
            else:
                group_b_centers.append(center)
                group_b_faces.append(face_index)
        self.logger.debug(
            "Group 1 has faces %s with centers %s, "
            "Group 2 has faces %s with centers %s",
            group_a_faces,
            group_a_centers,
            group_b_faces,
            group_b_centers,
        )

        group_a_normals = self._face_normals[np.array(group_a_faces)]
        group_b_normals = self._face_normals[np.array(group_b_faces)]
        self.logger.debug(
            "Group 1 has normals %s, Group 2 has normals %s",
            list(group_a_normals),
            list(group_b_normals),
        )

        groups_intersect = GeometryHelper.rays_intersect(
            point_a=group_a_centers[0][i],
            normal_a=group_a_normals[0][i],
            point_b=group_a_centers[1][i],
            normal_b=group_a_normals[1][i],
        )
        self.logger.debug("Groups intersect? %s", groups_intersect)
        self.logger.debug("Groups correct? %s", not groups_intersect)

        split_direction = line_direction_a if groups_intersect else line_direction_b
        self.logger.debug("Split direction: %s", split_direction)
        return split_direction

    def _split_point(
        self,
        point_to_split: NDArray,
        vertex_to_split: int,
    ) -> tuple[NDArray, int]:
        """Split the given point by duplicating the provided vertex in the mesh.

        Parameters
        ----------
        point_to_split : NDArray
            The (x, y, z) coordinates of the point to split.
        vertex_to_split : int
            The index of the vertex to split.

        Returns
        -------
        new_point : NDArray
            The (x, y, z) coordinates of the new point created from the split.
        new_vertex_index : int
            The index of the new vertex created from the split.
        """
        new_point = point_to_split.copy()
        self.mesh.vertices = np.vstack([self.mesh.vertices, new_point])
        new_vertex_index = self.mesh.vertices.shape[0] - 1
        self.logger.debug(
            "Split points: %d at %s and %d at %s",
            vertex_to_split,
            point_to_split,
            new_vertex_index,
            new_point,
        )
        return new_point, new_vertex_index

    def _reassign_face(
        self,
        face_index: int,
        face_center: NDArray,
        vertex_to_reassign: int,
        new_point: NDArray,
        new_vertex: int,
        split_direction: NDArray,
    ) -> None:
        """Reassign the given face to the new vertex based on the angles.

        Parameters
        ----------
        face_index : int
            The index of the face to reassign.
        face_center : NDArray
            The center point of the face to reassign.
        vertex_to_reassign : int
            The index of the vertex on this face to reassign.
        new_point : NDArray
            The (x, y, z) coordinates of the new point created from the split.
        new_vertex : int
            The index of the new vertex created from the split.
        split_direction : NDArray
            The split direction vector.

        Raises
        ------
        ValueError
            Raised if face center is equidistant to the split direction origin.
        """
        face_points = self.mesh.faces[face_index]
        matches_split_direction = GeometryHelper.point_in_direction(
            new_point,
            split_direction,
            face_center,
        )

        if matches_split_direction:
            face_points[face_points == vertex_to_reassign] = new_vertex
        else:
            pass  # No change

    def _split_edge(self, points: NDArray) -> tuple[NDArray, int, NDArray, int]:
        """Split the given edge by creating two new vertices in the mesh.

        Parameters
        ----------
        points : NDArray
            The (x, y, z) coordinates of the points defining the edge to split.

        Returns
        -------
        new_point_left : NDArray
            The (x, y, z) coordinates of the first new point created from the split.
        new_vertex_left : int
            The index of the first new vertex created from the split.
        new_point_right : NDArray
            The (x, y, z) coordinates of the second new point created from the split.
        new_vertex_right : int
            The index of the second new vertex created from the split.
        """
        new_point_left = np.mean(points, axis=0)
        new_point_right = np.mean(points, axis=0)
        self.mesh.vertices = np.vstack(
            [self.mesh.vertices, new_point_left, new_point_right],
        )
        new_vertex_index_left = self.mesh.vertices.shape[0] - 2
        new_vertex_index_right = self.mesh.vertices.shape[0] - 1
        self.logger.debug(
            "Split edge producing 2 points: %d at %s and %d at %s",
            new_vertex_index_left,
            new_point_left,
            new_vertex_index_right,
            new_point_right,
        )
        return (
            new_point_left,
            new_vertex_index_left,
            new_point_right,
            new_vertex_index_right,
        )

    def _split_face(
        self,
        edge_vertices: NDArray,
        face_index: int,
        face_center: NDArray,
        new_point: NDArray,
        new_vertex_left: int,
        new_vertex_right: int,
        split_direction: NDArray,
    ) -> int:
        """Split the face sharing an edge in half.

        Parameters
        ----------
        edge_vertices : NDArray
            The vertex indices defining the edge to split.
        face_index : int
            The index of the face to split.
        face_center : NDArray
            The center point of the face to split.
        new_point : NDArray
            The (x, y, z) coordinates of the new points created from the split.
        new_vertex_left : int
            The index of the first new vertex created from the split.
        new_vertex_right : int
            The index of the second new vertex created from the split.
        split_direction : NDArray
            The split direction vector.

        Returns
        -------
        int
            The index of the new face created from the split.

        Raises
        ------
        ValueError
            Raised if face center is equidistant to the split direction origin.
        """
        face_points = self.mesh.faces[face_index]
        matches_split_direction = GeometryHelper.point_in_direction(
            new_point,
            split_direction,
            face_center,
        )

        new_face_points = face_points.copy()
        if matches_split_direction:
            face_points[face_points == edge_vertices[0]] = new_vertex_left
            new_face_points[face_points == edge_vertices[1]] = new_vertex_left
        else:
            face_points[face_points == edge_vertices[0]] = new_vertex_right
            new_face_points[face_points == edge_vertices[1]] = new_vertex_right

        self.mesh.faces = np.vstack([self.mesh.faces, new_face_points])

        return self.mesh.faces.shape[0] - 1
