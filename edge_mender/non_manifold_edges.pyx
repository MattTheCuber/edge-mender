# cython: language_level=3
import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport malloc, free


cdef int MAX_FACES_PER_VERTEX = 24


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_non_manifold_edges(
    cnp.ndarray[cnp.int64_t, ndim=2] edges,
) -> tuple[cnp.ndarray[cnp.int64_t], cnp.ndarray[cnp.int64_t]]:
    """Finds and returns non-manifold edges.

    Non-manifold edges are defined as edges shared by 4 faces.
    """
    # Initialize variables
    # Loop index variables
    cdef Py_ssize_t edge
    # The number of edges in the mesh
    cdef Py_ssize_t num_edges = edges.shape[0]

    # Sort the edges to count sequentially
    edges = np.sort(edges, axis=1)
    cdef cnp.int64_t[::1] order = np.lexsort((edges[:, 1], edges[:, 0]))
    # The last unique edge index
    cdef cnp.int64_t last_unique_edge = 0
    # The number of faces found on the current edge
    cdef short edge_face_count = 0

    # The number of non-manifold edges found
    cdef long num_non_manifold_edges = 0

    # Pass 1: Count non-manifold edges
    with nogil:
        # For each edge
        for edge in range(num_edges):
            # Increment face count for this edge
            edge_face_count += 1

            # If this edge is different from the last unique edge
            if (
                edges[order[edge], 0] != edges[order[last_unique_edge], 0]
                or edges[order[edge], 1] != edges[order[last_unique_edge], 1]
            ):
                # Reset face count for new edge and update last unique edge
                edge_face_count = 1
                last_unique_edge = edge
                continue

            # If this edge has 4 faces, it's non-manifold
            if edge_face_count == 4:
                num_non_manifold_edges += 1

    # Initialize array to hold non-manifold edges
    cdef cnp.int64_t[:, ::1] non_manifold_edges = (
        np.zeros((num_non_manifold_edges, 2), dtype=np.int64)
    )
    # The counter for the current non-manifold edge index
    cdef long nme_index = 0
    # Initialize array to hold non-manifold edge faces
    cdef cnp.int64_t[:, ::1] non_manifold_edge_faces = (
        np.full((num_non_manifold_edges, 4), -1, dtype=np.int64)
    )

    # Reset variables
    edge_face_count = 0
    last_unique_edge = 0

    # Pass 2: Store non-manifold edges and their faces
    with nogil:
        # For each edge
        for edge in range(num_edges):
            # Break if we've found all non-manifold edges
            if nme_index >= num_non_manifold_edges:
                break

            # Increment face count for this edge
            edge_face_count += 1

            # If this edge is different from the last unique edge
            if (
                edges[order[edge], 0] != edges[order[last_unique_edge], 0]
                or edges[order[edge], 1] != edges[order[last_unique_edge], 1]
            ):
                # Reset face count for new edge and update last unique edge
                edge_face_count = 1
                last_unique_edge = edge

            # Add the face index to the non-manifold edge faces array
            non_manifold_edge_faces[nme_index, edge_face_count - 1] = (
                order[edge] // 3
            )

            # If this edge has 4 faces, it's non-manifold
            if edge_face_count == 4:
                # Store the non-manifold edge
                non_manifold_edges[nme_index, 0] = edges[order[edge], 0]
                non_manifold_edges[nme_index, 1] = edges[order[edge], 1]

                # Increment non-manifold edge index counter
                nme_index += 1

    return np.asarray(non_manifold_edges), np.asarray(non_manifold_edge_faces)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_faces_at_edge(
    cnp.int64_t e_v0,
    cnp.int64_t e_v1,
    cnp.ndarray[cnp.int64_t, ndim=2] faces,
) -> cnp.ndarray[cnp.int64_t]:
    """Finds and returns the face indices for the given edge vertices.

    This assumes edges have 4 faces at most.
    """
    # Initialize variables
    cdef cnp.int64_t[:, ::1] faces_view = faces
    # Loop index variables
    cdef Py_ssize_t face
    # The number of faces in the mesh
    cdef Py_ssize_t num_faces = faces.shape[0]

    # The current face vertex indices
    cdef cnp.int64_t f_v0, f_v1, f_v2
    # Match flags for each vertex
    cdef bint m0, m1, m2
    # The number of faces found
    cdef int num_faces_at_edge = 0
    # Initialize array to hold face indices
    cdef cnp.int64_t* edge_faces_ptr
    cdef cnp.ndarray[cnp.int64_t, ndim=1] edge_faces
    edge_faces_ptr = <cnp.int64_t*>malloc(4 * sizeof(cnp.int64_t))

    try:
        with nogil:
            # For each face
            for face in range(num_faces):
                # Retrieve the three vertices of the current face
                f_v0 = faces_view[face, 0]
                f_v1 = faces_view[face, 1]
                f_v2 = faces_view[face, 2]

                # Check for matching vertices
                m0 = (f_v0 == e_v0) or (f_v0 == e_v1)
                m1 = (f_v1 == e_v0) or (f_v1 == e_v1)
                m2 = (f_v2 == e_v0) or (f_v2 == e_v1)

                # If two vertices match, the face is on the edge
                if (m0 + m1 + m2) == 2:
                    edge_faces_ptr[num_faces_at_edge] = face
                    num_faces_at_edge += 1

                    if num_faces_at_edge == 4:
                        break

        # Create result array and copy data
        edge_faces = np.empty(num_faces_at_edge, dtype=np.int64)
        for face in range(num_faces_at_edge):
            edge_faces[face] = edge_faces_ptr[face]

        return edge_faces
    finally:
        free(edge_faces_ptr)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_faces_at_vertex(
    cnp.int64_t vertex,
    cnp.ndarray[cnp.int64_t, ndim=2] faces,
) -> cnp.ndarray[cnp.int64_t]:
    """Finds and returns the face indices for the given vertex."""
    # Initialize variables
    cdef cnp.int64_t[:, ::1] faces_view = faces
    # Loop index variables
    cdef Py_ssize_t face
    # The number of faces in the mesh
    cdef Py_ssize_t num_faces = faces.shape[0]

    # Initialize array to hold face indices
    cdef cnp.ndarray[cnp.int64_t, ndim=1] vertex_faces = np.empty(
        MAX_FACES_PER_VERTEX,
        dtype=np.int64,
    )
    # The number of faces found
    cdef int num_faces_at_vertex = 0

    with nogil:
        # For each face
        for face in range(num_faces):
            # If any vertex matches
            if (
                faces_view[face, 0] == vertex
                or faces_view[face, 1] == vertex
                or faces_view[face, 2] == vertex
            ):
                # Add the face index to the array and increment the count
                vertex_faces[num_faces_at_vertex] = face
                num_faces_at_vertex += 1

                if num_faces_at_vertex >= MAX_FACES_PER_VERTEX:
                    break

    return vertex_faces[:num_faces_at_vertex]
