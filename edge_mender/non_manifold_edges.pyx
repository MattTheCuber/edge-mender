# cython: language_level=3
import numpy as np
cimport numpy as cnp
cimport cython


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

                # Break if we've found all non-manifold edges
                if nme_index >= num_non_manifold_edges:
                    break

    return np.asarray(non_manifold_edges), np.asarray(non_manifold_edge_faces)
