# cython: language_level=3
import numpy as np
cimport numpy as cnp
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def has_non_manifold_vertices(
    cnp.ndarray[cnp.int64_t, ndim=2] faces,
    cnp.ndarray[cnp.int64_t, ndim=2] vertex_faces,
) -> bool:
    """Return whether the mesh has non-manifold vertices."""
    cdef long num_split_vertices = find_num_split_vertices(
        faces,
        vertex_faces,
        np.zeros(vertex_faces.shape[1], dtype=np.uint64),
    )
    return num_split_vertices > 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef long find_num_split_vertices(
    const cnp.int64_t[:, ::1] faces,
    const cnp.int64_t[:, ::1] vertex_faces,
    cnp.uint64_t[::1] visited_faces,
) nogil:
    """Finds non-manifold vertices and counts the number of splits to repair.

    Non-manifold vertices are defined as as vertices where more than one
    contiguous group of faces originates.
    """
    # Initialize variables
    # Loop index variables
    cdef Py_ssize_t vertex, i, j, k
    # The number of input vertices
    cdef Py_ssize_t num_vertices = vertex_faces.shape[0]
    # The maximum number of faces for the input vertices
    cdef Py_ssize_t num_vertex_faces = vertex_faces.shape[1]
    # The current and neighbor face indices
    cdef cnp.int64_t face, neighbor
    # The current and neighbor face vertex indicies
    cdef cnp.int64_t f_v0, f_v1, f_v2, n_v0, n_v1, n_v2

    # A counter to enable recycling `visited_faces` for each vertex
    cdef cnp.uint64_t stamp = 0

    # A counter for the number of groups found at the current vertex
    cdef cnp.uint64_t groups = 0
    # The number of vertices matched between the current and neighbor face
    # If two vertices match, the test face is a neighbor
    cdef short matched_vertex_count = 0

    # The number of vertex splits that will need to be made
    cdef long num_split_vertices = 0

    # For each vertex
    for vertex in range(num_vertices):
        # Increase stamp counter and reset group count for this vertex
        stamp += 1
        groups = 0

        # For each face at this vertex
        for i in range(num_vertex_faces):
            # Get the global face index
            face = vertex_faces[vertex, i]

            # Skip padding or already processed faces
            if face == -1 or visited_faces[i] == stamp:
                continue

            # Create a new group
            groups += 1

            # Mark face as visited
            visited_faces[i] = stamp

            # If more than one group, we have a non-manifold vertex
            if groups > 1:
                # We will need to split this vertex for this new group
                num_split_vertices += 1

            # Retrieve the 3 vertex indicies of the current face
            f_v0 = faces[face, 0]
            f_v1 = faces[face, 1]
            f_v2 = faces[face, 2]

            # Find faces in chain
            for j in range(num_vertex_faces):
                # Find neighbor face to the current face
                for k in range(num_vertex_faces):
                    neighbor = vertex_faces[vertex, k]

                    # Skip padding or already processed faces
                    if neighbor == -1 or visited_faces[k] == stamp:
                        continue

                    # Retrieve the three vertices of the neighbor face
                    n_v0 = faces[neighbor, 0]
                    n_v1 = faces[neighbor, 1]
                    n_v2 = faces[neighbor, 2]

                    # Count shared vertices
                    matched_vertex_count = (
                        (f_v0 == n_v0) + (f_v0 == n_v1) + (f_v0 == n_v2) + 
                        (f_v1 == n_v0) + (f_v1 == n_v1) + (f_v1 == n_v2) + 
                        (f_v2 == n_v0) + (f_v2 == n_v1) + (f_v2 == n_v2)
                    )

                    # If they share more than one vertex, they are connected
                    if matched_vertex_count > 1:
                        # Mark neighbor as visited
                        visited_faces[k] = stamp

                        # Change current face to neighbor
                        face = neighbor
                        f_v0 = n_v0
                        f_v1 = n_v1
                        f_v2 = n_v2

                        # Break out of this inner loop to begin searching
                        # through the faces again for the new neighbor
                        break

                # Finish chain if no more connected faces
                if matched_vertex_count < 2:
                    break

    return num_split_vertices

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def repair_vertices(
    cnp.ndarray[cnp.int64_t, ndim=2] faces,
    cnp.ndarray[cnp.float64_t, ndim=2] vertices,
    cnp.ndarray[cnp.int64_t, ndim=2] vertex_faces,
    cnp.ndarray[cnp.float64_t, ndim=2] triangles_center,
    *,
    double shift_distance = 0.0,
) -> cnp.ndarray[cnp.float64_t]:
    """Finds and repairs non-manifold vertices.

    Non-manifold vertices are defined as as vertices where more than one
    contiguous group of faces originates.

    This algorithm uses a two-step process for performance. The first step is
    to find the number of new vertices to split, then the new vertices array is
    initialized. The second step creates the vertex splits, filling the new
    vertices array and reassigning faces.
    """
    # Initialize variables
    # Loop index variables
    cdef Py_ssize_t vertex, i, j, k
    # The number of input vertices
    cdef Py_ssize_t num_vertices = vertex_faces.shape[0]
    # The maximum number of faces for the input vertices
    cdef Py_ssize_t num_vertex_faces = vertex_faces.shape[1]
    # The current and neighbor face indices
    cdef cnp.int64_t face, neighbor
    # The current and neighbor face vertex indicies
    cdef cnp.int64_t f_v0, f_v1, f_v2, n_v0, n_v1, n_v2

    # An array indicated which faces have been visited for the current vertex
    cdef cnp.uint64_t[::1] visited_faces = (
        np.zeros(num_vertex_faces, dtype=np.uint64)
    )
    # A counter to enable recycling `visited_faces` for each vertex
    cdef cnp.uint64_t stamp = 0

    # A counter for the number of groups found at the current vertex
    cdef cnp.uint64_t groups = 0
    # The number of vertices matched between the current and neighbor face
    # If two vertices match, the test face is a neighbor
    cdef short matched_vertex_count = 0

    # Step 1: Find the number of new vertices to create from splitting
    cdef long num_split_vertices = find_num_split_vertices(
        faces,
        vertex_faces,
        visited_faces,
    )
    # Reset the visited faces array
    for i in range(num_vertex_faces):
        visited_faces[i] = 0

    # The new vertices array created by this algorithm by splitting non-manifold
    # vertices
    cdef cnp.float64_t[:, ::1] new_vertices = (
        np.zeros((num_split_vertices, 3), dtype=np.float64)
    )
    # Keep track of the current vertex split index for face reassignment
    cdef long current_split_vertex_index = 0
    # The current new vertex index counter for face reassignment
    cdef cnp.int64_t new_vertex_index
    # An array to track which faces are part of the current group for face
    # reassignment and shifting
    cdef cnp.uint64_t[::1] group_faces = (
        np.zeros(num_vertex_faces, dtype=np.uint64)
    )
    # The current vertex original location 
    cdef cnp.float64_t v0, v1, v2

    # Shifting specific variables
    # The number of faces in the current group
    cdef cnp.uint64_t num_group_faces
    # The sum of the vertex coordinates, used to find the mean when shifting
    # later
    cdef cnp.float64_t group_sum_v0, group_sum_v1, group_sum_v2
    # The differences from the group faces mean to the original vertex location
    # Used to know which direction to shift
    cdef cnp.float64_t diff_v0, diff_v1, diff_v2
    # The number of visited faces counter for checking whether this is the last
    # group
    cdef cnp.int64_t num_visited_faces

    # Step 2: Create the vertex splits, filling the new vertices array and
    # reassigning faces
    with nogil:
        # For each vertex
        for vertex in range(num_vertices):
            # Increase stamp counter and reset group count for this vertex
            stamp += 1
            groups = 0

            # Store the vertex coordinates
            v0 = vertices[vertex, 0]
            v1 = vertices[vertex, 1]
            v2 = vertices[vertex, 2]

            # For each face at this vertex
            for i in range(num_vertex_faces):
                # Get the global face index
                face = vertex_faces[vertex, i]

                # Skip padding or already processed faces
                if face == -1 or visited_faces[i] == stamp:
                    continue

                # Create a new group
                groups += 1
                for j in range(num_vertex_faces):
                    group_faces[j] = 0

                # Mark face as part of this group
                group_faces[i] = 1

                # Mark face as visited
                visited_faces[i] = stamp

                # Store the initial group shift information
                if shift_distance:
                    num_group_faces = 1
                    group_sum_v0 = triangles_center[face, 0]
                    group_sum_v1 = triangles_center[face, 1]
                    group_sum_v2 = triangles_center[face, 2]

                # If more than one group, we have a non-manifold vertex
                if groups > 1:
                    # We will need to split this vertex for this new group
                    current_split_vertex_index += 1

                # Retrieve the three vertex indicies of the current face
                f_v0 = faces[face, 0]
                f_v1 = faces[face, 1]
                f_v2 = faces[face, 2]

                # Find faces in chain
                for j in range(num_vertex_faces):
                    # Find neighbor face to the current face
                    for k in range(num_vertex_faces):
                        neighbor = vertex_faces[vertex, k]

                        # Skip padding or already processed faces
                        if neighbor == -1 or visited_faces[k] == stamp:
                            continue

                        # Retrieve the 3 vertices of the neighbor face
                        n_v0 = faces[neighbor, 0]
                        n_v1 = faces[neighbor, 1]
                        n_v2 = faces[neighbor, 2]

                        # Count shared vertices
                        matched_vertex_count = (
                            (f_v0 == n_v0) + (f_v0 == n_v1) + (f_v0 == n_v2) + 
                            (f_v1 == n_v0) + (f_v1 == n_v1) + (f_v1 == n_v2) + 
                            (f_v2 == n_v0) + (f_v2 == n_v1) + (f_v2 == n_v2)
                        )

                        # If they share more than one vertex, they are connected
                        if matched_vertex_count > 1:
                            # Add the face shift information
                            if shift_distance:
                                num_group_faces += 1
                                group_sum_v0 += triangles_center[neighbor, 0]
                                group_sum_v1 += triangles_center[neighbor, 1]
                                group_sum_v2 += triangles_center[neighbor, 2]

                            # Mark face as part of this group
                            group_faces[k] = 1

                            # Mark neighbor as visited
                            visited_faces[k] = stamp

                            # Change current face to neighbor
                            face = neighbor
                            f_v0 = n_v0
                            f_v1 = n_v1
                            f_v2 = n_v2

                            # Break out of this inner loop to begin searching
                            # through the faces again for the new neighbor
                            break

                    # Finish chain if no more connected faces
                    if matched_vertex_count < 2 or k == num_vertex_faces - 1:
                        # If more than one group, we have a non-manifold vertex,
                        # split
                        if groups > 1:
                            # Create a new vertex
                            new_vertices[current_split_vertex_index - 1, 0] = v0
                            new_vertices[current_split_vertex_index - 1, 1] = v1
                            new_vertices[current_split_vertex_index - 1, 2] = v2
                            new_vertex_index = (
                                num_vertices + current_split_vertex_index - 1
                            )

                            # Update all faces in this group to use the new
                            # vertex
                            for k in range(num_vertex_faces):
                                if group_faces[k]:
                                    neighbor = vertex_faces[vertex, k]
                                    # Update the face to use the new vertex
                                    if faces[neighbor, 0] == vertex:
                                        faces[neighbor, 0] = new_vertex_index
                                    elif faces[neighbor, 1] == vertex:
                                        faces[neighbor, 1] = new_vertex_index
                                    elif faces[neighbor, 2] == vertex:
                                        faces[neighbor, 2] = new_vertex_index

                        # Shift the new vertex
                        if shift_distance:
                            # Get the shift differences
                            diff_v0 = group_sum_v0 / num_group_faces - v0
                            diff_v1 = group_sum_v1 / num_group_faces - v1
                            diff_v2 = group_sum_v2 / num_group_faces - v2
                            # If this is the first group, update the original
                            # vertex
                            if groups == 1:
                                # Only update if not all faces are in this group
                                # (aka, this is a manifold vertex)
                                num_visited_faces = 0
                                for k in range(num_vertex_faces):
                                    if (
                                        vertex_faces[vertex, k] == -1
                                        or visited_faces[k] == stamp
                                    ):
                                        num_visited_faces += 1

                                if num_visited_faces < num_vertex_faces:
                                    vertices[vertex, 0] += shift_distance * (
                                        1 if diff_v0 > 0 else -1 if diff_v0 < 0 else 0
                                    )
                                    vertices[vertex, 1] += shift_distance * (
                                        1 if diff_v1 > 0 else -1 if diff_v1 < 0 else 0
                                    )
                                    vertices[vertex, 2] += shift_distance * (
                                        1 if diff_v2 > 0 else -1 if diff_v2 < 0 else 0
                                    )
                            # Otherwise, update the new vertex
                            else:
                                new_vertices[current_split_vertex_index - 1, 0] += (
                                    shift_distance
                                    * (1 if diff_v0 > 0 else -1 if diff_v0 < 0 else 0)
                                )
                                new_vertices[current_split_vertex_index - 1, 1] += (
                                    shift_distance
                                    * (1 if diff_v1 > 0 else -1 if diff_v1 < 0 else 0)
                                )
                                new_vertices[current_split_vertex_index - 1, 2] += (
                                    shift_distance
                                    * (1 if diff_v2 > 0 else -1 if diff_v2 < 0 else 0)
                                )

                        # Chain is finished, search for next unvisited face
                        break

    # Return the updated vertices array
    return np.vstack([vertices, np.asarray(new_vertices, dtype=vertices.dtype)])
