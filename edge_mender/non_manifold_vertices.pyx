# cython: language_level=3
import numpy as np
cimport numpy as cnp
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def repair_vertices(
    cnp.ndarray[cnp.int64_t, ndim=2] faces,
    cnp.ndarray[cnp.float64_t, ndim=2] vertices,
    cnp.ndarray[cnp.int64_t, ndim=2] vertex_faces,
    cnp.ndarray[cnp.float64_t, ndim=2] triangles_center,
    *,
    double shift_distance = 0.0,
    bint debug = False,
) -> np.ndarray:
    # Initialize variables
    cdef Py_ssize_t vertex, i, j
    cdef int face, test_face, matched_idx
    cdef int face_v0, face_v1, face_v2, test_v0, test_v1, test_v2

    # Build a list of new vertices to add
    new_vertices: list = []

    # For each vertex
    for vertex in range(vertex_faces.shape[0]):
        # Get the faces that use this vertex
        current_faces = []
        for i in range(vertex_faces.shape[1]):
            face = vertex_faces[vertex, i]
            if face != -1:
                current_faces.append(face)

        if debug:
            print(f"Analyzing vertex {vertex} with faces {current_faces}")

        # Intialize the chains of connected faces
        face_groups: list[list[int]] = []

        # While there are still faces to process
        while current_faces:
            # Start a new face group
            face = current_faces.pop()
            group = [face]
            face_groups.append(group)
            if debug:
                print(f"  Started new face group with face {face}")

            # Find all connected faces in the chain
            while True:
                # Retrieve the 3 vertices of the current face
                face_v0 = faces[face, 0]
                face_v1 = faces[face, 1]
                face_v2 = faces[face, 2]

                # Find a neighboring face
                matched_idx = -1
                for j in range(len(current_faces)):
                    test_face = current_faces[j]
                    test_v0 = faces[test_face, 0]
                    test_v1 = faces[test_face, 1]
                    test_v2 = faces[test_face, 2]

                    # Count shared vertices
                    count = 0
                    if face_v0 == test_v0 or face_v0 == test_v1 or face_v0 == test_v2:
                        count += 1
                    if face_v1 == test_v0 or face_v1 == test_v1 or face_v1 == test_v2:
                        count += 1
                    if face_v2 == test_v0 or face_v2 == test_v1 or face_v2 == test_v2:
                        count += 1

                    # If they share more than one vertex, they are connected
                    if count > 1:
                        matched_idx = j
                        break

                if matched_idx < 0:
                    break

                face = current_faces.pop(matched_idx)
                group.append(face)
                if debug:
                    print(f"    Added neighbor face {face} to group")

        # If there are multiple face groups, we need to split
        if len(face_groups) > 1:
            if debug:
                print(
                    f"  Vertex {vertex} has {len(face_groups)} face groups, fixing...",
                )

                print(f"    Keeping vertex {vertex} for face group {face_groups[0]}")

            # For each additional face group
            for face_group in face_groups[1:]:
                # Create a new vertex
                new_vertex = vertices[vertex].copy()
                new_vertices.append(new_vertex)
                new_vertex_index = len(vertices) + len(new_vertices) - 1

                # Optionally shift the new vertex
                if shift_distance:
                    mean_position = np.mean(triangles_center[face_group], axis=0)
                    new_vertex[:] += shift_distance * np.sign(
                        mean_position - vertices[vertex],
                    )

                # Update the faces to use the new vertex
                for face in face_group:
                    if faces[face, 0] == vertex:
                        faces[face, 0] = new_vertex_index
                    elif faces[face, 1] == vertex:
                        faces[face, 1] = new_vertex_index
                    elif faces[face, 2] == vertex:
                        faces[face, 2] = new_vertex_index

                if debug:
                    print(
                        f"    Created new vertex {new_vertex_index} for face group {face_group}",
                    )

            # Optionally shift the original vertex
            if shift_distance:
                mean_position = np.mean(triangles_center[face_groups[0]], axis=0)
                vertices[vertex] += shift_distance * np.sign(
                    mean_position - vertices[vertex],
                )

    # Return the updated vertices array
    return np.vstack([vertices, np.asarray(new_vertices, dtype=vertices.dtype)])
