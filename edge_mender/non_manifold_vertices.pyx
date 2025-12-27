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
    float shift_distance = 0.0,
    bool debug = False,
) -> np.ndarray:
    # Build a list of new vertices to add
    new_vertices: list = []

    # For each vertex
    for vertex, raw_faces in enumerate(vertex_faces):
        # Get the faces that use this vertex
        current_faces: np.ndarray = raw_faces[raw_faces != -1]
        if debug:
            print(f"Analyzing vertex {vertex} with faces {current_faces.tolist()}")

        # Intialize the chains of connected faces
        face_groups: list[list[int]] = []

        # While there are still faces to process
        for _ in range(len(current_faces)):
            if not current_faces.size:
                break

            # Start a new face group
            face = current_faces[0]
            current_faces = current_faces[1:]
            face_group = [face]
            face_groups.append(face_group)
            if debug:
                print(f"  Started new face group with face {face}")

            # Find all connected faces in the chain
            for _ in range(len(current_faces)):
                matched = None
                for curr in range(len(current_faces)):
                    curr_face = faces[current_faces[curr]]
                    count = 0 
                    for v in faces[face]:
                        if v == curr_face[0] or v == curr_face[1] or v == curr_face[2]:
                            count += 1
                    if count > 1:
                        matched = curr
                        break
                if matched is None:
                    break

                face = current_faces[matched]
                current_faces = current_faces[current_faces != face]
                face_group.append(face)
                if debug:
                    print(f"    Added neighbor face {face} to group")

        # If there are multiple face groups, we need to split
        if len(face_groups) > 1:
            if debug:
                print(f"  Vertex {vertex} has {len(face_groups)} face groups, fixing...")
                
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
                    faces[face][faces[face] == vertex] = new_vertex_index

                if debug:
                    print(f"    Created new vertex {new_vertex_index} for face group {face_group}")

            # Optionally shift the original vertex
            if shift_distance:
                mean_position = np.mean(triangles_center[face_groups[0]], axis=0)
                vertices[vertex] += shift_distance * np.sign(
                    mean_position - vertices[vertex],
                )

    # Return the updated vertices array
    return np.vstack([vertices, np.asarray(new_vertices, dtype=vertices.dtype)])
