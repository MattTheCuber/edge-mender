# cython: language_level=3
import numpy as np
cimport numpy as cnp
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)   
def repair_vertices(
    cnp.ndarray[cnp.int64_t, ndim=2] faces,
    cnp.ndarray[cnp.float64_t, ndim=2] vertices,
    cnp.ndarray[cnp.int64_t, ndim=2] vertex_faces,
    cnp.ndarray[cnp.float64_t, ndim=2] triangles_center,
    *,
    float shift_distance = 0.0,
    bool debug = False,
) -> np.ndarray:
    new_vertices: list = []

    for vertex, raw_faces in enumerate(vertex_faces):
        current_faces: list[int] = raw_faces[raw_faces != -1].tolist()
        if debug:
            print(f"Analyzing vertex {vertex} with faces {current_faces}")
        face_groups: list[list[int]] = []

        for _ in range(len(current_faces)):
            if not current_faces:
                break

            face = current_faces.pop()
            face_group = [face]
            if debug:
                print(f"  Started new face group with face {face}")

            for _ in range(len(current_faces)):
                neighbor = np.where(
                    np.isin(faces[current_faces], faces[face]).sum(axis=1) > 1,
                )[0]
                if len(neighbor) == 0:
                    break

                face = current_faces.pop(neighbor[0])
                face_group.append(face)
                if debug:
                    print(f"    Added neighbor face {face} to group")

            face_groups.append(face_group)

        if len(face_groups) > 1:
            if debug:
                print(f"  Vertex {vertex} has {len(face_groups)} face groups, fixing...")
                
                print(f"    Keeping vertex {vertex} for face group {face_groups[0]}")

            for face_group in face_groups[1:]:
                new_vertex = vertices[vertex].copy()
                new_vertices.append(new_vertex)
                if shift_distance:
                    mean_position = np.mean(triangles_center[face_group], axis=0)
                    new_vertex[:] += shift_distance * np.sign(
                        mean_position - vertices[vertex],
                    )
                new_vertex_index = len(vertices) + len(new_vertices) - 1
                for face in face_group:
                    faces[face][faces[face] == vertex] = new_vertex_index
                if debug:
                    print(f"    Created new vertex {new_vertex_index} for face group {face_group}")

            if shift_distance:
                mean_position = np.mean(triangles_center[face_groups[0]], axis=0)
                vertices[vertex] += shift_distance * np.sign(
                    mean_position - vertices[vertex],
                )

    return np.vstack([vertices, new_vertices])
