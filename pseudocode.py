def repair(mesh):
    # Iterate over the mesh edges
    for edge in mesh:
        # Find non-manifold edges
        faces = find_faces_sharing_edge(edge)
        if len(faces) is 4:
            continue

        for vertex in edge_vertices:
            if vertex not in split_vertices:
                connected_faces = find_faces_connected_to_vertex(vertex)
                unique_normals = get_unique_normals(connected_faces)

                # Ensure vertex splitting would not create a gouge in the floor
                if not any_normal_colinear_with_edge(unique_normals, edge_direction):
                    handle_split_vertex(edge, vertex)

        # Handle edge if neither vertex was split
        if edge_vertices not in split_vertices:
            handle_split_edge(edge)


def handle_split_vertex(edge, vertex):
    # Determine split direction
    split_rays = find_split_rays(edge)

    # Split vertex
    split_vertex(vertex, split_rays)

    # Reconnect faces based on angle criteria
    for face in find_faces_connected_to_edge(edge):
        center = compute_face_center(face)
        best_vertex = find_vertex_with_smallest_angle(center, vertex, split_rays)
        reconnect_face(face, best_vertex)

    mark_vertex_as_split(vertex)


def handle_split_edge(edge):
    # Determine split direction
    split_rays = find_split_rays(edge)

    # Split edge at midpoint
    split_point = average(edge_vertices)
    v1, v2 = insert_split_vertices(split_point, split_rays)

    # Split each face on the edge and reconnect
    for face in find_faces_connected_to_edge(edge):
        split_face_in_half(face)
        center = compute_face_center(face)
        best_vertex = find_vertex_with_smallest_angle(center, (v1, v2), split_rays)
        reconnect_face(face, best_vertex)


def find_split_rays(edge):
    # Find orthogonal axes and diagonal rays
    axis_a, axis_b = find_orthogonal_axes(edge_direction)
    diagonal_rays = compute_diagonal_rays(axis_a, axis_b)

    # Find centers of faces connected to the edge
    connected_faces = find_faces_connected_to_edge(edge)
    centers = compute_face_centers(connected_faces)

    # Group faces by which side of the line they lie on (2D cross product)
    groups = group_faces_by_side(centers, edge_direction)

    # Compute 2D flattened normals for each group
    group_normals = flatten_to_2d(compute_group_normals(groups))

    # Choose the ray pair where the normals intersect
    return select_rays_with_normal_intersection(diagonal_rays, group_normals)
