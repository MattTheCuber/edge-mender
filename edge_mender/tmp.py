import fill_voids
import numpy as np
import pyvista as pv
import trimesh


def has_holes(data: np.ndarray) -> bool:
    return not np.array_equal(data, fill_voids.fill(data, in_place=False))


def find_all_non_manifold_edges(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    unique_edges, counts = np.unique(
        mesh.faces_unique_edges.flatten(), return_counts=True
    )
    edges = unique_edges[counts != 2]
    vertices = mesh.edges_unique[edges]
    return vertices, edges


def to_mesh(
    data: np.ndarray,
    *,
    smooth: bool = False,
    fix_normals: bool = True,
    faces_to_flip: list[int] | None = None,
) -> trimesh.Trimesh:
    # Convert a Numpy array to a mesh using Surface Nets from Pyvista/VTK
    pv_data: pv.ImageData = pv.wrap(data)
    mesh = pv_data.contour_labels(output_mesh_type="triangles", smoothing=False)
    faces = mesh.faces.reshape((mesh.n_cells, 4))[:, 1:]
    mesh = trimesh.Trimesh(mesh.points, faces, process=False)
    if faces_to_flip:
        mesh.faces[faces_to_flip] = mesh.faces[faces_to_flip][:, [0, 2, 1]]
    if fix_normals:
        mesh.fix_normals()
        if mesh.volume < 0:
            mesh.invert()
    if smooth:
        trimesh.smoothing.filter_laplacian(mesh)
    return mesh
