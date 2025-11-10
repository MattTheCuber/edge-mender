import numpy as np
import pyvista as pv
import trimesh


def to_mesh_surface_nets(data: np.ndarray) -> trimesh.Trimesh:
    # Convert a Numpy array to a mesh using Surface Nets from Pyvista/VTK
    pv_data: pv.ImageData = pv.wrap(data)
    mesh = pv_data.contour_labels(output_mesh_type="triangles", smoothing=False)
    faces = mesh.faces.reshape((mesh.n_cells, 4))[:, 1:]
    mesh = trimesh.Trimesh(mesh.points, faces)
    mesh.fix_normals()
    if mesh.volume < 0:
        mesh.invert()
    return mesh
