import os

import itk
import itk.itkArrayPython
import itk.itkMeshBasePython
import itk.itkPointPython
import itk.itkVectorContainerPython
import numpy as np
import pyvista as pv
import trimesh
from numpy.typing import NDArray

itk.Image  # implicitly load ITKCommon module
from itk.CuberillePython import cuberille_image_to_mesh_filter

POINTS_TYPE = itk.itkVectorContainerPython.itkVectorContainerULLPF3
CELLS_TYPE = itk.itkMeshBasePython.itkVectorContainerULLCIDCTI3FFULLULLULLPF3VCULLPF3
CELL_TYPE = itk.itkMeshBasePython.itkCellInterfaceDCTI3FFULLULLULLPF3VCULLPF3


class MeshGenerator:
    @staticmethod
    def to_mesh_surface_nets(data: NDArray) -> trimesh.Trimesh:
        """Convert a Numpy array to a mesh using Surface Nets from PyVista/VTK."""
        pv_data: pv.ImageData = pv.wrap(data)
        mesh = pv_data.contour_labels(output_mesh_type="triangles", smoothing=False)
        faces = mesh.faces.reshape((mesh.n_cells, 4))[:, 1:]
        mesh = trimesh.Trimesh(mesh.points, faces)
        mesh.fix_normals()
        if mesh.volume < 0:
            mesh.invert()
        return mesh

    @staticmethod
    def to_mesh_cuberille(data: NDArray) -> trimesh.Trimesh:
        # Generate the mesh using ITK's Cuberille implementation
        itk_mesh: itk.itkMeshBasePython.itkMeshD3 = cuberille_image_to_mesh_filter(
            data,
            project_vertices_to_iso_surface=False,
        )

        # Extract the vertices
        points: POINTS_TYPE = itk_mesh.GetPoints()
        n_points = points.Size()
        vertices = np.zeros((n_points, 3), dtype=float)
        for i in range(n_points):
            p: itk.itkPointPython.itkPointF3 = points.GetElement(i)
            vertices[i] = [p[0], p[1], p[2]]

        # Extract the faces
        cells: CELLS_TYPE = itk_mesh.GetCells()
        n_cells = cells.Size()
        faces = np.zeros((n_cells, 3), dtype=float)
        for i in range(n_cells):
            cell: CELL_TYPE = cells.GetElement(i)
            cell_points: itk.itkArrayPython.itkArrayULL = cell.GetPointIdsContainer()
            faces[i] = cell_points

        # Create the mesh using Trimesh
        return trimesh.Trimesh(vertices=vertices, faces=faces)
