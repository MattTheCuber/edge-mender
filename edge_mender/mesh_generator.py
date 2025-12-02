"""Module for generating meshes from 3D numpy arrays using different algorithms."""

import sys
from pathlib import Path

import itk
import itk.itkArrayPython
import itk.itkMeshBasePython
import itk.itkPointPython
import itk.itkVectorContainerPython
import numpy as np
import panda3d.core as pd
import pyvista as pv
import trimesh
from numpy.typing import NDArray


class MeshGenerator:
    """Class for generating meshes from 3D numpy arrays using different algorithms."""

    @staticmethod
    def to_mesh_cuberille(data: NDArray) -> trimesh.Trimesh:
        """Convert a Numpy array to a mesh using Cuberille from ITK."""
        # Implicitly load ITKCommon module
        if "itk.CuberillePython" not in sys.modules:
            print("Loading ITK Cuberille module. This will take a while...")  # noqa: T201

            itk.Image  # pyright: ignore[reportAttributeAccessIssue]  # noqa: B018

            from itk.CuberillePython import (  # noqa: PLC0415
                cuberille_image_to_mesh_filter,
            )

            print("ITK Cuberille module loaded.")  # noqa: T201

        from itk.CuberillePython import cuberille_image_to_mesh_filter  # noqa: PLC0415

        # Generate the mesh using ITK's Cuberille implementation
        itk_mesh: itk.itkMeshBasePython.itkMeshD3 = cuberille_image_to_mesh_filter(
            data,
            project_vertices_to_iso_surface=False,
        )

        # Extract the vertices
        points: itk.itkVectorContainerPython.itkVectorContainerULLPF3 = (
            itk_mesh.GetPoints()
        )
        n_points = points.Size()
        vertices = np.zeros((n_points, 3), dtype=float)
        for i in range(n_points):
            p: itk.itkPointPython.itkPointF3 = points.GetElement(i)
            vertices[i] = [p[0], p[1], p[2]]

        # Extract the faces
        cells: itk.itkMeshBasePython.itkVectorContainerULLCIDCTI3FFULLULLULLPF3VCULLPF3 = itk_mesh.GetCells()  # noqa: E501 - no choice
        n_cells = cells.Size()
        faces = np.zeros((n_cells, 3), dtype=float)
        for i in range(n_cells):
            cell: itk.itkMeshBasePython.itkCellInterfaceDCTI3FFULLULLULLPF3VCULLPF3 = (
                cells.GetElement(i)
            )
            cell_points: itk.itkArrayPython.itkArrayULL = cell.GetPointIdsContainer()
            faces[i] = cell_points

        # Create the mesh using Trimesh
        return trimesh.Trimesh(vertices=vertices, faces=faces)

    @staticmethod
    def to_mesh_surface_nets(data: NDArray) -> trimesh.Trimesh:
        """Convert a Numpy array to a mesh using Surface Nets from PyVista/VTK."""
        pv_data: pv.ImageData = pv.wrap(data)  # pyright: ignore[reportAssignmentType]
        mesh = pv_data.contour_labels(output_mesh_type="triangles", smoothing=False)
        faces = mesh.faces.reshape((mesh.n_cells, 4))[:, 1:]
        mesh = trimesh.Trimesh(mesh.points, faces)
        # TODO: This shouldn't be needed after https://gitlab.kitware.com/vtk/vtk/-/issues/19156
        mesh.fix_normals()
        if mesh.volume < 0:
            mesh.invert()
        return mesh

    @staticmethod
    def to_mesh_dual_contouring(data: NDArray) -> trimesh.Trimesh:
        """Convert a Numpy array to a mesh using Dual Contouring from Daniel Wilmes."""
        # Add the submodule to the path so we can import it
        project_root = Path(__file__).parent / "Dual_Contouring_Voxel"
        if not project_root.exists():  # pragma: no cover
            missing_module_error = (
                "Could not find Dual Contouring module. "
                "Perhaps you forgot to run `git submodule update --init`?"
            )
            raise FileNotFoundError(missing_module_error)
        sys.path.append(str(project_root))

        # Remove the app code from the Dual Contouring module
        dual_contouring_py = project_root / "Dual_Contouring.py"
        lines = dual_contouring_py.read_text().splitlines()
        try:
            idx = next(i for i, line in enumerate(lines) if line == "app = myapp()")
        except StopIteration:
            idx = None
        if idx is not None:  # pragma: no cover
            new_content = "\n".join(lines[:idx])
            dual_contouring_py.write_text(new_content + ("\n" if new_content else ""))

        from edge_mender.Dual_Contouring_Voxel.Dual_Contouring import (  # noqa: PLC0415
            dual_contouring,
        )
        from edge_mender.Dual_Contouring_Voxel.Voxel_Functions import (  # noqa: PLC0415
            make_mesh,
        )

        # Run dual contouring to generate the mesh
        gformat = pd.GeomVertexFormat.getV3cpt2()
        vdata = pd.GeomVertexData("Triangle", gformat, pd.Geom.UHStatic)
        faces, _ = dual_contouring(
            lambda x, y, z: bool(data[x, y, z]),
            xmin=0,
            xmax=data.shape[0] - 1,
            ymin=0,
            ymax=data.shape[1] - 1,
            zmin=0,
            zmax=data.shape[2] - 1,
            vdata=vdata,
            stepsize=1,
        )
        node = make_mesh(vdata, faces)

        # Convert to trimesh - very annoying
        geom = node.getGeom(0)
        vr = pd.GeomVertexReader(geom.getVertexData(), "vertex")
        vertices = [
            (v.x, v.y, v.z)
            for v in iter(lambda: vr.getData3f() if not vr.isAtEnd() else None, None)
        ]
        faces = []
        for i in range(geom.getNumPrimitives()):
            prim = geom.getPrimitive(i).decompose()
            for p in range(prim.getNumPrimitives()):
                s, e = prim.getPrimitiveStart(p), prim.getPrimitiveEnd(p)
                faces.append([prim.getVertex(i) for i in range(s, e)])

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.fix_normals()
        return mesh
