"""Module for visualizing meshes using PyVista."""

from typing import Literal

import numpy as np
import pyvista as pv
import trimesh


class Visualizer:
    """Class for visualizing meshes using PyVista."""

    @staticmethod
    def show_mesh(
        input_mesh: trimesh.Trimesh | pv.PolyData,
        *,
        highlight_faces: list[int] | None = None,
        highlight_vertices: list[int] | None = None,
        highlight_edges: list[list[int]] | None = None,
        add_face_normals: bool = False,
        add_face_labels: bool = False,
        add_vertex_labels: bool = False,
        add_edge_labels: bool = False,
        edge_labels: list[str | int] | None = None,
        opacity: float = 1.0,
        style: Literal["points", "wireframe", "surface"] = "surface",
    ) -> pv.Plotter:
        """Visualize a mesh with optional highlights and labels.

        Parameters
        ----------
        input_mesh : trimesh.Trimesh | pv.PolyData
            The mesh to visualize.
        highlight_faces : list[int] | None, optional
            A list of face indices to highlight, by default None
        highlight_vertices : list[int] | None, optional
            A list of vertex indices to highlight, by default None
        highlight_edges : list[list[int]] | None, optional
            A list of edges to highlight, by default None
        add_face_normals : bool, optional
            Whether to add face normal arrows, by default False
        add_face_labels : bool, optional
            Whether to add labels to faces, by default False
        add_vertex_labels : bool, optional
            Whether to add labels to vertices, by default True
        add_edge_labels : bool, optional
            Whether to add labels to edges, by default False
        edge_labels : list[str | int] | None, optional
            The labels to use for the edges, by default None
        opacity : float, optional
            The opacity of the mesh, by default 1.0
        style : Literal["points", "wireframe", "surface"], optional
            The style of the mesh, by default "surface"

        Returns
        -------
        pv.Plotter
            The PyVista plotter object used for visualization.

        References
        ----------
        .. [1] Sullivan, C., & Kaszynski, A. (2019). PyVista: 3D plotting and mesh
           analysis through a streamlined interface for the Visualization Toolkit (VTK).
           Journal of Open Source Software, 4(37), 1450. https://doi.org/10.21105/joss.01450
        .. [2] https://docs.pyvista.org/api/plotting
        """
        mesh: pv.PolyData = pv.wrap(input_mesh)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, show_edges=True, style=style, opacity=opacity)

        if add_face_normals:
            normals_mesh = mesh.compute_normals(
                cell_normals=True,
                point_normals=False,
                consistent_normals=False,
            )
            if highlight_faces:
                unique_faces = np.unique(highlight_faces)
                normals_mesh = normals_mesh.extract_cells(unique_faces)
            normals_glyphs: pv.PolyData = normals_mesh.glyph(
                orient="Normals",
                scale=False,
                factor=0.2,
            )  # pyright: ignore[reportAssignmentType]
            plotter.add_mesh(normals_glyphs, color="#FF5555", point_size=0)

        if highlight_faces:
            unique_faces = np.unique(highlight_faces)
            highlighted_faces: pv.PolyData = mesh.extract_cells(unique_faces)  # pyright: ignore[reportAssignmentType]
            plotter.add_mesh(highlighted_faces, color="lightgreen", show_edges=True)
            if add_face_labels:
                plotter.add_point_labels(
                    highlighted_faces.cell_centers().points,
                    unique_faces.tolist(),
                    point_size=0,
                    text_color="#B30909",
                )
        elif add_face_labels:
            plotter.add_point_labels(
                mesh.cell_centers().points,
                list(range(mesh.n_cells)),
                point_size=0,
                text_color="#B30909",
            )

        if highlight_vertices:
            unique_vertices = np.unique(highlight_vertices)
            hightlight_points = mesh.points[unique_vertices]
            if add_vertex_labels:
                plotter.add_point_labels(
                    hightlight_points,
                    [
                        f"{v}: {p}"
                        for v, p in zip(unique_vertices, hightlight_points, strict=True)
                    ],
                    render_points_as_spheres=True,
                    point_color="#00FFFF",
                    point_size=5,
                    text_color="#33FFFF",
                )
            else:
                plotter.add_points(
                    hightlight_points,
                    render_points_as_spheres=True,
                    color="#00FFFF",
                    point_size=5,
                )
        elif add_vertex_labels:
            plotter.add_point_labels(
                mesh.points,
                [f"{v}: {p}" for v, p in enumerate(mesh.points)],
                point_size=0,
                text_color="#33FFFF",
            )

        if highlight_edges:
            unique_edges = np.unique(highlight_edges, axis=1)
            cells = np.hstack([[2, *e] for e in unique_edges])
            edge_mesh = pv.UnstructuredGrid(
                cells,
                [pv.CellType.LINE] * len(unique_edges),
                mesh.points,
            )
            plotter.add_mesh(
                edge_mesh,
                render_lines_as_tubes=True,
                color="#FFD700",
                line_width=2.5,
            )
            if add_edge_labels:
                plotter.add_point_labels(
                    edge_mesh.cell_centers().points,
                    edge_labels or [f"v{e[0]} - v{e[1]}" for e in unique_edges],
                    point_size=0,
                    text_color="#FFF11A",
                )
        elif add_edge_labels:
            edges = mesh.extract_all_edges()
            lines = edges.lines.reshape(-1, 3)[:, 1:]
            plotter.add_point_labels(
                np.mean(edges.points[lines], axis=1),
                [f"v{v1} - v{v2}" for v1, v2 in lines],
                point_size=0,
                text_color="#FFF11A",
            )

        plotter.show()
        return plotter
