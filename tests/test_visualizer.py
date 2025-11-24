"""Test the Visualizer class."""

import pyvista as pv
import trimesh
from pytest_mock import MockerFixture

from edge_mender.visualizer import Visualizer


def test_show_mesh(mocker: MockerFixture) -> None:
    """Very basic test the Visualizer class."""
    mocker.patch.object(pv.Plotter, "show")
    mesh = trimesh.creation.box()
    Visualizer.show_mesh(mesh)
    Visualizer.show_mesh(
        mesh,
        add_face_labels=True,
        add_vertex_labels=True,
        add_edge_labels=True,
    )
    Visualizer.show_mesh(
        mesh,
        highlight_faces=[0, 1],
        highlight_vertices=[0, 1],
        highlight_edges=[[0, 1]],
        add_face_normals=True,
        add_face_labels=True,
        add_vertex_labels=True,
        add_edge_labels=True,
        edge_labels=["1"],
    )
    Visualizer.show_mesh(
        mesh,
        highlight_vertices=[0, 1],
        add_face_normals=True,
        add_vertex_labels=False,
        add_edge_labels=True,
    )
    Visualizer.show_mesh(
        mesh,
        highlight_faces=[0, 1],
        highlight_edges=[[0, 1]],
        add_vertex_labels=False,
    )
