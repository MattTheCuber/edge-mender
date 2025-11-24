"""Test private functions in the EdgeMender class."""

import trimesh

from edge_mender.edge_mender import EdgeMender


def test_edge_mender_init() -> None:
    """Test that the EdgeMender class can be initialized."""
    mesh = trimesh.creation.box()
    EdgeMender(mesh)
    EdgeMender(mesh, debug=True)
