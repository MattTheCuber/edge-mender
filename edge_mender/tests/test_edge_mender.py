import pytest
import trimesh

from edge_mender.data_factory import DataFactory
from edge_mender.edge_mender import EdgeMender
from edge_mender.mesh_generator import MeshGenerator


@pytest.mark.parametrize("spacing", [[1, 1, 1], [1.25, 0.5, 0.25]])
@pytest.mark.parametrize(
    "data",
    [
        DataFactory.simple_extrusion(),
        DataFactory.double_extrusion(),
        DataFactory.triple_extrusion(),
        DataFactory.stairs(),
        DataFactory.ceiling(),
        DataFactory.double_tower_ceiling(),
        DataFactory.hanging_points(),
        DataFactory.checkerboard(),
    ],
)
def test_validate(data, spacing):
    mesh = MeshGenerator.to_mesh_surface_nets(data)
    mesh.vertices *= spacing
    mender = EdgeMender(mesh)
    mender.validate(spacing=spacing)


def test_validate_fail_normals():
    # Pyramid with non-axis-aligned face normals
    mesh = trimesh.creation.cone(1, 1, sections=3)
    mender = EdgeMender(mesh)
    with pytest.raises(ValueError, match="non-axis-aligned face normals"):
        mender.validate(spacing=(1, 1, 1))


def test_validate_fail_angles():
    mesh = trimesh.creation.box()
    # Stretch the box to create bad angles
    mesh.vertices *= [1, 1, 1.25]
    mender = EdgeMender(mesh)
    with pytest.raises(ValueError, match="bad angles"):
        mender.validate(spacing=(1, 1, 1))


def test_validate_fail_areas():
    # Subdivide everything except one face to make the faces larger
    mesh = trimesh.creation.box().subdivide(list(range(10)))
    mender = EdgeMender(mesh)
    with pytest.raises(ValueError, match="non-uniform face areas"):
        mender.validate(spacing=(1, 1, 1))
