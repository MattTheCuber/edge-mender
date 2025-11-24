import numpy as np
import pytest
from numpy.typing import NDArray

from edge_mender.data_factory import DataFactory


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
        DataFactory.hole(),
        DataFactory.kill_you(),
        DataFactory.random(seed=0),
    ],
)
def test_data_factory(data: NDArray) -> None:
    """Test that the data factory creates the expected data."""
    assert data is not None
    assert isinstance(data, np.ndarray)
    assert data.ndim == 3  # noqa: PLR2004
    assert np.any(data == 1)
