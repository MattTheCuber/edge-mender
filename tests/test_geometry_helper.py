"""Test the GeometryHelper class."""

import numpy as np
import pytest

from edge_mender.geometry_helper import GeometryHelper


@pytest.mark.parametrize(
    ("direction", "test_directions", "expected"),
    [
        ([1, 0, 0], [[1, 0, 0], [0, 1, 0]], True),
        ([1, 0, 0], [[-1, 0, 0], [0, 1, 0]], False),
        ([1, 0, 0], [[0, 1, 0], [0, 0, 1]], False),
    ],
)
def test_any_directions_match(
    direction: list[int],
    test_directions: list[list[int]],
    *,
    expected: bool,
) -> None:
    """Test GeometryHelper.any_directions_match."""
    result = GeometryHelper.any_directions_match(
        direction=np.array(direction),
        test_directions=np.array(test_directions),
    )
    assert result == expected


@pytest.mark.parametrize(
    ("direction", "expected_a", "expected_b"),
    [
        ([1, 0, 0], [0, 1, 1], [0, -1, 1]),
        ([0, 1, 0], [1, 0, 1], [-1, 0, 1]),
        ([0, 0, 1], [1, 1, 0], [-1, 1, 0]),
    ],
)
def test_diagonal_orthogonal_directions(
    direction: list[int],
    expected_a: list[int],
    expected_b: list[int],
) -> None:
    """Test GeometryHelper.get_diagonal_orthogonal_directions."""
    a, b = GeometryHelper.get_diagonal_orthogonal_directions(np.array(direction))
    assert a.tolist() == expected_a
    assert b.tolist() == expected_b


@pytest.mark.parametrize(
    ("line_point", "line_direction", "test_point", "expected"),
    [
        ([0, 0], [1, 0], [0, 1], True),
        ([0, 0], [1, 0], [0, -1], False),
        ([0, 0], [0, 1], [1, 0], False),
        ([0, 0], [0, 1], [-1, 0], True),
        ([0, 0], [1, 0], [1, 1], True),
        ([0, 0], [1, 0], [1, -1], False),
        ([0, 0], [1, 0], [2, 0], "Point is on the line"),
    ],
)
def test_is_left(
    line_point: list[int],
    line_direction: list[int],
    test_point: list[int],
    *,
    expected: bool | str,
) -> None:
    """Test GeometryHelper.is_left."""
    if isinstance(expected, bool):
        assert (
            GeometryHelper.is_left(
                line_point=np.array(line_point),
                line_direction=np.array(line_direction),
                test_point=np.array(test_point),
            )
            == expected
        )
    else:
        with pytest.raises(ValueError, match=expected):
            GeometryHelper.is_left(
                line_point=np.array(line_point),
                line_direction=np.array(line_direction),
                test_point=np.array(test_point),
            )


@pytest.mark.parametrize(
    ("point", "ray_origin", "ray_direction", "expected"),
    [
        ([1, 0, 0], [0, 0, 0], [1, 0, 0], True),
        ([1, 0, 0], [0, 0, 0], [2, 0, 0], True),
        ([1, 1, 0], [0, 0, 0], [1, 0, 0], True),
        ([2, 2, 0], [0, 0, 0], [1, 0, 0], True),
        ([1, -1, 0], [0, 0, 0], [1, 0, 0], True),
        ([1, 1, 0], [0, 0, 0], [-1, 0, 0], False),
        ([0, 1, 1], [0, 0, 0], [0, -1, 0], False),
        ([-1, -1, -1], [0, 0, 0], [1, 1, 1], False),
        ([0, 1, 1], [1, 1, 1], [1, 0, 0], False),
    ],
)
def test_point_in_direction(
    point: list[int],
    ray_origin: list[int],
    ray_direction: list[int],
    *,
    expected: bool,
) -> None:
    """Test GeometryHelper.point_in_direction."""
    matches_direction = GeometryHelper.point_in_direction(
        ray_origin=np.array(ray_origin),
        ray_direction=np.array(ray_direction),
        test_point=np.array(point),
    )
    assert matches_direction == expected


def test_point_in_direction_fails() -> None:
    """Test GeometryHelper.point_in_direction."""
    with pytest.raises(ValueError, match="Point is equidistant"):
        GeometryHelper.point_in_direction(
            ray_origin=np.array([0, 0, 0]),
            ray_direction=np.array([1, 0, 0]),
            test_point=np.array([0, 1, 0]),
        )


@pytest.mark.parametrize(
    ("point_1", "normal_1", "point_2", "normal_2", "expected"),
    [
        ([0, 0], [1, 0], [1, 1], [0, -1], True),
        ([0, 0], [1, 0], [1, 1], [0, 1], False),
        ([0, 0], [1, 0], [1, 1], [1, 0], "Parallel"),
        ([0, 0], [1, 0], [0, 0], [1, 0], "Colinear"),
        ([0, 0], [1, 0], [1, 0], [-1, 0], "Colinear"),
    ],
)
def test_rays_intersect(
    point_1: list[int],
    normal_1: list[int],
    point_2: list[int],
    normal_2: list[int],
    *,
    expected: bool | str,
) -> None:
    """Test GeometryHelper.rays_intersect."""
    if isinstance(expected, bool):
        assert (
            GeometryHelper.rays_intersect(
                point_a=np.array(point_1),
                normal_a=np.array(normal_1),
                point_b=np.array(point_2),
                normal_b=np.array(normal_2),
            )
            == expected
        )
    else:
        with pytest.raises(ValueError, match=expected):
            GeometryHelper.rays_intersect(
                point_a=np.array(point_1),
                normal_a=np.array(normal_1),
                point_b=np.array(point_2),
                normal_b=np.array(normal_2),
            )
