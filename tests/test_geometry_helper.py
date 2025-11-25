"""Test the GeometryHelper class."""

import numpy as np
import pytest

from edge_mender.geometry_helper import GeometryHelper


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
    ("point", "ray_point", "ray_dir", "expected_angle"),
    [
        ([1, 0, 0], [0, 0, 0], [1, 0, 0], 0),
        ([1, 0, 0], [0, 0, 0], [2, 0, 0], 0),
        ([1, 1, 0], [0, 0, 0], [1, 0, 0], 45),
        ([2, 2, 0], [0, 0, 0], [1, 0, 0], 45),
        ([1, -1, 0], [0, 0, 0], [1, 0, 0], 45),
        ([0, 1, 0], [0, 0, 0], [1, 0, 0], 90),
        ([1, 1, 0], [0, 0, 0], [-1, 0, 0], 135),
        ([0, 1, 1], [0, 0, 0], [0, -1, 0], 135),
        ([-1, -1, -1], [0, 0, 0], [1, 1, 1], 180),
        ([0, 1, 1], [1, 1, 1], [1, 0, 0], 180),
    ],
)
def test_angle_between_point_and_ray(
    point: list[int],
    ray_point: list[int],
    ray_dir: list[int],
    expected_angle: int,
) -> None:
    """Test GeometryHelper.angle_between_point_and_ray."""
    angle = GeometryHelper.angle_between_point_and_ray(
        point=np.array(point),
        ray_point=np.array(ray_point),
        ray_dir=np.array(ray_dir),
    )
    assert np.isclose(np.rad2deg(angle), expected_angle)


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
