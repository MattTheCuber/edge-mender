"""Provides a class containing helper functions for geometry calculations."""

import numpy as np
from numpy.typing import NDArray


class GeometryHelper:
    """A class containing helper functions for geometry calculations."""

    @staticmethod
    def any_directions_match(direction: NDArray, test_directions: NDArray) -> bool:
        """Check whether any of the test directions match the given direction.

        Parameters
        ----------
        direction : NDArray
            The direction to test against.
        test_directions : NDArray
            The directions to test.

        Returns
        -------
        bool
            Whether any of the test directions match the given direction.
        """
        dot = np.dot(test_directions, direction)
        return np.any(dot == 1).item()

    @staticmethod
    def is_left(
        line_point: NDArray,
        line_direction: NDArray,
        test_point: NDArray,
    ) -> bool:
        """Check if a 2D point is left of a line.

        Parameters
        ----------
        line_point : NDArray
            An (x, y) coordinate of a point on the line.
        line_direction : NDArray
            The direction vector of the line.
        test_point : NDArray
            The (x, y) coordinate of the test point.

        Returns
        -------
        bool
            True if the test point is left of the line, False if the test point is right
            of the line.

        Raises
        ------
        ValueError
            If the test point is on the line.
        """
        vx, vy = test_point[0] - line_point[0], test_point[1] - line_point[1]
        cross = line_direction[0] * vy - line_direction[1] * vx
        if cross == 0:
            msg = "Point is on the line"
            raise ValueError(msg)
        return cross > 0

    @staticmethod
    def point_in_direction(
        ray_origin: NDArray,
        ray_direction: NDArray,
        test_point: NDArray,
    ) -> bool:
        """Check whether a test point is in the direction of a ray.

        Parameters
        ----------
        ray_origin : NDArray
            The origin point for the ray.
        ray_direction : NDArray
            The direction of the ray.
        test_point : NDArray
            The test point.

        Returns
        -------
        bool
            Whether the test point is in the direction of the ray.

        Raises
        ------
        ValueError
            If the test point is equidistant from the ray origin.
        """
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        s = np.dot(test_point - ray_origin, ray_direction)
        if s == 0:
            msg = (
                "Point is equidistant to the ray origin, this is impossible. "
                "Are any of your face normals inverted?"
            )
            raise ValueError(msg)
        return s > 0

    @staticmethod
    def rays_intersect(
        point_a: NDArray,
        normal_a: NDArray,
        point_b: NDArray,
        normal_b: NDArray,
        *,
        tolerance: float = 1e-8,
    ) -> bool:
        """Check whether two 2D rays intersect.

        Parameters
        ----------
        point_a : NDArray
            The origin point for the first ray.
        normal_a : NDArray
            The normal vector of the first ray.
        point_b : NDArray
            The origin point for the second ray.
        normal_b : NDArray
            The normal vector of the second ray.
        tolerance : float, optional
            The tolerance for determining parallelism and colinearity, by default 1e-8

        Returns
        -------
        bool
            Whether the rays intersect.

        Raises
        ------
        ValueError
            If the rays are parallel and colinear.
        ValueError
            If the rays are parallel but not colinear.
        """
        # Build the linear system
        coefficients = np.array(
            [[normal_a[0], -normal_b[0]], [normal_a[1], -normal_b[1]]],
        )
        # Difference between ray origins
        delta = point_b - point_a

        # Determinant tells whether the directions are linearly dependent
        determinant = np.linalg.det(coefficients)
        if abs(determinant) < tolerance:
            # Use a cross product to check if delta is parallel to the first ray
            cross_value = np.cross(np.append(normal_a, 0), np.append(delta, 0))
            if np.linalg.norm(cross_value) < tolerance:
                msg = "Colinear"
                raise ValueError(msg)
            msg = "Parallel"
            raise ValueError(msg)

        # Solve for where the rays would intersect
        t, s = np.linalg.solve(coefficients, delta)
        # Check if the intersection point is in the positive direction of both rays
        return t >= -tolerance and s >= -tolerance
