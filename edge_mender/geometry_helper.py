"""Provides a class containing helper functions for geometry calculations."""

import numpy as np
from numpy.typing import NDArray


class GeometryHelper:
    """A class containing helper functions for geometry calculations."""

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
    def angle_between_point_and_ray(
        point: NDArray,
        ray_point: NDArray,
        ray_dir: NDArray,
    ) -> float:
        """Compute the angle between a test point and a ray.

        Parameters
        ----------
        point : NDArray
            The test point.
        ray_point : NDArray
            The origin point for the ray.
        ray_dir : NDArray
            The direction of the ray.

        Returns
        -------
        float
            The angle in radians between the test point and the ray.
        """
        # Vector from ray point to external point
        v = point - ray_point

        # Normalize direction and v
        d_norm = ray_dir / np.linalg.norm(ray_dir)
        v_norm = v / np.linalg.norm(v)

        # Clamp to handle floating point precision issues
        dot = np.clip(np.dot(d_norm, v_norm), -1.0, 1.0)

        return np.arccos(dot)

    @staticmethod
    def rays_intersect(
        point_1: NDArray,
        normal_1: NDArray,
        point_2: NDArray,
        normal_2: NDArray,
        *,
        tolerance: float = 1e-8,
    ) -> bool:
        """Check whether two 2D rays intersect.

        Parameters
        ----------
        point_1 : NDArray
            The origin point for the first ray.
        normal_1 : NDArray
            The normal vector of the first ray.
        point_2 : NDArray
            The origin point for the second ray.
        normal_2 : NDArray
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
            [[normal_1[0], -normal_2[0]], [normal_1[1], -normal_2[1]]],
        )
        # Difference between ray origins
        delta = point_2 - point_1

        # Determinant tells whether the directions are linearly dependent
        determinant = np.linalg.det(coefficients)
        if abs(determinant) < tolerance:
            # Use a cross product to check if delta is parallel to the first ray
            cross_value = np.cross(np.append(normal_1, 0), np.append(delta, 0))
            if np.linalg.norm(cross_value) < tolerance:
                msg = "Colinear"
                raise ValueError(msg)
            msg = "Parallel"
            raise ValueError(msg)

        # Solve for where the rays would intersect
        t, s = np.linalg.solve(coefficients, delta)
        # Check if the intersection point is in the positive direction of both rays
        return t >= -tolerance and s >= -tolerance
