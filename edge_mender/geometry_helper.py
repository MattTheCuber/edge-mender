"""Provides a class containing helper functions for geometry calculations."""

import numpy as np
from numpy.typing import NDArray


class GeometryHelper:
    """A class containing helper functions for geometry calculations."""

    @staticmethod
    def point_in_line(
        test_point: NDArray,
        line_point: NDArray,
        direction: NDArray,
        *,
        tolerance: float = 1e-9,
    ) -> bool:
        """Test if a point is contained on a line.

        Parameters
        ----------
        test_point : NDArray
            The (x, y, z) coordinate of the point to test.
        line_point : NDArray
            An (x, y, z) coordinate of a point on the line.
        direction : NDArray
            The direction vector of the line.
        tolerance : float = 1e-9
            The tolerance for floating point comparisons.

        Returns
        -------
        bool
            True if the point lies on the line, False otherwise.
        """
        difference = test_point - line_point

        # If the direction vector is zero, line is undefined
        if np.allclose(direction, 0):
            return np.allclose(test_point, line_point)

        # Handle division by zero by checking ratios only where direction ≠ 0
        t_values = []
        for i in range(3):
            if abs(direction[i]) > tolerance:
                t_values.append(difference[i] / direction[i])
            elif abs(difference[i]) > tolerance:
                # If direction is zero but diff isn't, point can't lie on line
                return False

        # All non-zero ratios must be the same
        return np.allclose(t_values, t_values[0])

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
    def is_left(line_point: NDArray, direction: NDArray, test_point: NDArray) -> bool:
        vx, vy = test_point[0] - line_point[0], test_point[1] - line_point[1]
        cross = direction[0] * vy - direction[1] * vx
        if cross == 0:
            msg = "Point is on the line"
            raise ValueError(msg)
        return cross > 0

    @staticmethod
    def rays_intersect(
        point_1: NDArray,
        normal_1: NDArray,
        point_2: NDArray,
        normal_2: NDArray,
        *,
        tol: float = 1e-9,
    ) -> bool:
        A = np.array([[normal_1[0], -normal_2[0]], [normal_1[1], -normal_2[1]]])
        b = point_2 - point_1

        det = np.linalg.det(A)
        if abs(det) < tol:
            if np.linalg.norm(np.cross(np.append(normal_1, 0), np.append(b, 0))) < tol:
                msg = "Colinear"
                raise ValueError(msg)
            msg = "Parallel"
            raise ValueError(msg)

        t, s = np.linalg.solve(A, b)
        return t >= -tol and s >= -tol
