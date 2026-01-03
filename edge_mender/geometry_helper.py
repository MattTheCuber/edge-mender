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

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Dot_product
        .. [2] https://stackoverflow.com/questions/49535295/how-to-check-if-vectors-are-facing-same-direction
        """
        dot = np.dot(test_directions, direction)
        return np.any(dot == 1).item()

    @staticmethod
    def get_diagonal_orthogonal_directions(
        direction: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Calculate two diagonal directions orthogonal to the given direction.

        Parameters
        ----------
        direction : NDArray
            The input direction vector.

        Returns
        -------
        NDArray
            The first orthogonal direction vectors.
        NDArray
            The second orthogonal direction vectors.
        """
        line_direction_a = [1, 1, 1] - np.abs(direction)
        line_direction_b = line_direction_a.copy()
        line_direction_b[np.argmax(line_direction_a)] = -1
        return line_direction_a, line_direction_b

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

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Cross_product
        .. [2] https://stackoverflow.com/a/3461533/9725459
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

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Dot_product
        .. [2] https://math.stackexchange.com/a/1330214/612214
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

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
        .. [2] https://stackoverflow.com/questions/2931573/determining-if-two-rays-intersect
        .. [3] https://discourse.threejs.org/t/solved-how-to-find-intersection-between-two-rays/6464/8
        .. [4] https://scicomp.stackexchange.com/questions/36421/how-to-determine-if-2-rays-intersect
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

    @staticmethod
    def hypotenuse_points(points: NDArray) -> tuple[int, int]:
        """Find the two points that form the hypotenuse of a right triangle.

        Parameters
        ----------
        points : NDArray
            The three 3D points to analyze.

        Returns
        -------
        tuple[int, int]
            The two hypotenuse endpoint indices.

        References
        ----------
        .. [1] https://stackoverflow.com/a/54057593/9725459
        """
        # Compute squared distances
        distance_squared = np.sum(
            (points[:, None, :] - points[None, :, :]) ** 2,
            axis=-1,
        )

        # Find the indices of the maximum distance
        i, j = np.unravel_index(np.argmax(distance_squared), distance_squared.shape)

        return i.item(), j.item()

        # Test cases (using ceiling test case data):
        # face indices 22
        # face points [18 16 19] = [18, 16]
        # face vertices [[0.5 2.5 1.5]
        #  [1.5 1.5 1.5]
        #  [1.5 2.5 1.5]]
        # face indices 24
        # face points [16  7 19] = [16, 7]
        # face vertices [[1.5 1.5 1.5]
        #  [1.5 2.5 0.5]
        #  [1.5 2.5 1.5]]
        # face indices 27
        # face points [16 19 17] = [19, 17]
        # face vertices [[1.5 1.5 1.5]
        #  [1.5 2.5 1.5]
        #  [2.5 1.5 1.5]]
        # face indices 55
        # face points [16 28 19] = [28, 19]
        # face vertices [[1.5 1.5 1.5]
        #  [1.5 1.5 2.5]
        #  [1.5 2.5 1.5]]
