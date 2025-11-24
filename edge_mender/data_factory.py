import cc3d
import fill_voids
import numpy as np
from numpy.typing import NDArray


class DataFactory:
    @staticmethod
    def kill_you() -> NDArray:
        # Create initial data
        data = np.zeros((10, 10, 10))
        data[3:-3, 3:-3, 3:-3] = 1

        # Case 1 - simple extrusion
        data[2, 4, 4] = 1
        data[2, 5, 5] = 1

        # Case 2 - double extrusion
        data[4, 4, 2] = 1
        data[4, 4, 1] = 1
        data[5, 5, 2] = 1
        data[5, 5, 1] = 1

        # Case 3 - triple extrusion
        data[4, 7, 4] = 1
        data[4, 8, 4] = 1
        data[4, 9, 4] = 1
        data[5, 7, 5] = 1
        data[5, 8, 5] = 1
        data[5, 9, 5] = 1

        # Case 4 - stairs
        data[4, 4, 7] = 1
        data[4, 4, 8] = 1
        data[4, 5, 7] = 1
        data[5, 5, 7] = 1
        data[5, 5, 8] = 1
        data[5, 5, 9] = 1

        # Case 5 - ceiling
        data[4, 1, 4] = 1
        data[4, 1, 5] = 1
        data[4, 2, 4] = 1
        data[5, 1, 4] = 1
        data[5, 1, 5] = 1
        data[5, 2, 5] = 1

        # Case 6 - checkboard
        data[7, 5, 4] = 1
        data[7, 4, 5] = 1
        data[8, 4, 4] = 1
        data[8, 3, 4] = 1
        data[8, 3, 3] = 1
        data[7, 3, 3] = 1
        data[8, 5, 5] = 1
        data[8, 6, 5] = 1
        data[8, 6, 6] = 1
        data[7, 6, 6] = 1

        return data

    @staticmethod
    def simple_extrusion() -> NDArray:
        # Create initial data
        data = np.zeros((4, 4, 4))

        # Floor
        data[1:3, 1, 1:3] = 1

        # Extrusion
        data[1, 2, 1] = 1
        data[2, 2, 2] = 1

        return data

    @staticmethod
    def double_extrusion() -> NDArray:
        # Create initial data
        data = np.zeros((4, 5, 4))

        # Floor
        data[1:3, 1, 1:3] = 1

        # Double Extrusion
        data[1, 2:4, 1] = 1
        data[2, 2:4, 2] = 1

        return data

    @staticmethod
    def triple_extrusion() -> NDArray:
        # Create initial data
        data = np.zeros((4, 6, 4))

        # Floor
        data[1:3, 1, 1:3] = 1

        # Triple Extrusion
        data[1, 2:5, 1] = 1
        data[2, 2:5, 2] = 1

        return data

    @staticmethod
    def stairs() -> NDArray:
        # Create initial data
        data = np.zeros((4, 6, 4))

        # Floor
        data[1:3, 1, 1:3] = 1

        # Step 1
        data[1, 2, 2] = 1

        # Step 2
        data[2, 2:4, 2] = 1

        # Step 3
        data[1, 2:5, 1] = 1

        return data

    @staticmethod
    def ceiling() -> NDArray:
        # Create initial data
        data = np.zeros((4, 5, 4))

        # Floor
        data[1:3, 1, 1:3] = 1

        # Extrusion
        data[1, 2, 1] = 1
        data[2, 2, 2] = 1

        # Ceiling
        data[1:3, 3, 1:3] = 1

        return data

    @staticmethod
    def double_tower_ceiling() -> NDArray:
        # Create initial data
        data = np.zeros((4, 5, 5))

        # Floor
        data[1:3, 1, 1:3] = 1

        # Extrusion 1
        data[1, 2, 1] = 1
        data[2, 2, 2] = 1

        # Extrusion 2
        data[1, 3, 1] = 1
        data[2, 3, 2] = 1

        # Ceiling
        data[1:3, 4, 1:3] = 1

        # Connector
        data[2, 1, 3] = 1
        data[2, 4, 3] = 1
        data[2, 1:5, 4] = 1

        return data

    @staticmethod
    def hanging_points() -> NDArray:
        # Create initial data
        data = np.zeros((5, 5, 3))

        # Points
        data[2, 3, 1] = 1
        data[3, 2, 1] = 1

        # Connector
        data[1:4, 1, 1] = 1
        data[1, 1:4, 1] = 1

        return data

    @staticmethod
    def checkerboard() -> NDArray:
        # Create initial data
        data = np.zeros((4, 5, 5))

        # Floor
        data[1:3, 1, 1:3] = 1

        # Extrusion 1
        data[1, 2, 1] = 1
        data[2, 2, 2] = 1

        # Extrusion 2
        data[1, 3, 2] = 1
        data[2, 3, 1] = 1

        # Ceiling
        data[1:3, 4, 1:3] = 1

        # Connector
        data[2, 1, 3] = 1
        data[2, 4, 3] = 1
        data[2, 1:5, 4] = 1

        return data

    @staticmethod
    def hole() -> NDArray:
        """faces_to_flip=[44, 45, 54, 55, 66, 67, 68, 69, 70, 71]"""
        # Create initial data
        data = np.zeros((5, 5, 5))

        # Cube
        data[1:4, 1:4, 1:4] = 1

        # Hole
        data[2, 2, 2] = 0

        # Front cut outs
        data[1, 1, 2] = 0
        data[2, 1, 1] = 0
        data[2, 1, 3] = 0
        data[3, 1, 2] = 0

        # Back cut outs
        data[1:4, 3, 1:4] = 0
        data[2, 3, 2] = 1

        return data

    @staticmethod
    def random(*, size: int = 16, seed: int | None = None) -> NDArray:
        rng = np.random.default_rng(seed)
        data = rng.integers(0, 2, (size, size, size), dtype=np.uint8)
        data = cc3d.largest_k(data, k=1, connectivity=6, binary_image=True).astype(
            np.uint8
        )
        data = fill_voids.fill(data, in_place=True)
        data = fill_voids.fill(data, in_place=True)
        return np.pad(data, pad_width=1, mode="constant", constant_values=0)
