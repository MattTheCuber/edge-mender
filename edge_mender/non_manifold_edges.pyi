import numpy as np
from numpy.typing import NDArray

def find_non_manifold_edges(
    edges: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]: ...
