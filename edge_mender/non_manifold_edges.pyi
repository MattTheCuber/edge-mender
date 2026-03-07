import numpy as np
from numpy.typing import NDArray

def find_non_manifold_edges(
    edges: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]: ...
def get_faces_at_edge(
    v0: np.int64,
    v1: np.int64,
    faces: NDArray[np.int64],
) -> NDArray[np.int64]: ...
def get_faces_at_vertex(
    vertex: np.int64,
    faces: NDArray[np.int64],
) -> NDArray[np.int64]: ...
