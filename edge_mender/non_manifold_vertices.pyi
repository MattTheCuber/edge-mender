import numpy as np
from numpy.typing import NDArray

def find_num_non_manifold_vertices(
    faces: NDArray[np.int64],
    vertex_faces: NDArray[np.int64],
) -> int: ...
def repair_vertices(
    faces: NDArray[np.int64],
    vertices: NDArray[np.float64],
    vertex_faces: NDArray[np.int64],
    triangles_center: NDArray[np.float64],
    *,
    shift_distance: float = ...,
) -> NDArray[np.float64]: ...
