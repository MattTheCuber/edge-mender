"""Setup script for edge_mender package with Cython extensions."""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        name="edge_mender.non_manifold_edges",
        sources=["edge_mender/non_manifold_edges.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=[],
    ),
    Extension(
        name="edge_mender.non_manifold_vertices",
        sources=["edge_mender/non_manifold_vertices.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=[],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
