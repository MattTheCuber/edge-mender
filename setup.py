from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
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
