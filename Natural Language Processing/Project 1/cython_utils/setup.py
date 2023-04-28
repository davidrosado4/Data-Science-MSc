from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("cython_utils/tf_idf.pyx"),
)
