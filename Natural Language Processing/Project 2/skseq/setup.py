from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(['skseq/sequence_list_c.pyx', 'skseq/structured_perceptron_c.pyx'])
)