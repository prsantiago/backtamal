from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("backtamal/engine_cython.pyx", language_level=3)
)
