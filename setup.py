from setuptools import setup, find_packages
from Cython.Build import cythonize

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="backtamal",
    version="0.1.0",
    author="Santiago Peña",
    author_email="prsantiago96@gmail.com",
    description="Un pequeño motor de autograd para valores escalares y una librería de redes neuronales sobre él, con una API similar a PyTorch. Incluye opción de aceleración con Cython para mayor rendimiento.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/prsantiago/backtamal",
    ext_modules=cythonize([
        "backtamal/engine_cython.pyx",
        "backtamal/nn_cython.pyx"
    ], language_level=3),
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "Cython==3.0.12"
    ],
)
