#!/usr/bin/env python
import imp
import io
import os
import sys

try:
    from setuptools import find_packages, setup
except ImportError:
    raise ImportError(
        "'setuptools' is required but not installed. To install it, "
        "follow the instructions at "
        "https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py"
    )


def read(*filenames, **kwargs):
    encoding = kwargs.get("encoding", "utf-8")
    sep = kwargs.get("sep", "\n")
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


root = os.path.dirname(os.path.realpath(__file__))
version_module = imp.load_source(
    "version", os.path.join(root, "nengo_spa", "version.py")
)
testing = "test" in sys.argv or "pytest" in sys.argv

numpy = "numpy>=1.8"
docs_require = [
    "jupyter_client>=5.3.4",
    "sphinx>=2.2.1",
    "nbsphinx>=0.5.0",
    "nengo_sphinx_theme>=1.2.0",
]
optional_requires = ["scipy>=1.4.1"]
tests_require = [
    "jupyter>=1.0.0",
    "matplotlib>=2.0",
    "nbformat>=5.0.7",
    "pytest>=3.6",
    "pytest-plt>=1.0.1",
    "pytest-rng>=1.0.0",
]

setup(
    name="nengo_spa",
    version=version_module.version,
    author="Applied Brain Research",
    author_email="info@appliedbrainresearch.com",
    url="https://www.nengo.ai/nengo-spa/",
    packages=find_packages(exclude=["*.tests"]),
    scripts=[],
    license="Free for non-commercial use",
    description="An implementation of the Semantic Pointer Architecture for " "Nengo",
    long_description=read("README.rst", "CHANGES.rst"),
    zip_safe=True,
    include_package_data=True,
    setup_requires=["pytest-runner>=5.2"] if testing else [] + [numpy,],
    install_requires=["nengo>=2.7", numpy,],
    extras_require={
        "all": docs_require + optional_requires + tests_require,
        "docs": docs_require,
        "optional": optional_requires,
        "tests": tests_require,
    },
    tests_require=tests_require,
    entry_points={},
    classifiers=[  # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: Free for non-commercial use",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
