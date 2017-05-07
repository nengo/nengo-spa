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
        "https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py")


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


root = os.path.dirname(os.path.realpath(__file__))
version_module = imp.load_source(
    'version', os.path.join(root, 'nengo_spa', 'version.py'))
testing = 'test' in sys.argv or 'pytest' in sys.argv

setup(
    name="nengo_spa",
    version=version_module.version,
    author="Applied Brain Research",
    author_email="info@appliedbrainresearch.com",
    packages=find_packages(),
    scripts=[],
    license="Free for non-commercial use",
    description="An implementation of the Semantic Pointer Architecture for "
                "Nengo",
    long_description=read('README.rst', 'CHANGES.rst'),
    zip_safe=True,
    include_package_data=True,
    setup_requires=['pytest-runner'] if testing else [] + [
        'numpy>=1.7',
    ],
    install_requires=[
        'nengo>=2.4',
        'numpy>=1.7',
    ],
    extras_require={
        'prob_cleanup': ['scipy'],
    },
    tests_require=[
        'jupyter',
        'matplotlib>=1.4',
        'pytest>=2.3',
    ],
    entry_points={
    },
    classifiers=[  # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: Free for non-commercial use',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
