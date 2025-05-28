#!/usr/bin/env python3
"""
MersenneHunter Setup Script
Advanced Prime Number Discovery Platform
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read dependencies
def read_requirements():
    with open("dependencies.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mersenne-hunter",
    version="2.0.0",
    author="MersenneHunter Project",
    author_email="contact@mersennehunter.org",
    description="Advanced distributed system for discovering Mersenne prime numbers",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mersenne-hunter",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.11",
    install_requires=[
        "Flask>=2.3.0",
        "numpy>=1.24.0",
        "numba>=0.58.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "gpu": ["cupy-cuda12x>=12.0.0"],
        "quantum": ["qiskit>=0.45.0", "qiskit-aer>=0.13.0"],
        "full": ["cupy-cuda12x>=12.0.0", "qiskit>=0.45.0", "qiskit-aer>=0.13.0"],
    },
    entry_points={
        "console_scripts": [
            "mersenne-hunter=main:main",
            "mersenne-nonstop=mersenne_nonstop:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["templates/*.html", "static/*"],
    },
)