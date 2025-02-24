import setuptools
import os

# Read the long description from the README file.
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="brain_graph_metrics",
    version="0.1",
    author="Francesco Alberti",
    author_email="fnc.alberti@gmail.com",
    description="A Python package for computing graph theoretical metrics on brain connectivity networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alberti-f/brain_graph_metrics",
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "networkx"
    ],
    entry_points={
        "console_scripts": [
            "brain-graph-metrics=brain_graph_metrics.__main__:main",
        ],
    },
)
