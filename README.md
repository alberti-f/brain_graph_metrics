# **brain_graph_metrics**

## Overview
`brain_graph_metrics` is a NetworkX-based module to compute, in one call, all the most common graph metrics from csv file storing a connectivity matrix. The module's functions can be called in a Python script, but also from terminal.

## Purpose
The package streamlines the analysis of brain connectivity data by:
- Computing node-level metrics such as degree, strength, clustering coefficient, betweenness, eigenvector centrality, nodal efficiency, local efficiency, and participation coefficient.
- Calculating global network metrics including global efficiency, average clustering, shortest path length, density, and modularity.
- Allowing flexible conversion of connectivity matrices to binary networks and thresholding of connection strengths.

## Usage
The main nodal and global graph metrics of a graph can be computed from terminal as:
```bash
brain-graph-metrics <matrix.csv> --threshold 95 --binary --node_out <output_node_metrics.csv> --global_out <output_global_metrics.csv>
```

- **<matrix.csv>**: Input CSV file containing the connectivity matrix.\
The file must have no header nor row names.
- **threshold**: Optional threshold; Value between 0 and 100. Percentile below whitch connections are removed. Default is 0.
- **binary**: Converts the connectivity matrix to a binary (unweighted) graph.
- **node_out <output_node_metrics.csv>**: Output CSV file for node-wise metrics.
- **global_out <output_global_metrics.csv>**: Output CSV file for global network metrics.

For detailed usage and options see ``` python script.py --help```\
Otherwise brain_graph_metrics can be usede as any other Python module.

## Installation
You can install the package directly from GitHub using pip:

```bash
pip install git+https://github.com/alberti-f/brain_graph_metrics.git
```

## Dependencies
The following Python packages will be installed with brain_graph_metrics if not already present in the environment:
- Python >= 3.8
- numpy
- scipy
- pandas
- networkx
