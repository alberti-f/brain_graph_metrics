# brain_graph_metrics

## Overview
`brain_graph_metrics` is a NetworkX-based module to compute, in one call, all the most common graph metrics from csv file storing a connectivity matrix. The module's functions can be called in a Python script, but also from terminal.

## Purpose
The package streamlines the analysis of brain connectivity data by:
- Computing node-level metrics such as degree, strength, clustering coefficient, betweenness, eigenvector centrality, nodal efficiency, local efficiency, and participation coefficient.
- Calculating global network metrics including global efficiency, average clustering, shortest path length, density, and modularity.
- Allowing flexible conversion of connectivity matrices to binary networks and thresholding of connection strengths.

## Installation
You can install the package directly from GitHub using pip:

```bash
pip install git+https://github.com/yourusername/brain_graph_metrics.git
