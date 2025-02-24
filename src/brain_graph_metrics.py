#!/usr/bin/env python
import csv
import argparse
import numpy as np
import pandas as pd
import networkx as nx


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute graph theoretical metrics from a tractography-based structural connectivity CSV file."
    )
    parser.add_argument("connectivity", help="Path to the CSV file containing the connectivity matrix.")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Threshold below which connection weights are set to zero (default: 0.0, i.e. no thresholding).")
    parser.add_argument("--binary", action="store_true",
                        help="Convert the connectivity matrix to a binary (unweighted) graph.")
    parser.add_argument("--node_out", type=str, default="node_metrics.csv",
                        help="Output CSV file for node-wise metrics (default: node_metrics.csv).")
    parser.add_argument("--global_out", type=str, default="global_metrics.csv",
                        help="Output CSV file for global network metrics (default: global_metrics.csv).")
    return parser.parse_args()

def load_connectivity_matrix(filename, threshold=0.0, binary=False):
    # Determine the delimiter of the CSV file.
    with open(filename, 'r', encoding='utf-8') as f:
        sample = f.read(1024)
        f.seek(0)
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter

    df = pd.read_csv(filename, index_col=None, header=None, delimiter=delimiter)
    matrix = df.values.astype(float)

    # Apply thresholding if specified.
    if threshold > 0.0:
        matrix[matrix < threshold] = 0.0

    # Convert matrix to binary if the binary flag is set.
    if binary:
        matrix = (matrix > 0).astype(float)

    return matrix


def global_efficiency_node(G):
    # Global efficiency of node i: average of inverse shortest path lengths to all other nodes.
    n = len(G)
    denom = n - 1
    global_eff = dict(zip(G.nodes(), np.zeros(n)))

    lengths = nx.all_pairs_shortest_path_length(G)
    for source, targets in lengths:
        for _, distance in targets.items():
            if distance > 0:
                global_eff[source] += 1 / distance
        global_eff[source] /= denom

    return global_eff


def local_efficiency_node(G):
    # Local efficiency of node i: global efficiency of the subgraph induced by i's neighbors.
    local_eff = (nx.global_efficiency(G.subgraph(G[v])) for v in G)
    local_eff = dict(zip(G.nodes, local_eff))
    return local_eff


def assign_communities(G):
    # Use greedy modularity communities detection to assign community labels.
    communities = list(nx.algorithms.community.greedy_modularity_communities(G, weight='weight'))
    # Create a dictionary mapping node -> community index.
    comm_dict = {}
    for i, comm in enumerate(communities):
        for node in comm:
            comm_dict[node] = i
    return comm_dict, communities

def compute_participation_coefficient(G, communities):
    """
    Compute the participation coefficient for each node.
    Participation coefficient quantifies how evenly a node's connections
    are distributed across communities.
    """
    communities = dict(zip(G.nodes(), communities))
    participation = {}
    for node in G.nodes():
        total_strength = G.degree(node, weight='weight')
        if total_strength == 0:
            participation[node] = 0.0
            continue
        # Sum weights of edges from node to each community.
        comm_weights = {}
        for neighbor in G.neighbors(node):
            weight = G[node][neighbor].get('weight', 1.0)
            comm = communities[neighbor]
            comm_weights[comm] = comm_weights.get(comm, 0.0) + weight
        sum_sq = sum((w / total_strength)**2 for w in comm_weights.values())
        participation[node] = 1.0 - sum_sq
    return participation

def compute_node_metrics(G, communities=None):
    # Compute degree (binary) and strength (weighted degree)
    degree_dict = dict(G.degree())
    strength_dict = dict(G.degree(weight='weight'))

    # Compute clustering coefficient (weighted)
    clustering_dict = nx.clustering(G, weight='weight')

    # Compute betweenness centrality (weighted)
    betweenness_dict = nx.betweenness_centrality(G, weight='weight', normalized=True)

    # Compute nodal efficiency (average inverse shortest path lengths)
    global_eff = global_efficiency_node(G)

    # Compute local efficiency for each node
    local_eff = local_efficiency_node(G)

    # Compute communities and participation coefficient.
    
    if communities is None:
        communities, communities_list = assign_communities(G)
    else:
        communities_list = [communities[node] for node in G]

    participation = compute_participation_coefficient(G, communities)

    # Combine all node metrics into a DataFrame.
    data = {
        "Degree": degree_dict,
        "Strength": strength_dict,
        "Clustering": clustering_dict,
        "Betweenness": betweenness_dict,
        "Global_Efficiency": global_eff,
        "Local_Efficiency": local_eff,
        "Community": communities,
        "Participation": participation,
    }
    node_metrics = pd.DataFrame(data)
    node_metrics.index.name = "Node"
    return node_metrics


def compute_global_metrics(G, communities):
    # Global efficiency
    global_eff = nx.global_efficiency(G)
    local_eff = nx.local_efficiency(G)
    
    # Average clustering coefficient
    avg_clustering = nx.average_clustering(G, weight='weight')
    
    # Average shortest path length: compute on the largest connected component if necessary.
    if nx.is_connected(G):
        avg_shortest_path = nx.average_shortest_path_length(G, weight='weight')
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc)
        avg_shortest_path = nx.average_shortest_path_length(subG, weight='weight')
    
    # Network density
    density = nx.density(G)
    
    # Modularity of the partition obtained by greedy modularity communities algorithm.
    modularity = nx.algorithms.community.quality.modularity(G, communities, weight='weight')
    
    # Average degree and strength
    avg_degree = np.mean([d for _, d in dict(G.degree()).items()])
    avg_strength = np.mean([s for _, s in dict(G.degree(weight='weight')).items()])

    seed = 42
    # small_world_index = nx.algorithms.smallworld.sigma(G, niter=100, nrand=10, seed=seed)

    metrics = {
        "Global_Efficiency": global_eff,
        "Local_Efficiency": local_eff,
        "Average_Clustering": avg_clustering,
        "Average_Shortest_Path_Length": avg_shortest_path,
        "Density": density,
        "Modularity": modularity,
        "Average_Degree": avg_degree,
        "Average_Strength": avg_strength,
        "Number_of_Nodes": G.number_of_nodes(),
        "Number_of_Edges": G.number_of_edges(),
        # "Small_World_Index": small_world_index
    }
    return pd.DataFrame([metrics])

def main():
    args = parse_args()
    
    # Load the connectivity matrix.
    matrix = load_connectivity_matrix(args.connectivity, threshold=args.threshold, binary=args.binary)
    
    # Create a weighted undirected graph from the connectivity matrix.
    # Assumes the matrix is symmetric.
    G = nx.from_numpy_array(matrix)
        
    # Compute communities again for global modularity (we already computed them in compute_node_metrics).
    communities_dict, communities = assign_communities(G)

    # Compute node-level metrics.
    node_metrics = compute_node_metrics(G, communities_dict)

    
    # Compute global network metrics.
    global_metrics = compute_global_metrics(G, communities)
    
    # Save outputs to CSV files.
    node_metrics.to_csv(args.node_out)
    global_metrics.to_csv(args.global_out, index=False)
    print("Node metrics saved to:", args.node_out)
    print("Global metrics saved to:", args.global_out)

if __name__ == '__main__':
    main()
