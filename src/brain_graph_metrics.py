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
    # Read the CSV file into a pandas DataFrame and then convert to a numpy array.
    df = pd.read_csv(filename, index_col=0)
    matrix = df.values.astype(float)
    
    # Apply thresholding if specified.
    if threshold > 0.0:
        matrix[matrix < threshold] = 0.0

    # Convert matrix to binary if the binary flag is set.
    if binary:
        matrix = (matrix > 0).astype(float)
        
    return matrix, df.index.tolist()

def compute_nodal_efficiency(G):
    # Efficiency of node i: average of inverse shortest path lengths to all other nodes.
    n = G.number_of_nodes()
    efficiency = {}
    for node in G.nodes():
        lengths = nx.single_source_dijkstra_path_length(G, node, weight='weight')
        # Sum 1/d for all reachable nodes (skip self: distance==0)
        eff_sum = 0.0
        for target, d in lengths.items():
            if node != target and d > 0:
                eff_sum += 1.0/d
        efficiency[node] = eff_sum / (n - 1) if n > 1 else 0.0
    return efficiency

def compute_local_efficiency_per_node(G):
    # Local efficiency of node i: global efficiency of the subgraph induced by i's neighbors.
    local_eff = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            local_eff[node] = 0.0
        else:
            subgraph = G.subgraph(neighbors)
            local_eff[node] = nx.global_efficiency(subgraph)
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

def compute_participation_coefficient(G, comm_dict):
    """
    Compute the participation coefficient for each node.
    Participation coefficient quantifies how evenly a node's connections
    are distributed across communities.
    """
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
            comm = comm_dict[neighbor]
            comm_weights[comm] = comm_weights.get(comm, 0.0) + weight
        sum_sq = sum((w / total_strength)**2 for w in comm_weights.values())
        participation[node] = 1.0 - sum_sq
    return participation

def compute_node_metrics(G):
    # Compute degree (binary) and strength (weighted degree)
    degree_dict = dict(G.degree())
    strength_dict = dict(G.degree(weight='weight'))
    
    # Compute clustering coefficient (weighted)
    clustering_dict = nx.clustering(G, weight='weight')
    
    # Compute betweenness centrality (weighted)
    betweenness_dict = nx.betweenness_centrality(G, weight='weight', normalized=True)
    
    # Compute eigenvector centrality using the numpy-based algorithm.
    try:
        eigenvector_dict = nx.eigenvector_centrality_numpy(G, weight='weight')
    except Exception as e:
        print("Eigenvector centrality did not converge:", e)
        eigenvector_dict = {node: None for node in G.nodes()}
    
    # Compute nodal efficiency (average inverse shortest path lengths)
    nodal_eff = compute_nodal_efficiency(G)
    
    # Compute local efficiency for each node
    local_eff = compute_local_efficiency_per_node(G)
    
    # Compute communities and participation coefficient.
    comm_dict, _ = assign_communities(G)
    participation = compute_participation_coefficient(G, comm_dict)
    
    # Combine all node metrics into a DataFrame.
    data = {
        "Degree": degree_dict,
        "Strength": strength_dict,
        "Clustering": clustering_dict,
        "Betweenness": betweenness_dict,
        "Eigenvector": eigenvector_dict,
        "Nodal_Efficiency": nodal_eff,
        "Local_Efficiency": local_eff,
        "Community": comm_dict,
        "Participation": participation,
    }
    node_metrics = pd.DataFrame(data)
    node_metrics.index.name = "Node"
    return node_metrics

def compute_global_metrics(G, communities):
    # Global efficiency
    global_eff = nx.global_efficiency(G)
    
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
    
    metrics = {
        "Global_Efficiency": global_eff,
        "Average_Clustering": avg_clustering,
        "Average_Shortest_Path_Length": avg_shortest_path,
        "Density": density,
        "Modularity": modularity,
        "Average_Degree": avg_degree,
        "Average_Strength": avg_strength,
        "Number_of_Nodes": G.number_of_nodes(),
        "Number_of_Edges": G.number_of_edges(),
    }
    return pd.DataFrame([metrics])

def main():
    args = parse_args()
    
    # Load the connectivity matrix.
    matrix, labels = load_connectivity_matrix(args.connectivity, threshold=args.threshold, binary=args.binary)
    
    # Create a weighted undirected graph from the connectivity matrix.
    # Assumes the matrix is symmetric.
    G = nx.from_numpy_array(matrix)
    
    # Compute node-level metrics.
    node_metrics = compute_node_metrics(G)
    
    # Compute communities again for global modularity (we already computed them in compute_node_metrics).
    _, communities = assign_communities(G)
    
    # Compute global network metrics.
    global_metrics = compute_global_metrics(G, communities)
    
    # Save outputs to CSV files.
    node_metrics.to_csv(args.node_out)
    global_metrics.to_csv(args.global_out, index=False)
    print("Node metrics saved to:", args.node_out)
    print("Global metrics saved to:", args.global_out)

if __name__ == '__main__':
    main()

