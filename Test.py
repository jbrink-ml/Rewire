import sys
import os
import networkx as nx
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt


# Import model function
from Rewire import evolutionary_network_sim

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

"""The purpose of this file is to run an example simulations of the rewire model with easily editable parameters"""

#Set up parameters
CONFIG = {
    "N": 100,               # Number of nodes
    "T": 10000,                 # Time steps
    "k": 4,                 # Average degree
    "q": 0.9,               # Quote probability
    "network_type": "WS",   # "BA", "ER", or "WS"
    "rewire_time": 10       # Rewire time parameter
}

def get_initial_graph(network_type, N, k):
    """Helper to generate the initial graph based on type."""
    if network_type == "BA":
        m = int(k / 2)
        G0 = nx.barabasi_albert_graph(N, m)
    elif network_type == "ER":
        p = k / (N - 1)
        G0 = nx.erdos_renyi_graph(N, p)
    else:  # Watts-Strogatz
        G0 = nx.watts_strogatz_graph(n=N, k=k, p=0.1)
    
    return nx.DiGraph(G0)

def run_experiment(config):
    N, T, k, q, net_type, rewire_time = config["N"], config["T"], config["k"], config["q"], config["network_type"], config["rewire_time"]
    
    print(f"--- Starting Experiment ---")
    print(f"Params: N={N}, k={k}, q={q}, T={T}, Type={net_type}, Rewire={rewire_time}")
    
    # Setup Graph
    G = get_initial_graph(net_type, N, k)
    
    # Run simulation
    clust,glob_clust, avg_p, diam, degree_counts, G_final = evolutionary_network_sim(
        G, q, T, rewire_time, verbose=True
    )
    
    print("Simulation complete.")
    print(f"Final clustering: {clust[-1] if clust else 'N/A'}")
    print(f"Final global clustering: {glob_clust[-1] if glob_clust else 'N/A'}")
    
    # Draw the final network
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G_final)
    nx.draw(G_final, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title(f"Final Network: N={N}, k={k}, q={q}, T={T}")
    plt.show()

if __name__ == "__main__":
    run_experiment(CONFIG)
