import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
from itertools import combinations
from typing import Iterable, Union, Tuple, List
import LCSFinder as lcs
from collections import Counter

def evolutionary_network_sim(
    G: nx.Graph,
    quote_prob: float,
    timesteps: int,
    rewire_time: int,
    outdir: str = "./",
    outfile: str = "test_output.txt",
    dunbar: Union[int, None] = None,
    verbose: bool = False,  # TODO: cleaner implementation with logging module. This is more for testing
    SBM_graph: bool = False,
    poisson_lambda: Union[float, int] = 3,
):
    """Simulate the quoter model on a graph G. Nodes take turns generating content according to two
    mechanisms: (i) creating new content from a specified vocabulary distribution, (ii) quoting
    from a neighbor's past text. After each node has tweeted an average of rewire_time times, we rewire the graph
    according to the following procedure: (1) calculate the cross-entropy between all pairs of nodes, (2) add an 
    edge between the pair with lowest cross-entropy if they are not already connected, and (3) remove the edge between
    the pair with highest cross-entropy if they are connected.


    [1] Bagrow, J. P., & Mitchell, L. (2018). The quoter model: A paradigmatic model of the social
    flow of written information. Chaos: An Interdisciplinary Journal of Nonlinear Science, 28(7),
    075304.

    Args:
        G (nx.Graph): Directed graph to simulate quoter model on
        quote_prob (float): Quote probability q as defined in [1]
        timesteps (int): Number of time-steps to simulate for. timesteps=1000 really means 1000*nx.number_of_nodes(G), i.e. each node will have 'tweeted' ~1000 times
        rewire_time (int): Average number of times each node should tweet before rewiring the graph.
        outdir (string): Name of directory for data to be stored in
        outfile (string): Name of file for this simulation
        dunbar (int or None): If int, limit in-degree to dunbar's number
        verbose: <temp> giving useful output during testing

    TODO: add args from other previous experiments, such as
        lambda (quote length > 0) - from q-lambda [added as poisson_lambda]
        alpha_alter, alpha_ego - from theory_link
    and potentially others
    """
    if verbose:
        # TODO would be useful to add summary statistics of provided networks in documentation
        print(f"G has {len(G.nodes())} nodes and {len(G.edges())} edges")

    # vocabulary distribution - NOTE: currently just uniform distribution of integers. Would like to choose optional distribution, including importing them or generating from LLMs
    alpha = 1.5
    z = 1000
    vocab = np.arange(1, z + 1)
    weights = vocab ** (-alpha)
    weights /= weights.sum()

    # limit IN-DEGREE to just dunbar's number
    if dunbar:
        for node in G.nodes():
            nbrs = list(G.predecessors(node))
            if len(nbrs) > dunbar:
                nbrs_rmv = random.sample(nbrs, len(nbrs) - dunbar)
                G.remove_edges_from([(nbr, node) for nbr in nbrs_rmv])

    # create initial tweet for each user
    startWords = 20
    for node in G.nodes():
        newWords = np.random.choice(
            vocab, size=startWords, replace=True, p=weights
        ).tolist()
        G.nodes[node]["words"] = newWords
        G.nodes[node]["times"] = [0] * len(newWords)

    if verbose:
        print("Initial vocab created; starting simulation")
    # Set up metrics to track over time
    clust=[nx.average_clustering(G)] # Average clustering coefficient
    glob_clust=[nx.transitivity(G)] # Global clustering coefficient
    avg_p=[nx.average_shortest_path_length(G)] # Average shortest path length
    diam=[nx.diameter(G)] # Diameter of the graph
    # simulate quoter model
    for timestep_ in range(1, timesteps * nx.number_of_nodes(G)):
        if verbose:
            print(timestep_)

        node = random.choice(list(G.nodes))

        # length of tweet
        tweetLength = np.random.poisson(lam=poisson_lambda)

        # quote with probability quote_prob, provided ego has alters to quote from
        nbrs = list(G.predecessors(node))
        if random.random() < quote_prob and len(nbrs) > 0:
            # pick a neighbor to quote from (simplifying assumption: uniformly at random from all neighbors)
            user_copied = random.choice(nbrs)

            # find a valid position in the neighbor's text to quote from
            words_friend = G.nodes[user_copied]["words"]
            numWords_friend = len(words_friend)
            copy_pos_start = random.choice(
                list(range(max(0, numWords_friend - tweetLength)))
            )
            copy_pos_end = min(numWords_friend - 1, copy_pos_start + tweetLength)
            newWords = words_friend[copy_pos_start:copy_pos_end]

        else:  # new content
            newWords = np.random.choice(
                vocab, size=tweetLength, replace=True, p=weights
            ).tolist()

        G.nodes[node]["words"].extend(newWords)
        G.nodes[node]["times"].extend([timestep_] * len(newWords))

        # Rewire graph every rewire_time steps (on average)
        if timestep_ % (len(G)*rewire_time) == 0:
            G=rewire(G)
            #Update metrics
            curr_clust=nx.average_clustering(G)
            curr_glob_clust=nx.transitivity(G)
            clust.append(curr_clust)
            glob_clust.append(curr_glob_clust)
            largest_scc = max(nx.strongly_connected_components(G), key=len)
            # Create a subgraph of just that component
            G_largest_scc = G.subgraph(largest_scc)
            
            # Calculate the metric on the strongly connected subgraph
            avg_path = nx.average_shortest_path_length(G_largest_scc)
            #print(curr_clust)
            diameter=nx.diameter(G_largest_scc)
            diam.append(diameter)
            avg_p.append(avg_path)
            
            #print(edge_clustering_coeff(G, 1, 4, draw=True))

    # Get degree distribution at end of simulation
    degrees = [d for n, d in G.in_degree()]
    # Count occurrences of each degree
    degree_counts = Counter(degrees)
    # Output example: Counter({5: 18, 4: 15, 6: 12, ...})
    print(degree_counts)
    return clust,glob_clust, avg_p, diam, degree_counts, G



def rewire(G: nx.Graph):
    """Rewire the graph according to the following procedure: (1) calculate the cross-entropy 
    between all pairs of nodes, (2) add an edge between the pair with lowest cross-entropy if 
    they are not already connected, and (3) remove the edge between the pair with highest 
    cross-entropy if they are connected.
    """
    l=len(G)
    hx_disconn = np.full((l, 1), 100000,dtype=float)
    hx_conn=np.full((l, 1), -1,dtype=float)
    u=random.choice(list(G.nodes))

    # Iterate through all N pairs of node u and its neighbors
    while G.in_degree(u)==0:
        u=random.choice(list(G.nodes))
    for v in range(l):
         if v==u:
            continue

         if G.has_edge(u, v):
                hx_conn[v] = cross_entropy(
                G,u,v
                )
         else:
            hx_disconn[v] = cross_entropy(
                G,u,v
                )
    print(u)
    v_add = np.argmin(hx_disconn)
    v_remove=np.argmax(hx_conn)
    print("connecting nodes:", u, v_add)
    G.add_edge(u, v_add)
    G.add_edge(v_add, u)
    print("disconnecting nodes:", u, v_remove)
    G.remove_edge(u, v_remove)  
    G.remove_edge(v_remove, u)
    return G
    


def cross_entropy(G, source_id, target_id):
    """Using the LCSFinder package, compute the cross entropy between the source and target nodes."""
    src_words = G.nodes[source_id]["words"]
    src_times = G.nodes[source_id]["times"]
    
    tgt_words = G.nodes[target_id]["words"]
    tgt_times = G.nodes[target_id]["times"]

    source_vec = lcs.Vector1D([int(x) for x in src_words])
    target_vec = lcs.Vector1D([int(x) for x in tgt_words])
    
    cutoffs = np.searchsorted(tgt_times, src_times, side='right')
    
    search_indices = [(i, int(c)) for i, c in enumerate(cutoffs)]
    l_t = lcs.Vector2D(search_indices)
    
    finder = lcs.LCSFinder(source_vec, target_vec)
    matches = finder.ComputeAllLCSs(l_t)
    matches_adjusted = [m + 1 for m in matches]
    entropy=len(src_words)*np.log2(len(tgt_words))/sum(matches_adjusted)
    return entropy