# Rewire
Adaptation of the quoter model to evolving networks

An easily usable version of the Rewire model. Dependent on LCSFinder package which requires C++ build tools.
The core Idea of the rewire model is as follows:
1. Run the quoter model over an N node network for T time steps.
2. When every node has quoted on average m times, rewire one node by the following process:
   -Pick a random node, u.
   -Identify the neighbour with the highest cross-entropy, v.
   -Identify the non-neighbour with the lowest cross-entropy, w.
   -Switch u-v edge to u-w edge.

More information on the quoter model and cross-entropy estimate can be found in [Bagrow & Mitchell's 2018 Paper](https://pubs.aip.org/aip/cha/article/28/7/075304/386316/The-quoter-model-A-paradigmatic-model-of-the).
   

# Installation and Usage
Required packages can be installed as such:
`pip install networkx numpy matplotlib pandas LCSFinder`

# Rewire.Py
Contains functions for performing rewire network analysis.

# Test.py
Easily editable script that calls the rewire model and stores average clustering coefficient, global clustering coefficient, average path length, diameter, degree counts/distribution, and final graph in variables. Right now, this script prints the final average clustering coefficient, global clustering coefficient, and draws the final graph. Edit the parameters as you see fit.

WARNING: Due to how cross-entropy is estimated, these simulations will blow up for large values of T.
