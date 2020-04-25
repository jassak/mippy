import Pyro4

from localnode import n_nodes
from local_nodes import LocalNodes


class CentralNode:
    def __init__(self, datasets):
        self.nodes = LocalNodes([f"PYRONAME:local_node{i}" for i in range(n_nodes)], datasets)
