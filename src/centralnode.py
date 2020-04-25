import Pyro4

from localnode import n_nodes


class CentralNode:
    def __init__(self):
        self.local_nodes = [
            Pyro4.Proxy(f"PYRONAME:local_node{i}") for i in range(n_nodes)
        ]
