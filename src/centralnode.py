from typing import List
from localnodes import n_nodes, LocalNodes


class CentralNode:
    def __init__(self, datasets: List[str]):
        self.nodes = LocalNodes([f"local_node{i}" for i in range(n_nodes)], datasets)
