import re
from abc import ABC

from addict import Dict
from workingnodes import WorkingNodes

n_nodes = 3


class Master(ABC):
    def __init__(self, params: Dict):
        self.name = self.get_name()
        self.nodes = WorkingNodes(
            [f"local_node{i}" for i in range(n_nodes)], params.datasets
        )
        self.nodes.set_workers(params)

    def get_name(self):
        name = type(self).__name__.replace("Master", "")
        pattern = re.compile(r"(?<!^)(?=[A-Z])")
        return pattern.sub("_", name).lower()


class Worker(ABC):
    def __init__(self, idx: int):
        self.idx = idx
        self.params = None
        self.data = None

    def load_data(self, parameters: Dict, db):
        self.params = parameters
        self.data = db.read_data_from_db(parameters)

    def __repr__(self):
        cls = type(self).__name__
        return f"{cls}({self.idx})"
