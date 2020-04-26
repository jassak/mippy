import itertools
import re
from functools import partial
from typing import List

import Pyro4
import Pyro4.errors
from addict import Dict

from exceptions import LocalNodesError

n_nodes = 3


class Master:
    def __init__(self, datasets: List[str]):
        self.name = self.get_name()
        self.nodes = LocalNodes([f"local_node{i}" for i in range(n_nodes)], datasets)

    def get_name(self):
        name = type(self).__name__.replace("Master", "")
        pattern = re.compile(r"(?<!^)(?=[A-Z])")
        return pattern.sub("_", name).lower()


class LocalNodes:
    def __init__(self, names: List[str], datasets: List[str]):
        self._nodes = [Pyro4.Proxy(f"PYRONAME:{name}") for name in names]
        node_datasets = self.get_datasets()
        valid = [
            set(datasets) & set(node_datasets)
            for node, node_datasets in zip(self._nodes, node_datasets)
        ]
        self._nodes = list(itertools.compress(self._nodes, valid))
        self.names = list(itertools.compress(names, valid))

    def __len__(self):
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes)

    def __getitem__(self, item):
        return self._nodes[item]

    def __getattr__(self, method):
        return partial(self._run, method)

    def _run(self, method: str, *args, **kwargs):
        try:
            return [node.run_on_worker(method, *args, **kwargs) for node in self]
        except Pyro4.errors.CommunicationError as e:
            raise LocalNodesError("Unresponsive node.")
        except AttributeError as e:
            raise LocalNodesError(f"Method could not be found: {method}")

    def set_workers(self, params):
        [node.set_worker(params) for node in self]

    def get_datasets(self):
        return [node.get_datasets() for node in self]


class Worker:
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
