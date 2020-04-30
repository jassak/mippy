import operator
from functools import partial, reduce
from typing import List, Set

import Pyro4
from Pyro4.errors import CommunicationError
from Pyro4.errors import NamingError
from addict import Dict
from mippy.exceptions import NodeError


__all__ = ["WorkingNode", "WorkingNodes"]


class WorkingNode:
    def __init__(self, name: str, params: Dict):
        self._proxy = Pyro4.Proxy(f"PYRONAME:{name}")
        self._datasets = None
        self.name = name
        self.params = params
        self.task = params.task

    def __getattr__(self, method):
        return partial(self._run, method)

    def _run(self, method: str, *args, **kwargs):
        return self._proxy.run_on_worker(
            self.params, self.task, method, *args, **kwargs
        )

    @property
    def datasets(self) -> set:
        if self._datasets:
            return self._proxy.get_datasets()
        else:
            self._datasets = self._proxy.get_datasets()
            return self._datasets

    def run_on_worker(self, method: str, *args, **kwargs):
        return getattr(self._proxy.worker, method)(*args, **kwargs)


class WorkingNodes:
    def __init__(self, names: List[str], params: Dict):
        input_datasets = set(params.datasets)
        self._datasets = None
        self._nodes = [
            node
            for name in names
            if contains_any_dataset(node := WorkingNode(name, params), input_datasets)
        ]
        if missing := input_datasets - self.datasets:
            msg = f"Datasets '{missing}' cannot be found on any node."
            raise ValueError(msg)

    def __len__(self):
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes)

    def __getitem__(self, item):
        return self._nodes[item]

    def __getattr__(self, method):
        return partial(self._run, method)

    def _run(self, method: str, *args, **kwargs):
        return [getattr(node, method)(*args, **kwargs) for node in self]

    @property
    def datasets(self) -> set:
        if self._datasets:
            return self._datasets
        else:
            self._datasets = reduce(operator.or_, (node.datasets for node in self))
            return self._datasets


def contains_any_dataset(node: WorkingNode, datasets: Set[str]):
    if datasets == "all":
        return True
    return bool(node.datasets & datasets)
