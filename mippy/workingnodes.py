from functools import partial
from typing import List

import Pyro4
from Pyro4.errors import CommunicationError
from Pyro4.errors import NamingError
from addict import Dict
from .exceptions import WorkingNodeError
from .exceptions import WorkingNodesError

__all__ = ["WorkingNode", "WorkingNodes"]


class WorkingNode:
    def __init__(self, name: str, params: Dict):
        self._proxy = Pyro4.Proxy(f"PYRONAME:{name}")
        self.name = name
        self.params = params
        self.datasets = None
        self.task = params.task

    def __getattr__(self, method):
        return partial(self._run, method)

    def _run(self, method: str, *args, **kwargs):
        try:
            return self._proxy.run_on_worker(
                self.params, self.task, method, *args, **kwargs
            )
        except CommunicationError:
            raise WorkingNodeError(f"Unresponsive node {self.name}.")
        except NamingError:
            raise WorkingNodeError(f"Node {self.name} was removed.")
        except AttributeError:
            raise WorkingNodeError(f"Method {method} not found on node {self.name}.")

    def get_datasets(self) -> set:
        return self.datasets if self.datasets is not None else self._proxy.get_datasets()

    def run_on_worker(self, method: str, *args, **kwargs):
        return getattr(self._proxy.worker, method)(*args, **kwargs)


class WorkingNodes:
    def __init__(self, names: List[str], params: Dict):
        self._nodes = [WorkingNode(name, params, ) for name in names]

        if len(params.datasets) == 0:
            return

        all_datasets = [dataset for datasets in self.get_datasets() for dataset in datasets]
        for dataset in params.datasets:
            if dataset not in all_datasets:
                raise WorkingNodesError(f"Dataset '{dataset}' could not be found on the active nodes.")

        self._nodes = [
            node for node in self if contains_any_dataset(node, params.datasets)
        ]

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

    def get_datasets(self) -> List[set]:
        return [node.get_datasets() for node in self]


def contains_any_dataset(node: WorkingNode, datasets: List[str]):
    return bool(set(node.get_datasets()) & set(datasets))
