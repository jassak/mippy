import operator
import functools
from typing import List, Set

import numpy as np
import Pyro5.api
from addict import Dict

from mippy.reduce import operators


__all__ = ["WorkingNode", "WorkingNodes"]


class WorkingNode:
    def __init__(self, name: str, params: Dict):
        self._proxy = Pyro5.api.Proxy(f"PYRONAME:{name}")
        self._datasets = None
        self.name = name
        self.params = params
        self.task = params.task

    def __getattr__(self, method):
        return functools.partial(self._run, method)

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


class WorkingNodes:
    def __init__(self, names: List[str], params: Dict, master: str):
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
        self.master = master
        self.worker = master.replace("Master", "Worker")

    def __len__(self):
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes)

    def __getitem__(self, item):
        return self._nodes[item]

    def __getattr__(self, method):
        return functools.partial(self._run, method)

    def _run(self, method: str, *args, **kwargs):
        args = tuple(
            arg.tolist() if isinstance(arg, np.ndarray) else arg for arg in args
        )
        kwargs = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in kwargs.items()
        }
        result = [getattr(node, method)(*args, **kwargs) for node in self]
        result = [
            [np.array(n[i]) if isinstance(n[i], list) else n[i] for n in result]
            for i in range(len(result[0]))
        ]
        return self.reduce(result, method)

    @property
    def datasets(self) -> set:
        if self._datasets:
            return self._datasets
        else:
            self._datasets = functools.reduce(
                operator.or_, (node.datasets for node in self)
            )
            return self._datasets

    def reduce(self, result, method):
        from mippy.ml import reduction_rules  # import here to avoid cyclic imports

        rules = reduction_rules[self.worker][method]
        merged = [
            functools.reduce(operators[rule], res) for rule, res in zip(rules, result)
        ]
        if len(merged) == 1:
            return merged[0]
        return merged


def contains_any_dataset(node: WorkingNode, datasets: Set[str]):
    if datasets == "all":
        return True
    return bool(node.datasets & datasets)
