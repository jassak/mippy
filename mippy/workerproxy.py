import operator
import functools
from typing import List, Set, Optional

import numpy as np
import Pyro5.api
from addict import Dict

from mippy.reduce import operators


__all__ = ["WorkerProxy", "WorkerPool"]


class WorkerProxy:
    def __init__(self, server_name: str, *, params: Dict, worker_kind: str):
        self._proxy = Pyro5.api.Proxy(f"PYRONAME:{server_name}")
        self.worker_kind = worker_kind
        self._datasets: Optional[Set[str]] = None
        self.server_name = server_name
        self.params = params

    def __getattr__(self, method):
        return functools.partial(self._run, method)

    def _run(self, method: str, *args, **kwargs):
        return self._proxy.run_on_worker(
            self.params, self.worker_kind, method, *args, **kwargs
        )

    @property
    def datasets(self) -> Set[str]:
        if self._datasets:
            return self._proxy.get_datasets()
        else:
            self._datasets = self._proxy.get_datasets()
            return self._datasets  # type: ignore  # mypy is confused somehow


class WorkerPool:
    def __init__(self, server_names: List[str], params: Dict, master: str):
        input_datasets = set(params.datasets)
        self._datasets = None
        self.worker_kind = master.replace("Master", "Worker")
        self._workers = [
            worker
            for name in server_names
            if contains_any_dataset(
                worker := WorkerProxy(
                    name, params=params, worker_kind=self.worker_kind
                ),
                datasets=input_datasets,
            )
        ]
        if missing := input_datasets - self.datasets:
            msg = f"Dataset(s) '{missing}' cannot be found on any server."
            raise ValueError(msg)

    def __len__(self):
        return len(self._workers)

    def __iter__(self):
        return iter(self._workers)

    def __getitem__(self, item):
        return self._workers[item]

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
    def datasets(self) -> Set[str]:
        if self._datasets:
            return self._datasets
        else:
            self._datasets = functools.reduce(
                operator.or_, (node.datasets for node in self)
            )
            return self._datasets  # type: ignore  # mypy is confused somehow

    def eval(self, expr):
        return self._run("eval", str(expr), expr.mocks, expr.arrays)

    def reduce(self, result, method):
        from mippy.machinelearning import reduction_rules

        rules = reduction_rules[self.worker_kind][method]
        merged = [
            functools.reduce(operators[rule], res) for rule, res in zip(rules, result)
        ]
        if len(merged) == 1:
            return merged[0]
        return merged


def contains_any_dataset(worker: WorkerProxy, datasets: Set[str]):
    if datasets == "all":
        return True
    return bool(worker.datasets & datasets)
