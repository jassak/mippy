import operator
import functools
from typing import List, Set, Optional

import numpy as np
import Pyro5.api
from addict import Dict

# from mippy.reduce import operators


__all__ = ["WorkerProxy", "WorkerPool"]


class WorkerProxy:
    def __init__(self, server_name: str, *, params: Dict):
        self._proxy = Pyro5.api.Proxy(f"PYRONAME:{server_name}")
        self._datasets: Optional[Set[str]] = None
        self.server_name = server_name
        self.params = params

    def eval(self, expr):
        return self._proxy.run_on_worker(
            self.params, "eval", str(expr), expr.mocks, expr.arrays
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
                worker := WorkerProxy(name, params=params), datasets=input_datasets,
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

    def eval(self, expr):
        result = [node.eval(expr) for node in self]
        result = [np.array(r) if isinstance(r, list) else r for r in result]
        return WorkersResult(result)

    @property
    def datasets(self) -> Set[str]:
        if self._datasets:
            return self._datasets
        else:
            self._datasets = functools.reduce(
                operator.or_, (node.datasets for node in self)
            )
            return self._datasets  # type: ignore  # mypy is confused somehow


class WorkersResult:
    def __init__(self, results):
        self._results = results

    def __repr__(self):
        return repr(self._results)

    def __getitem__(self, item):
        return self._results[item]

    def __len__(self):
        return len(self._results)

    def __iter__(self):
        return iter(self._results)

    def sum(self):
        return sum(self._results)

    def max(self):
        return max(self._results)

    def min(self):
        return min(self._results)

    def reduce(self, op):
        return functools.reduce(op, self._results)


def contains_any_dataset(worker: WorkerProxy, datasets: Set[str]):
    if datasets == "all":
        return True
    return bool(worker.datasets & datasets)
