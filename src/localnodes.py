from functools import partial
import itertools
from typing import Type, List
from pathlib import Path
import Pyro4

from database import DataBase

n_nodes = 3
db_root = Path(__file__).parent.parent / "dbs"


class LocalNode:
    def __init__(self, idx: int):
        self.idx = idx
        print(f"Started server on node {idx}")
        db_path = db_root / f"local_dataset{idx}.db"
        self.db = DataBase(db_path=db_path)

    def __repr__(self):
        cls = type(self).__name__
        return f"{cls}({self.idx})"

    @Pyro4.expose
    def get_datasets(self):
        return ["adni", "ppmi", "edsd"]


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
            return [getattr(node, method)(*args, **kwargs) for node in self._nodes]
        except Pyro4.errors.CommunicationError as e:
            raise LocalNodesError("Unresponsive node.")
        except AttributeError as e:
            raise LocalNodesError(f"Method could not be found: {method}")


def start_server(local_node: Type[LocalNode]):
    daemon = Pyro4.Daemon()
    ns = Pyro4.locateNS()
    [
        ns.register(f"local_node{i}", daemon.register(local_node(i)))
        for i in range(n_nodes)
    ]
    daemon.requestLoop()


class LocalNodesError(Exception):
    pass
