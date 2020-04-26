from pathlib import Path
from typing import Mapping

import Pyro4
import Pyro4.errors
from addict import Dict

from database import DataBase
from logistic_regression import LogisticRegressionWorker
from nodes import n_nodes

db_root = Path(__file__).parent.parent / "dbs"


class LocalNode:
    def __init__(self, idx: int):
        self.idx = idx
        print(f"Started server on node {idx}")
        db_path = db_root / f"local_dataset{idx}.db"
        self.db = DataBase(db_path=db_path)
        self.worker = None
        self.datasets = self.db.get_datasets()

    @Pyro4.expose
    def set_worker(self, params: Mapping):
        params = Dict(params)
        self.worker = LogisticRegressionWorker(idx=self.idx)
        self.worker.load_data(params, self.db)

    @Pyro4.expose
    def run_on_worker(self, method: str, *args, **kwargs):
        return getattr(self.worker, method)(*args, **kwargs)

    @Pyro4.expose
    def get_datasets(self):
        return self.datasets


def start_server():
    daemon = Pyro4.Daemon()
    ns = Pyro4.locateNS()
    [
        ns.register(f"local_node{i}", daemon.register(LocalNode(i)))
        for i in range(n_nodes)
    ]
    daemon.requestLoop()


if __name__ == "__main__":
    start_server()
