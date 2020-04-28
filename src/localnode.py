from typing import Mapping, Type

import Pyro4
import Pyro4.errors
from addict import Dict
from baseclasses import n_nodes, Worker
from database import DataBase
from logistic_regression import LogisticRegressionWorker
from pca import PCAWorker
from src import root

workers = {"logistic regression": LogisticRegressionWorker, "pca": PCAWorker}

db_root = root / "dbs"


class LocalNode:
    def __init__(self, idx: int):
        self.idx = idx
        print(f"Started server on node {idx}")
        db_path = db_root / f"local_dataset{idx}.db"
        self.db = DataBase(db_path=db_path)
        self.datasets = self.db.get_datasets()

    def get_worker(self, params: Mapping, task: str) -> Type[Worker]:
        params = Dict(params)
        worker = workers[task](idx=self.idx)
        worker.load_data(params, self.db)
        return worker

    @Pyro4.expose
    def run_on_worker(self, params: Mapping, task: str, method: str, *args, **kwargs):
        worker = self.get_worker(params, task)
        return getattr(worker, method)(*args, **kwargs)

    @Pyro4.expose
    def get_datasets(self):
        return self.datasets


def start_server() -> None:
    daemon = Pyro4.Daemon()
    ns = Pyro4.locateNS()
    [
        ns.register(f"local_node{i}", daemon.register(LocalNode(i)))
        for i in range(n_nodes)
    ]
    daemon.requestLoop()


if __name__ == "__main__":
    start_server()
