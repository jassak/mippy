from typing import Mapping, Type

import Pyro4
import Pyro4.errors
from addict import Dict
from mippy.database import DataBase
from mippy.baseclasses import Worker
from mippy import root, n_nodes
from mippy.ml.logistic_regression import LogisticRegressionWorker
from mippy.ml.pca import PCAWorker
from mippy.ml.naive_bayes import NaiveBayesWorker
from mippy.ml.linear_regression import LinearRegressionWorker

__all__ = ["LocalNode", "start_server"]

workers = {
    "logistic regression": LogisticRegressionWorker,
    "pca": PCAWorker,
    "naive bayes": NaiveBayesWorker,
    "linear regression": LinearRegressionWorker,
}
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
    for i in range(n_nodes):
        ns.register(f"local_node{i}", daemon.register(LocalNode(i)))
    daemon.requestLoop()


if __name__ == "__main__":
    start_server()
