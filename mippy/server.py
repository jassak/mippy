from typing import Mapping, Type, Any

import numpy as np
import Pyro5.api
import Pyro5.server
from addict import Dict

from mippy.database import DataBase, root
from mippy.worker import Worker

__all__ = ["Server", "start_server"]


db_root = root / "dbs"


class Server:
    def __init__(self, name: str):
        self.name = name
        print(f"Starting server {name}")
        db_path = db_root / f"dataset-{name}.db"
        self.db = DataBase(db_path=db_path)
        self.datasets = self.db.get_datasets()
        self.workers = get_workers()

    def get_worker(self, params: Mapping, name: str) -> Type[Worker]:
        params = Dict(params)
        worker = self.workers[name](name=self.name)
        worker.load_data(params, self.db)
        return worker

    @Pyro5.api.expose
    def run_on_worker(
        self, params: Mapping, task: str, method: Any, *args, **kwargs
    ) -> Any:
        worker = self.get_worker(params, task)
        method = getattr(worker, method)
        if isinstance(results := method(*args, **kwargs), tuple):
            results = tuple(
                result.tolist() if isinstance(result, np.ndarray) else result
                for result in results
            )
        else:
            results = (
                results.tolist() if isinstance(results, np.ndarray) else results,
            )
        if len(results) != len(method.rules):
            raise ValueError("Method rules should match the number of return values.")
        return results

    @Pyro5.api.expose
    def get_datasets(self) -> set:
        return self.datasets


def start_server(name: str) -> None:
    daemon = Pyro5.api.Daemon()
    ns = Pyro5.api.locate_ns()
    ns.register(f"local-server.{name}", daemon.register(Server(name)))
    daemon.requestLoop()


def get_workers():
    import importlib

    importlib.import_module(name="machinelearning", package="mippy")
    return {w.__name__: w for w in Worker.__subclasses__()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--servername")
    args = parser.parse_args()
    start_server(args.servername)
