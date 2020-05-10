from typing import Mapping, Any

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

    def get_worker(self, params: Mapping) -> Worker:
        params = Dict(params)
        worker = Worker(name=self.name)
        worker.load_data(params, self.db)
        return worker

    @Pyro5.api.expose
    def run_on_worker(self, params: Mapping, method: Any, *args, **kwargs) -> Any:
        worker = self.get_worker(params)
        return getattr(worker, method)(*args, **kwargs)

    @Pyro5.api.expose
    def get_datasets(self) -> set:
        return self.datasets


def start_server(name: str) -> None:
    daemon = Pyro5.api.Daemon()
    ns = Pyro5.api.locate_ns()
    ns.register(f"local-server.{name}", daemon.register(Server(name)))
    daemon.requestLoop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--servername")
    args = parser.parse_args()
    start_server(args.servername)
