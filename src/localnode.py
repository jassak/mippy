from typing import Type
from pathlib import Path
import Pyro4

from database import DataBase

n_nodes = 3
db_root = Path(__file__).parent.parent / "dbs"


class LocalNode:
    def __init__(self, node: int):
        print(f"Started server on node {node}")
        db_path = db_root / f"local_dataset{node}.db"
        self.db = DataBase(db_path=db_path)


def start_server(local_node: Type[LocalNode]):
    daemon = Pyro4.Daemon()
    ns = Pyro4.locateNS()
    [
        ns.register(f"local_node{i}", daemon.register(local_node(i)))
        for i in range(n_nodes)
    ]
    daemon.requestLoop()
