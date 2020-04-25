from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, xlogy
import Pyro4

from logistic_regression import properties, n_nodes
from database import DataBase

db_root = Path(__file__).parent.parent.parent / "dbs"


@Pyro4.expose
class LocalNode:
    def __init__(self, node: int):
        db_path = db_root / f"local_dataset{node}.db"
        db = DataBase(db_path=db_path)
        data = db.read_data_from_db(properties.parameters)
        self.prepare_data(data)

    def prepare_data(self, data: pd.DataFrame):
        self.X = data[properties.parameters.columns.features.names]
        self.X["Intercept"] = 1
        self.X = self.X[[self.X.columns[-1]] + self.X.columns[:-1].tolist()]
        self.X = np.array(self.X)
        self.y = data[properties.parameters.columns.target.names]
        self.y = np.array(pd.get_dummies(self.y).iloc[:, 0])

    def get_local_parameters(self, coeff: np.ndarray):
        coeff = np.array(coeff)
        X, y = self.X, self.y

        z = X @ coeff
        s = expit(z)
        d = s * (1 - s)
        D = np.diag(d)

        hess = X.T @ D @ X
        y_ratio = (y - s) / d
        y_ratio[(y == 0) & (s == 0)] = -1
        y_ratio[(y == 1) & (s == 1)] = 1

        grad = X.T @ D @ (z + y_ratio)

        ll = np.sum(xlogy(y, s) + xlogy(1 - y, 1 - s))
        return grad.tolist(), hess.tolist(), ll


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
