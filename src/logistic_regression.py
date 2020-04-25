import numpy as np
import pandas as pd
from scipy.special import expit, xlogy
from addict import Dict
import Pyro4

from localnode import LocalNode, start_server, n_nodes
from centralnode import CentralNode

properties = Dict(
    {
        "name": "Logistic Regression",
        "parameters": {
            "columns": {
                "target": {
                    "names": ["alzheimerbroadcategory"],
                    "required": True,
                    "types": ["categorical"],
                },
                "features": {
                    "names": ["lefthippocampus"],
                    "required": True,
                    "types": ["numerical"],
                },
            },
            "dataset": ["adni"],
            "filter": {"alzheimerbroadcategory": ["CN", "AD"]},
        },
    }
)

n_obs = 1000
n_cols = len(properties.parameters.columns.features.names) + 1
ntot_obs = n_obs * n_nodes


class LogisticRegressionCentral(CentralNode):
    def __init__(self, datasets):
        super().__init__(datasets)

    def run(self):
        coeff, ll = self.init_model(n_cols, ntot_obs)
        while True:
            print(f"loss: {-ll}")
            res = self.nodes.run("get_local_parameters", coeff.tolist())
            grad, hess, ll_new = self.merge_local_results(res)

            coeff = self.update_coefficients(grad, hess)
            if abs((ll - ll_new) / ll) <= 1e-6:
                break
            ll = ll_new
        print(f"\nDone!\n  loss= {-ll},\n  model coefficients = {coeff}")

    @staticmethod
    def merge_local_results(res: list):
        grad = sum(np.array(r[0]) for r in res)
        hess = sum(np.array(r[1]) for r in res)
        ll_new = sum(r[2] for r in res)
        return grad, hess, ll_new

    @staticmethod
    def init_model(n_cols: int, n_obs: int):
        ll = -2 * n_obs * np.log(2)
        coeff = np.zeros(n_cols)
        return coeff, ll

    @staticmethod
    def update_coefficients(grad: np.ndarray, hess: np.ndarray):
        covariance = np.linalg.inv(hess)
        coeff = covariance @ grad
        return coeff


class LogisticRegressionLocal(LocalNode):
    def __init__(self, node):
        super().__init__(node)
        data = self.db.read_data_from_db(properties.parameters)
        self.prepare_data(data)

    def prepare_data(self, data: pd.DataFrame):
        self.X = data[properties.parameters.columns.features.names]
        self.X["Intercept"] = 1
        self.X = self.X[[self.X.columns[-1]] + self.X.columns[:-1].tolist()]
        self.X = np.array(self.X)
        self.y = data[properties.parameters.columns.target.names]
        self.y = np.array(pd.get_dummies(self.y).iloc[:, 0])

    @Pyro4.expose
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

    @Pyro4.expose
    def get_datasets(self):
        return ["adni", "ppmi", "edsd"]


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        help="Mode should be server (for local nodes) of client (for central node).",
    )
    args = parser.parse_args(sys.argv[1:])
    if args.mode == "server":
        start_server(local_node=LogisticRegressionLocal)
    elif args.mode == "client":
        import time

        s = time.perf_counter()
        LogisticRegressionCentral("adni").run()
        elapsed = time.perf_counter() - s
        print(f"\nExecuted in {elapsed:0.2f} seconds.")
