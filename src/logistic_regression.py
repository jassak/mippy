from typing import Tuple

import Pyro4
import numpy as np
import pandas as pd
from addict import Dict
from scipy.special import expit, xlogy

from nodes import Master, Worker
from parameters import get_parameters

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
            "datasets": ["adni"],
            "filter": {"alzheimerbroadcategory": ["CN", "AD"]},
        },
    }
)


class LogisticRegressionMaster(Master):
    def __init__(self, params: Dict):
        super().__init__(params.datasets)
        self.nodes.set_workers(params)
        pass

    def run(self):
        n_feat = self.nodes.get_num_features()[0]
        n_obs = sum(self.nodes.get_num_obs())
        coeff, ll = self.init_model(n_feat, n_obs)
        while True:
            print(f"loss: {-ll}")
            res = self.nodes.get_local_parameters(coeff.tolist())
            grad, hess, ll_new = self.merge_local_results(res)

            coeff = self.update_coefficients(grad, hess)
            if abs((ll - ll_new) / ll) <= 1e-6:
                break
            ll = ll_new
        print(f"\nDone!\n  loss= {-ll},\n  model coefficients = {coeff}")

    @staticmethod
    def merge_local_results(res: list) -> Tuple:
        grad = sum(np.array(r[0]) for r in res)
        hess = sum(np.array(r[1]) for r in res)
        ll_new = sum(r[2] for r in res)
        return grad, hess, ll_new

    @staticmethod
    def init_model(n_feat: int, n_obs: int) -> Tuple:
        ll = -2 * n_obs * np.log(2)
        coeff = np.zeros(n_feat + 1)
        return coeff, ll

    @staticmethod
    def update_coefficients(grad: np.ndarray, hess: np.ndarray):
        covariance = np.linalg.inv(hess)
        coeff = covariance @ grad
        return coeff


class LogisticRegressionWorker(Worker):
    def __init__(self, idx: int):
        super().__init__(idx)

    def prepare_data(self) -> Tuple:
        X = self.data[self.params["columns"]["features"]]
        X["Intercept"] = 1
        X = X[[X.columns[-1]] + X.columns[:-1].tolist()]
        X = np.array(X)
        y = self.data[self.params.columns.target]
        y = np.array(pd.get_dummies(y).iloc[:, 0])
        return X, y

    @Pyro4.expose
    def get_num_features(self) -> int:
        return len(self.params["columns"]["features"])

    @Pyro4.expose
    def get_num_obs(self) -> int:
        return len(self.data)

    @Pyro4.expose
    def get_local_parameters(self, coeff: np.ndarray) -> Tuple:
        coeff = np.array(coeff)
        X, y = self.prepare_data()

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


if __name__ == "__main__":
    parameters = get_parameters(properties)

    import time

    s = time.perf_counter()
    LogisticRegressionMaster(parameters).run()
    elapsed = time.perf_counter() - s
    print(f"\nExecuted in {elapsed:0.2f} seconds.")
