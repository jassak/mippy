from typing import Tuple

import Pyro4
import numpy as np
from scipy.special import expit, xlogy
from addict import Dict
from mippy.baseclasses import Master, Worker
from mippy.parameters import get_parameters
import mippy.reduce as reduce

__all__ = ["LogisticRegressionMaster", "LogisticRegressionWorker"]

properties = Dict(
    {
        "name": "logistic regression",
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
            "outcome": "AD",
        },
    }
)


class LogisticRegressionMaster(Master):
    def run(self):
        n_feat = self.nodes.get_num_features()
        n_obs = self.nodes.get_num_obs()
        coeff, loglike = self.init_model(n_feat, n_obs)
        while True:
            print(f"loss: {-loglike}")
            loglike_new, grad, hess = self.nodes.get_loss_function(coeff)
            coeff = self.update_coefficients(grad, hess)
            if abs((loglike - loglike_new) / loglike) <= 1e-6:
                break
            loglike = loglike_new
        print("\nDone!\n")
        print(f"loss = {-loglike}\n")
        print(f"model coefficients = \n{coeff}\n")

    @staticmethod
    def init_model(n_feat: int, n_obs: int) -> Tuple[np.ndarray, float]:
        ll = -2 * n_obs * np.log(2)
        coeff = np.zeros(n_feat + 1)
        return coeff, ll

    @staticmethod
    def update_coefficients(grad: np.ndarray, hess: np.ndarray) -> np.ndarray:
        covariance = np.linalg.inv(hess)
        coeff = covariance @ grad
        return coeff


class LogisticRegressionWorker(Worker):
    @Pyro4.expose
    @reduce.rules(None)
    def get_num_features(self) -> int:
        return len(self.params["columns"]["features"])

    @Pyro4.expose
    @reduce.rules("add", "add", "add")
    def get_loss_function(self, coeff: list) -> Tuple[float, list, list]:
        coeff = np.array(coeff)
        X = self.get_design_matrix(self.params.columns.features)
        y = self.get_target_column(self.params.columns.target, self.params.outcome)
        X, y = np.array(X), np.array(y)

        z = X @ coeff
        s = expit(z)
        d = s * (1 - s)
        D = np.diag(d)

        hess = X.T @ D @ X
        y_ratio = (y - s) / d
        y_ratio[(y == 0) & (s == 0)] = -1
        y_ratio[(y == 1) & (s == 1)] = 1

        grad = X.T @ D @ (z + y_ratio)

        loglike = float(np.sum(xlogy(y, s) + xlogy(1 - y, 1 - s)))
        return loglike, grad, hess


if __name__ == "__main__":
    parameters = get_parameters(properties)

    import time

    s = time.perf_counter()
    LogisticRegressionMaster(parameters).run()
    elapsed = time.perf_counter() - s
    print(f"\nExecuted in {elapsed:0.3f} seconds.")
