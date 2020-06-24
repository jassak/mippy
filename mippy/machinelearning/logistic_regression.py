import numpy as np
from addict import Dict

from mippy.master import Master
from mippy.parameters import get_parameters
from mippy.expressions import (
    new_design_matrix,
    new_numpy_array,
    expit,
    diag,
    sum_,
    xlogy,
)

__all__ = ["LogisticRegressionMaster"]

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
        X = new_design_matrix("1 lefthippocampus")
        y = new_design_matrix("alzheimerbroadcategory", target_outcome="AD")
        n_feat = X.shape[1]
        n_obs = self.workers.eval(X.len()).sum()
        loglike = -2 * n_obs * np.log(2)
        coeff = new_numpy_array(np.zeros(n_feat))
        while True:
            print(f"loss: {-loglike}")
            z = X @ coeff
            s = expit(z)
            d = s * (1 - s)
            D = diag(d)
            loglike_new = sum_(xlogy(y, s) + xlogy(1 - y, 1 - s))
            loglike_new = self.workers.eval(loglike_new).sum()
            hess = self.workers.eval(X.T @ D @ X).sum()
            y_ratio = (y - s) / d
            grad = self.workers.eval(X.T @ D @ (z + y_ratio)).sum()
            covariance = np.linalg.inv(hess)
            coeff = covariance @ grad
            coeff = new_numpy_array(coeff)
            if abs((loglike - loglike_new) / loglike) <= 1e-6:
                break
            loglike = loglike_new

        print("\nDone!\n")
        print(f"loss = {-loglike}\n")
        print(f"model coefficients = \n{coeff.array}\n")


if __name__ == "__main__":
    parameters = get_parameters(properties)

    import time

    start = time.perf_counter()
    LogisticRegressionMaster(parameters).run()
    elapsed = time.perf_counter() - start
    print(f"\nExecuted in {elapsed:0.3f} seconds.")
