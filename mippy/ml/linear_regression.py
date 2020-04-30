from typing import Tuple

import Pyro4
import numpy as np
from addict import Dict

from mippy.baseclasses import Master, Worker
from mippy.parameters import get_parameters

__all__ = ["LinearRegressionMaster", "LinearRegressionWorker"]

properties = Dict(
    {
        "name": "linear regression",
        "parameters": {
            "columns": {
                "target": {
                    "names": ["lefthippocampus"],
                    "required": True,
                    "types": ["categorical"],
                },
                "features": {
                    "names": [
                        "leftententorhinalarea",
                        "leftptplanumtemporale",
                        "leftprgprecentralgyrus",
                    ],
                    "required": True,
                    "types": ["numerical"],
                },
            },
            "datasets": ["adni"],
            "filter": None,
        },
    }
)


class LinearRegressionMaster(Master):
    def run(self):
        res = self.nodes.get_gramian_and_moment_matrix()
        gramian, moment_matrix = self.sum_local_arrays(res)
        covariance = np.linalg.inv(gramian)
        coeff = covariance @ moment_matrix
        print("Done!\n")
        print(f"model coefficients = \n{coeff}")


class LinearRegressionWorker(Worker):
    @Pyro4.expose
    def get_gramian_and_moment_matrix(self) -> Tuple[list, list]:
        X = self.get_design_matrix(self.params.columns.features)
        y = self.get_design_matrix(self.params.columns.target, intercept=False)
        X, y = np.array(X), np.array(y)

        gramian = X.T @ X
        moment_matrix = X.T @ y
        return gramian, moment_matrix


if __name__ == "__main__":
    parameters = get_parameters(properties)

    import time

    s = time.perf_counter()
    LinearRegressionMaster(parameters).run()
    elapsed = time.perf_counter() - s
    print(f"\nExecuted in {elapsed:0.3f} seconds.")
