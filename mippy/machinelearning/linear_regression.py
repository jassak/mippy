from typing import Tuple

import Pyro5.api
import numpy as np
from addict import Dict

from mippy.worker import Worker
from master import Master
from mippy.parameters import get_parameters
import mippy.reduce as reduce

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
            "datasets": ["adni", "ppmi", "edsd"],
            "filter": None,
        },
    }
)


class LinearRegressionMaster(Master):
    def run(self):
        gramian, moment_matrix = self.workers.get_gramian_and_moment_matrix()
        covariance = np.linalg.inv(gramian)
        coeff = covariance @ moment_matrix
        print("Done!\n")
        print(f"model coefficients = \n{coeff}")


class LinearRegressionWorker(Worker):
    @Pyro5.api.expose
    @reduce.rules("add", "add")
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
