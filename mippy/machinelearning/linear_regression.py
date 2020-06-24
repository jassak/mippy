import numpy as np
from addict import Dict

from mippy.worker import Worker
from mippy.master import Master
from mippy.parameters import get_parameters
from mippy.expressions import new_design_matrix

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
        X = new_design_matrix(
            "1 leftententorhinalarea leftptplanumtemporale leftprgprecentralgyrus"
        )
        y = new_design_matrix("lefthippocampus")
        G = self.workers.eval(X.T @ X).sum()
        M = self.workers.eval(X.T @ y).sum()
        covariance = np.linalg.inv(G)
        coeff = covariance @ M
        print("Done!\n")
        print(f"model coefficients = \n{coeff}")


class LinearRegressionWorker(Worker):
    pass


if __name__ == "__main__":
    parameters = get_parameters(properties)

    import time

    s = time.perf_counter()
    LinearRegressionMaster(parameters).run()
    elapsed = time.perf_counter() - s
    print(f"\nExecuted in {elapsed:0.3f} seconds.")
