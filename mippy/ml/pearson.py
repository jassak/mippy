import Pyro5.api
import numpy as np
from addict import Dict
from scipy import special

from mippy.baseclasses import Master, Worker
from mippy.parameters import get_parameters
import mippy.reduce as reduce

__all__ = ["PearsonWorker", "PearsonMaster"]

properties = Dict(
    {
        "name": "pca",
        "parameters": {
            "columns": {
                "variables": {
                    "names": [
                        "lefthippocampus",
                        "leftocpoccipitalpole",
                        "leftscasubcallosalarea",
                        "righthippocampus",
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


class PearsonMaster(Master):
    def run(self):
        n_obs = self.nodes.get_num_obs()
        sx, sxx, sxy, sy, syy = self.nodes.get_local_sums()
        df = n_obs - 2
        d = (
            np.sqrt(n_obs * sxx - sx * sx)
            * np.sqrt(n_obs * syy - sy * sy)[:, np.newaxis]
        )
        r = (n_obs * sxy - sx * sy[:, np.newaxis]) / d
        r[d == 0] = 0
        r = r.clip(-1, 1)
        t_squared = r ** 2 * (df / ((1.0 - r) * (1.0 + r)))
        prob = special.betainc(
            0.5 * df, 0.5, np.fmin(np.asarray(df / (df + t_squared)), 1.0)
        )
        prob[abs(r) == 1] = 0
        print("Done!\n")
        print(f"correlations = \n{r}\n")
        print(f"p-values = \n{prob}\n")


class PearsonWorker(Worker):
    @Pyro5.api.expose
    @reduce.rules("add", "add", "add", "add", "add")
    def get_local_sums(self):
        X = self.get_design_matrix(self.params.columns.variables, intercept=False)
        Y = X = np.array(X)
        sx = X.sum(axis=0)
        sy = Y.sum(axis=0)
        sxx = (X ** 2).sum(axis=0)
        sxy = (X * Y.T[:, :, np.newaxis]).sum(axis=1)
        syy = (Y ** 2).sum(axis=0)
        return sx, sxx, sxy, sy, syy


if __name__ == "__main__":
    parameters = get_parameters(properties)

    import time

    s = time.perf_counter()
    PearsonMaster(parameters).run()
    elapsed = time.perf_counter() - s
    print(f"\nExecuted in {elapsed:0.3f} seconds.")
