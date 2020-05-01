import Pyro5.api
import numpy as np
from addict import Dict
from mippy.baseclasses import Master, Worker
from mippy.parameters import get_parameters
import mippy.reduce as reduce

__all__ = ["PCAWorker", "PCAMaster"]

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


class PCAMaster(Master):
    def run(self):
        n_obs = self.nodes.get_num_obs()
        sx, sxx = self.nodes.get_local_sums()
        means, sigmas = self.get_moments(n_obs, sx, sxx)
        gramian = self.nodes.get_standardized_gramian(means, sigmas)
        covariance = np.divide(gramian, n_obs - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        eigenvectors = eigenvectors.T
        print("Done!\n")
        print(f"eigenvalues = \n{eigenvalues}\n")
        print(f"eigenvectors = \n{eigenvectors}\n")

    @staticmethod
    def get_moments(n_obs, sx, sxx):
        means = sx / n_obs
        sigmas = ((sxx - n_obs * means ** 2) / (n_obs - 1)) ** 0.5
        return means, sigmas


class PCAWorker(Worker):
    @Pyro5.api.expose
    @reduce.rules("add", "add")
    def get_local_sums(self):
        X = self.get_design_matrix(self.params.columns.variables, intercept=False)
        X = np.array(X)
        sx = X.sum(axis=0)
        sxx = (X ** 2).sum(axis=0)
        return sx, sxx

    @Pyro5.api.expose
    @reduce.rules("add")
    def get_standardized_gramian(self, means, sigmas):
        means = np.array(means)
        sigmas = np.array(sigmas)
        X = self.get_design_matrix(self.params.columns.variables, intercept=False)
        X = np.array(X)
        if X.shape == (0, 0):
            return 0
        X -= means
        X /= sigmas
        gramian = np.dot(X.T, X)
        return gramian


if __name__ == "__main__":
    parameters = get_parameters(properties)

    import time

    s = time.perf_counter()
    PCAMaster(parameters).run()
    elapsed = time.perf_counter() - s
    print(f"\nExecuted in {elapsed:0.3f} seconds.")
