import Pyro4
import numpy as np
from addict import Dict
from mippy.baseclasses import Master, Worker
from mippy.parameters import get_parameters
import mippy.reduce as reduce

__all__ = ["KMeansMaster", "KMeansWorker"]

from mippy.reduce import Mean

properties = Dict(
    {
        "name": "kmeans",
        "parameters": {
            "columns": {
                "features": {
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
            "k": 2,
        },
    }
)


class KMeansMaster(Master):
    def run(self):
        k = self.params.k
        means = self.nodes.init_means(k)
        for _ in range(10):
            means_new = self.nodes.update_means(means)
            means_new = [(means_new[i][0] / means_new[i][1]).tolist() for i in range(k)]
            print(means_new)
            if (
                max(np.linalg.norm(np.array(means) - np.array(means_new), axis=1))
                < 1e-4
            ):
                break
            means = means_new


class KMeansWorker(Worker):
    @Pyro4.expose
    @reduce.rules(None)
    def init_means(self, k: int):
        X = self.get_design_matrix(self.params.columns.features, intercept=False)
        X = np.array(X)
        return X[:k]

    @Pyro4.expose
    @reduce.rules("mediants")
    def update_means(self, means):
        k = len(means)
        means = np.array(means)
        X = self.get_design_matrix(self.params.columns.features, intercept=False)
        X = np.array(X)
        dist = np.linalg.norm(X - means[:, np.newaxis], axis=2)
        cluster_idx = dist.argmin(axis=0)
        sums = [X[cluster_idx == i].sum(axis=0) for i in range(k)]
        counts = [len(X[cluster_idx == i]) for i in range(k)]
        means = [Mean(s.tolist(), c) for s, c in zip(sums, counts)]
        return means


if __name__ == "__main__":
    parameters = get_parameters(properties)

    import time

    s = time.perf_counter()
    KMeansMaster(parameters).run()
    elapsed = time.perf_counter() - s
    print(f"\nExecuted in {elapsed:0.3f} seconds.")
