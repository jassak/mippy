import Pyro5.api
import numpy as np
from addict import Dict
from mippy.worker import Worker
from master import Master
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
        means = self.workers.init_means(k)
        for _ in range(10):
            means_new = self.workers.update_means(means)
            means_new = [(means_new[i][0] / means_new[i][1]).tolist() for i in range(k)]
            print(means_new)
            if max_euclidean_distance(means, means_new) < 1e-6:
                break
            means = means_new
        print("\nDone!")


class KMeansWorker(Worker):
    @Pyro5.api.expose
    @reduce.rules(None)
    def init_means(self, k: int):
        X = self.get_design_matrix(self.params.columns.features, intercept=False)
        X = np.array(X)
        return X[:k]

    @Pyro5.api.expose
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


def max_euclidean_distance(x, y):
    return max(np.linalg.norm(np.array(x) - np.array(y), axis=1))


if __name__ == "__main__":
    parameters = get_parameters(properties)

    import time

    s = time.perf_counter()
    KMeansMaster(parameters).run()
    elapsed = time.perf_counter() - s
    print(f"\nExecuted in {elapsed:0.3f} seconds.")
