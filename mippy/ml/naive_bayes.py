import Pyro5.api
from addict import Dict
import pprint

from mippy.baseclasses import Master, Worker
from mippy.parameters import get_parameters
import mippy.reduce as reduce

__all__ = ["NaiveBayesWorker", "NaiveBayesMaster"]

properties = Dict(
    {
        "name": "naive bayes",
        "parameters": {
            "columns": {
                "target": {
                    "names": ["alzheimerbroadcategory"],
                    "required": True,
                    "types": ["categorical"],
                },
                "features": {
                    "names": ["gender"],
                    "required": True,
                    "types": ["categorical"],  # for now only multinomial naive bayes
                },
            },
            "alpha": {"value": 0.5, "required": True, "types": ["float"]},
            "datasets": ["adni", "ppmi", "edsd"],
            "filter": None,
        },
    }
)


class NaiveBayesMaster(Master):
    def run(self):
        alpha = self.params.alpha.value
        target_counts, pair_counts = self.nodes.get_counts()
        n_obs = sum(count for count in target_counts.values())
        theta = {}
        for key, count in pair_counts.items():
            target_count = target_counts[key[0]]
            parameter = (count + alpha) / (target_count + n_obs * alpha)
            theta[key] = parameter

        pp = pprint.PrettyPrinter(indent=4)
        print("\nDone!\n")
        print("model parameres = \n")
        pp.pprint(theta)


class NaiveBayesWorker(Worker):
    @Pyro5.api.expose
    @reduce.rules("add_dict", "add_dict")
    def get_counts(self):
        X = self.get_design_matrix(
            self.params.columns.features + self.params.columns.target, intercept=False
        )
        target = self.params.columns.target[0]

        target_counts = {}
        for group in X.groupby(target):
            key = target + ": " + group[0]
            count = len(group[1])
            target_counts[key] = count

        pair_counts = {}
        for feature in self.params.columns.features:
            for group in X.groupby(target):
                for subgroup in group[1].groupby(feature):
                    key = (target + ": " + group[0], feature + ": " + subgroup[0])
                    count = len(subgroup[1])
                    pair_counts[key] = count
        return target_counts, pair_counts


if __name__ == "__main__":
    parameters = get_parameters(properties)

    import time

    s = time.perf_counter()
    NaiveBayesMaster(parameters).run()
    elapsed = time.perf_counter() - s
    print(f"\nExecuted in {elapsed:0.3f} seconds.")
