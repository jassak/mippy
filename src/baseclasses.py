import re
from abc import ABC, abstractmethod

import Pyro4
import pandas as pd
from addict import Dict
from workingnodes import WorkingNodes

n_nodes = 3


class Master(ABC):
    def __init__(self, params: Dict):
        self.name = self.get_name()
        self.nodes = WorkingNodes(
            [f"local_node{i}" for i in range(n_nodes)], params.datasets
        )
        self.nodes.set_workers(params)
        self.params = params

    def __repr__(self):
        cls = type(self).__name__
        return f"{cls}({self.params})"

    def get_name(self):
        name = type(self).__name__.replace("Master", "")
        pattern = re.compile(r"(?<!^)(?=[A-Z])")
        return pattern.sub("_", name).lower()

    @abstractmethod
    def run(self):
        """Main execution of algorithm. Should be implemented in child classes."""


class Worker(ABC):
    def __init__(self, idx: int):
        self.idx = idx
        self.params = None
        self.data = None

    def __repr__(self):
        cls = type(self).__name__
        return f"{cls}({self.idx})"

    def load_data(self, parameters: Dict, db):
        self.params = parameters
        self.data = db.read_data_from_db(parameters)

    @Pyro4.expose
    def get_num_obs(self) -> int:
        return len(self.data)

    def get_design_matrix(self, columns, intercept=True):
        X = self.data[columns]
        if intercept:
            X["Intercept"] = 1
            cols = list(X)
            cols.insert(0, cols.pop(cols.index("Intercept")))  # Move intercept to front
            X = X.loc[:, cols]
        return X

    def get_target_column(self, target, outcome):
        y = self.data[target]
        y = pd.get_dummies(y)
        outcome = target[0] + "_" + outcome
        y = y[outcome]
        return y
