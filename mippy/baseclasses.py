from abc import ABC, abstractmethod
from typing import List, Tuple

import Pyro4
import numpy as np
import pandas as pd
from addict import Dict
from mippy import n_nodes
from mippy.workingnodes import WorkingNodes

__all__ = ["Master", "Worker"]


class Master(ABC):
    def __init__(self, params: Dict):
        self.nodes = WorkingNodes([f"local_node{i}" for i in range(n_nodes)], params)
        self.params = params

    def __repr__(self):
        cls = type(self).__name__
        return f"{cls}({self.params})"

    @abstractmethod
    def run(self):
        """Main execution of algorithm. Should be implemented in child classes."""

    @staticmethod
    def sum_local_arrays(res: list) -> Tuple:
        merged = []
        for i in range(len(res[0])):
            merged.append(sum(np.array(r[i]) for r in res))
        return tuple(merged)

    @staticmethod
    def sum_local_dictionary(res: list):
        merged_dict = {}
        for key in res[0].keys():
            total = sum(d[key] for d in res)
            merged_dict[key] = total
        return merged_dict

    def sum_local_dictionaries(self, res):
        merged = []
        for i in range(len(res[0])):
            lst = []
            for j in range(len(res)):
                lst.append(res[j][i])
            merged.append(self.sum_local_dictionary(lst))
        return merged


class Worker(ABC):
    def __init__(self, idx: int):
        self.idx = idx
        self.params = None
        self.data = None

    def __repr__(self):
        cls = type(self).__name__
        return f"{cls}({self.idx})"

    def load_data(self, parameters: Dict, db) -> None:
        self.params = parameters
        self.data = db.read_data(parameters)

    @Pyro4.expose
    def get_num_obs(self) -> int:
        return len(self.data)

    def get_design_matrix(
        self, columns: List[str], intercept: bool = True
    ) -> pd.DataFrame:
        X = self.data[columns]
        if intercept:
            X["Intercept"] = 1
            cols = list(X)
            cols.insert(0, cols.pop(cols.index("Intercept")))  # Move intercept to front
            X = X.loc[:, cols]
        return X

    def get_target_column(self, target: List[str], outcome: str) -> pd.DataFrame:
        y = self.data[target]
        y = pd.get_dummies(y)
        outcome = target[0] + "_" + outcome
        y = y[outcome]
        return y