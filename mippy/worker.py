import re
from abc import ABC
from typing import List, Optional

import Pyro5.api
import numpy as np
import pandas as pd
from addict import Dict
import mippy.reduce as reduce

__all__ = ["Worker"]


class Worker(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.params: Optional[Dict] = None
        self.data: Optional[pd.DataFrame] = None

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}({self.name})"

    def load_data(self, parameters: Dict, db) -> None:
        self.params = parameters
        self.data = db.read_data(parameters)

    @Pyro5.api.expose
    @reduce.rules("add")
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

    @Pyro5.api.expose
    @reduce.rules("add")
    def eval(self, expr, registry):
        t = dict()
        for key, columns in registry.items():
            if "1" in columns:
                columns.remove("1")
                t[key] = np.array(self.get_design_matrix(columns, intercept=True))
            else:
                t[key] = np.array(self.get_design_matrix(columns, intercept=False))
        expr = re.sub("(t_[0-9]+)", "t['\g<1>']", expr)
        res = eval(expr)
        return res.tolist()
