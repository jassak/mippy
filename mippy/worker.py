from abc import ABC
from typing import List, Optional

import Pyro5.api
import numpy as np
import pandas as pd
import scipy.special  # noqa F401
from addict import Dict

import mippy.reduce as reduce
from mippy.stackmachine import StackMachine

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

    def get_design_matrix(
        self, columns: List[str], intercept: bool = True
    ) -> np.ndarray:
        X = self.data[columns]
        if intercept:
            X["Intercept"] = 1
            cols = list(X)
            cols.insert(0, cols.pop(cols.index("Intercept")))  # Move intercept to front
            X = X.loc[:, cols]
        return np.array(X)

    def get_target_column(self, target: List[str], outcome: str) -> np.ndarray:
        y = self.data[target]
        y = pd.get_dummies(y)
        outcome = target[0] + "_" + outcome
        y = y[outcome]
        return np.array(y)

    def materialize_mocks(self, mocks):
        m = dict()
        for key, matrix in mocks.items():
            if matrix["target_outcome"]:
                m[key] = self.get_target_column(
                    matrix["varnames"], matrix["target_outcome"]
                )
            else:
                if "1" in matrix["varnames"]:
                    matrix["varnames"].remove("1")
                    m[key] = self.get_design_matrix(matrix["varnames"], intercept=True)
                else:
                    m[key] = self.get_design_matrix(matrix["varnames"], intercept=False)
        return m

    def extract_arrays(self, arrays):
        a = dict()
        for key, value in arrays.items():
            a[key] = np.array(value)
        return a

    @Pyro5.api.expose
    @reduce.rules("add")
    def eval(self, expr, mocks, arrays):
        m = self.materialize_mocks(mocks)  # noqa F841
        a = self.extract_arrays(arrays)  # noqa F841
        res = eval(expr)
        if isinstance(res, np.ndarray):
            return res.tolist()
        else:
            return res

    @Pyro5.api.expose
    @reduce.rules("add")
    def eval_instructions(self, instructions, mocks):
        m = self.materialize_mocks(mocks)  # noqa F841
        machine = StackMachine(memory=m)
        res = machine.execute(instructions)
        if isinstance(res, np.ndarray):
            return res.tolist()
        else:
            return res
