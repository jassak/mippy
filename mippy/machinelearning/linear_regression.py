#!/usr/bin/env python

import numpy as np
from addict import Dict

from mippy.parameters import get_parameters
from mippy.expressions import new_design_matrix
from mippy.workerproxy import WorkerPool
from mippy.pretty import pretty

from mippy import demo_servers

__all__ = ["linear_regression"]

properties = Dict(
    {
        "name": "linear regression",
        "parameters": {
            "columns": {
                "target": {
                    "names": ["lefthippocampus"],
                    "required": True,
                    "types": ["numerical"],
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


@pretty
def linear_regression(workers):
    X = new_design_matrix(
        "1 leftententorhinalarea leftptplanumtemporale leftprgprecentralgyrus"
    )
    y = new_design_matrix("lefthippocampus")
    G = workers.eval(X.T @ X).sum()
    M = workers.eval(X.T @ y).sum()
    covariance = np.linalg.inv(G)
    coeff = covariance @ M
    return coeff


if __name__ == "__main__":
    parameters = get_parameters(properties)
    workers = WorkerPool(demo_servers, parameters)
    linear_regression(workers)
