#!/usr/bin/env python

import numpy as np  # noqa F401
from addict import Dict

from mippy.parameters import get_parameters
from mippy.expressions import new_design_matrix, inv
from mippy.workerproxy import WorkerPool
from mippy.stackmachine import remote_eval
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
def linear_regression(workers, target, features):
    X = new_design_matrix(features, intercept=True)
    y = new_design_matrix(target)

    coeff = remote_eval(workers, inv(X.T @ X) @ (X.T @ y))

    return coeff


if __name__ == "__main__":
    parameters = get_parameters(properties)
    target = parameters.columns.target
    features = parameters.columns.features
    workers = WorkerPool(demo_servers, parameters)
    linear_regression(workers, target, features)
