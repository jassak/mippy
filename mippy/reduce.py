from collections import namedtuple
import numpy as np


operators = {
    None: lambda a, b: a,
    "add": lambda a, b: a + b,
    "add_dict": lambda a, b: {k: v + b[k] for k, v in a.items()},
    "mediants": lambda a, b: [
        Mean(np.array(ai[0]) + np.array(bi[0]), np.array(ai[1]) + np.array(bi[1]))
        for ai, bi in zip(a, b)
    ],
}


def rules(*rules_tup):
    def wrapper(method):
        method.rules = rules_tup
        return method

    return wrapper


Mean = namedtuple("Mean", "numerator denominator")
