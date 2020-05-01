from addict import Dict
from mippy.ml.logistic_regression import *
from mippy.ml.pca import *
from mippy.ml.naive_bayes import *
from mippy.ml.linear_regression import *
from mippy.ml.kmeans import *

from mippy.baseclasses import Worker

__all__ = []
__all__.extend(logistic_regression.__all__)
__all__.extend(pca.__all__)
__all__.extend(naive_bayes.__all__)
__all__.extend(linear_regression.__all__)
__all__.extend(kmeans.__all__)


def get_reduction_rules():
    rules = Dict()
    for worker_class in Worker.__subclasses__():
        for name, attr in vars(worker_class).items():
            try:
                rules[worker_class.__name__][name] = attr.rules
            except AttributeError:
                pass
        for name, attr in vars(Worker).items():
            try:
                rules[worker_class.__name__][name] = attr.rules
            except AttributeError:
                pass
    return rules


reduction_rules = get_reduction_rules()

__all__.append("reduction_rules")
