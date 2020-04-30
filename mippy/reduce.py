operators = {
    None: lambda a, b: a,
    "add": lambda a, b: a + b,
    "add_dict": lambda a, b: {k: v + b[k] for k, v in a.items()},
}


def rules(*rules_tup):
    def wrapper(method):
        method.rules = rules_tup
        return method

    return wrapper
