from addict import Dict


def get_parameters(properties: Dict) -> Dict:
    args = parse_args(properties)
    parameters = Dict()
    for name, column in properties.parameters.columns.items():
        if getattr(args, name):
            parameters.columns[name] = getattr(args, name).split(",")
        else:
            parameters.columns[name] = column.names
    for name, param in properties.parameters.items():
        if name == "columns":
            continue
        if getattr(args, name):
            parameters[name] = getattr(args, name).split(",")
        else:
            parameters[name] = param
    return parameters


def parse_args(properties: Dict):
    params = properties.parameters
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    for column in params.columns.keys():
        parser.add_argument(f"--{column}")
    for param in params.keys():
        if param == "columns":
            continue
        parser.add_argument(f"--{param}")
    args = parser.parse_args(sys.argv[1:])
    return args
