import logging

import Pyro4

logging.basicConfig(level=logging.DEBUG)


class LocalNodesException(Exception):
    pass


class LocalNodes:
    def __init__(self, names, datasets):
        self.datasets = datasets
        self.nodes = [Pyro4.Proxy(name) for name in names]
        node_datasets = self.run("get_datasets")

        # Get only the nodes that are needed
        nodes_needed = []
        for node, node_datasets in zip(self.nodes, node_datasets):
            for dataset in node_datasets:
                if dataset in datasets:
                    nodes_needed.append(node)
                    break
        self.nodes = nodes_needed

    def run(self, method, *args, **kwargs):
        try:
            return [getattr(node, method)(*args, **kwargs) for node in self.nodes]

        except Pyro4.errors.CommunicationError as e:
            logging.error(e)
            raise LocalNodesException("Unresponsive node.")
        except AttributeError:
            raise LocalNodesException(f"Method could not be found: {method}")
