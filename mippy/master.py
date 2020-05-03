from abc import ABC, abstractmethod

from addict import Dict

from mippy import WorkerPool

server_names = ["serverA", "serverB", "serverC"]


class Master(ABC):
    def __init__(self, params: Dict) -> None:
        global server_names  # todo this should be a parameter
        cls = type(self).__name__
        self.workers = WorkerPool(
            [f"local-server.{name}" for name in server_names], params, master=cls
        )
        self.params = params

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}({self.params})"

    @abstractmethod
    def run(self) -> None:
        """Main execution of algorithm. Should be implemented in child classes."""
