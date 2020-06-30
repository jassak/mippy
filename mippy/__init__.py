from . import worker
from . import database
from . import server
from . import parameters
from . import workerproxy
from . import expressions

# from . import machinelearning


__all__ = []
__all__.extend(worker.__all__)
__all__.extend(database.__all__)
__all__.extend(server.__all__)
__all__.extend(parameters.__all__)
__all__.extend(workerproxy.__all__)
__all__.extend(expressions.__all__)
# __all__.extend(machinelearning.__all__)

demo_servers = [f"local-server.{name}" for name in ["serverA", "serverB", "serverC"]]
__all__.append("demo_servers")
