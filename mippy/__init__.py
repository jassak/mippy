from pathlib import Path


root = Path(__file__).parent.parent
n_nodes = 3

import mippy.baseclasses
import mippy.database
import mippy.exceptions
import mippy.localnode
import mippy.parameters
import mippy.workingnodes

__all__ = ["root", "n_nodes"]
__all__.extend(baseclasses.__all__)
__all__.extend(database.__all__)
__all__.extend(exceptions.__all__)
__all__.extend(localnode.__all__)
__all__.extend(parameters.__all__)
__all__.extend(workingnodes.__all__)
