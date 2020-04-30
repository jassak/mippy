from pathlib import Path


root = Path(__file__).parent.parent
n_nodes = 3

from mippy.baseclasses import *
from mippy.database import *
from mippy.localnode import *
from mippy.parameters import *
from mippy.workingnodes import *

__all__ = ["root", "n_nodes"]
__all__.extend(baseclasses.__all__)
__all__.extend(database.__all__)
__all__.extend(localnode.__all__)
__all__.extend(parameters.__all__)
__all__.extend(workingnodes.__all__)
