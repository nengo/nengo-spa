"""SPA modules.

SPA modules derive from `nengo_spa.Network` and thus provide information about
inputs and outputs that can be used in action rules and the associated
vocabularies. Note that a module might have no inputs and outputs that can be
used in action rules (like `BasalGanglia` and `Thalamus`), but only inputs and
outputs that must be manually connected to. Many SPA modules are networks that
might be automatically created by building action rules. Because of this, it is
possible to set module parameters with `nengo.Config` objects to allow to
easily change parameters of networks created in this way.

Note that SPA modules can be used as standalone networks without using
nengo_spa features.
"""

from .assoc_mem import (
    AssociativeMemory,
    IAAssocMem,
    ThresholdingAssocMem,
    WTAAssocMem)
from .basalganglia import BasalGanglia
from .bind import Bind
from .compare import Compare
from .input import Input
from .product import Product
from .scalar import Scalar
from .state import State
from .thalamus import Thalamus