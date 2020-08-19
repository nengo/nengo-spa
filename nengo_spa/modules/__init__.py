"""SPA modules with defined inputs and outputs usable in action rules.

SPA modules derive from `nengo_spa.Network` and thus provide information about
inputs and outputs that can be used in action rules and the associated
vocabularies. Note that a module might have no inputs and outputs that can be
used in action rules (like `BasalGanglia` and `Thalamus`), but only inputs and
outputs that must be manually connected to. Many SPA modules are networks that
might be automatically created by building action rules. Because of this, it is
possible to set module parameters with `nengo.Config` objects to allow to
easily change parameters of networks created in this way.

Note that SPA modules can be used as standalone networks without using
any other NengoSPA features.
"""

from nengo_spa.ast import dynamic as _dynamic

from .associative_memory import (
    AssociativeMemory,
    IAAssocMem,
    ThresholdingAssocMem,
    WTAAssocMem,
)
from .basalganglia import BasalGanglia
from .bind import Bind
from .compare import Compare
from .product import Product
from .scalar import Scalar
from .state import State
from .superposition import Superposition
from .thalamus import Thalamus
from .transcode import Transcode


def _register_default_modules():
    _dynamic.BasalGangliaRealization = BasalGanglia
    _dynamic.BindRealization = Bind
    _dynamic.DotProductRealization = Compare
    _dynamic.ProductRealization = Product
    _dynamic.ScalarRealization = Scalar
    _dynamic.StateRealization = State
    _dynamic.SuperpositionRealization = Superposition
    _dynamic.ThalamusRealization = Thalamus


_register_default_modules()
