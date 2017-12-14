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
from .product import Product
from .scalar import Scalar
from .state import State
from .thalamus import Thalamus
from .transcode import Transcode
from nengo_spa import ast
from nengo_spa import ast_dynamic


def register_default_modules():
    ast.DotProduct.DotProductRealization = Compare
    ast.Product.BindRealization = Bind
    ast.Product.ProductRealization = Product
    ast.ActionSet.BasalGangliaRealization = BasalGanglia
    ast.ActionSet.ThalamusRealization = Thalamus

    ast_dynamic.BasalGangliaRealization = BasalGanglia
    ast_dynamic.BindRealization = Bind
    ast_dynamic.DotProductRealization = Compare
    ast_dynamic.ProductRealization = Product
    ast_dynamic.ScalarRealization = Scalar
    ast_dynamic.StateRealization = State
    ast_dynamic.ThalamusRealization = Thalamus


register_default_modules()
