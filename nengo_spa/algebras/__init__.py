"""Algebras define the specific superposition and (un)binding operations."""

from .base import (
    AbstractAlgebra,
    CommonProperties,
    ElementSidedness,
    supports_sidedness,
)
from .hrr_algebra import HrrAlgebra, HrrProperties
from .tvtb_algebra import TvtbAlgebra, TvtbProperties
from .vtb_algebra import VtbAlgebra, VtbProperties
