"""Algebras define the specific superposition and (un)binding operations."""

from .base import AbstractAlgebra, AbstractSign, CommonProperties, ElementSidedness
from .hrr_algebra import HrrAlgebra, HrrProperties, HrrSign
from .tvtb_algebra import TvtbAlgebra, TvtbProperties
from .vtb_algebra import VtbAlgebra, VtbProperties
