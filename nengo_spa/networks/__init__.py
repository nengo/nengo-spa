"""Basic networks that are used by NengoSPA.

These networks do not provide any information about inputs, outputs or used
vocabularies and are completely independent of SPA specifics.
"""

from . import selection
from .circularconvolution import CircularConvolution
from .identity_ensemble_array import IdentityEnsembleArray
from .matrix_multiplication import MatrixMult
from .tvtb import TVTB
from .vtb import VTB
