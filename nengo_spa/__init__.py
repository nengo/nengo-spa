from nengo_spa import version

from nengo_spa.actions import Actions
from nengo_spa.examine import pairs, similarity, text
from nengo_spa.modules import (
    AssociativeMemory, IAAssocMem, ThresholdingAssocMem, WTAAssocMem,
    BasalGanglia,
    Bind,
    Compare,
    Product,
    Scalar,
    State,
    Thalamus,
    Transcode)
from nengo_spa.network import create_inhibit_node, Network
from nengo_spa.vocab import Vocabulary, VocabularyMap


__copyright__ = "2013-2017, Applied Brain Research"
__license__ = "Free for non-commercial use; see LICENSE.rst"
__version__ = version.version
