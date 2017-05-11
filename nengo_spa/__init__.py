from nengo_spa.actions import Actions
from nengo_spa.modules.assoc_mem import AssociativeMemory, ThresholdingAssocMem, WTAAssocMem
from nengo_spa.modules.basalganglia import BasalGanglia
from nengo_spa.modules.bind import Bind
from nengo_spa.modules.compare import Compare
from nengo_spa.modules.input import Input
from nengo_spa.network import Network
from nengo_spa.pointer import SemanticPointer
from nengo_spa.modules.product import Product
from nengo_spa.modules.scalar import Scalar
from nengo_spa.modules.state import State
from nengo_spa.modules.thalamus import Thalamus
from nengo_spa.examine import similarity
from nengo_spa.vocab import Vocabulary, VocabularyMap

from nengo_spa import networks
from nengo_spa import version

__copyright__ = "2013-2018, Applied Brain Research"
__license__ = "Free for non-commercial use; see LICENSE.rst"
__version__ = version.version
