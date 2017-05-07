from .actions import Actions
from .assoc_mem import AssociativeMemory, ThresholdingAssocMem, WTAAssocMem
from .basalganglia import BasalGanglia
from .bind import Bind
from .compare import Compare
from .input import Input
from .network import Network
from .pointer import SemanticPointer
from .product import Product
from .scalar import Scalar
from .state import State
from .thalamus import Thalamus
from .utils import similarity
from .vocab import Vocabulary, VocabularyMap

from nengo_spa import networks
from nengo_spa import version

__copyright__ = "2013-2018, Applied Brain Research"
__license__ = "Free for non-commercial use; see LICENSE.rst"
__version__ = version.version
