import inspect

import nengo
from nengo.config import Config, SupportDefaultsMixin
from nengo.utils.compat import is_number, is_string
import numpy as np

from nengo_spa.actions import ifmax as actions_ifmax
from nengo_spa.actions import ModuleInput
from nengo_spa.ast.base import Node
from nengo_spa.ast.dynamic import (
    input_network_registry, input_vocab_registry, ModuleOutput,
    output_vocab_registry)
from nengo_spa.ast.symbolic import FixedScalar
from nengo_spa.exceptions import SpaTypeError
from nengo_spa.types import TScalar, TVocabulary
from nengo_spa.vocab import VocabularyMap, VocabularyMapParam


class _AutoConfig(object):
    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, name):
        return getattr(self._cfg, name)

    def __getitem__(self, key):
        if inspect.isclass(key) and key not in self._cfg.params:
            self._cfg.configures(key)
        return self._cfg[key]


def as_ast_node(obj):
    if isinstance(obj, Node):
        return obj
    elif is_number(obj):
        return FixedScalar(obj)
    elif isinstance(obj, Network):
        output = obj.output
    else:
        output = obj

    try:
        # Trying to create weakref on access of weak dict can raise TypeError
        vocab = output_vocab_registry[output]
    except (KeyError, TypeError) as cause:
        err = SpaTypeError("{} was not registered as a SPA output.".format(
            output))
        err.__suppress_context__ = True
        raise err
    finally:
        err = None  # prevent cyclic reference, traceback might reference this

    if vocab is None:
        return ModuleOutput(output, TScalar)
    else:
        return ModuleOutput(output, TVocabulary(vocab))


def as_sink(obj):
    if isinstance(obj, Network):
        input_ = obj.input
    else:
        input_ = obj

    try:
        # Trying to create weakref on access of weak dict can raise TypeError
        vocab = input_vocab_registry[input_]
    except (KeyError, TypeError) as cause:
        err = SpaTypeError("{} was not registered as a SPA input.".format(
            input_))
        err.__suppress_context__ = True
        raise err
    finally:
        err = None  # prevent cyclic reference, traceback might reference this

    if vocab is None:
        return ModuleInput(input_, TScalar)
    else:
        return ModuleInput(input_, TVocabulary(vocab))


class SpaOperatorMixin(object):
    """Mixin class that implements the SPA operators.

    All operands will be converted to AST node and the implementation of the
    operator itself is delegated to the implementation provided by those nodes.
    """

    @staticmethod
    def __define_unary_op(op):
        def op_impl(self):
            return getattr(as_ast_node(self), op)()
        return op_impl

    @staticmethod
    def __define_binary_op(op):
        def op_impl(self, other):
            return getattr(as_ast_node(self), op)(as_ast_node(other))
        return op_impl

    __invert__ = __define_unary_op.__func__('__invert__')
    __neg__ = __define_unary_op.__func__('__neg__')

    __add__ = __define_binary_op.__func__('__add__')
    __radd__ = __define_binary_op.__func__('__radd__')
    __sub__ = __define_binary_op.__func__('__sub__')
    __rsub__ = __define_binary_op.__func__('__rsub__')
    __mul__ = __define_binary_op.__func__('__mul__')
    __rmul__ = __define_binary_op.__func__('__rmul__')
    __matmul__ = __define_binary_op.__func__('__matmul__')
    __rmatmul__ = __define_binary_op.__func__('__rmatmul__')

    def __rshift__(self, other):
        return as_ast_node(self) >> as_sink(other)

    def __rrshift__(self, other):
        return as_ast_node(other) >> as_sink(self)

    dot = __define_binary_op.__func__('dot')
    rdot = __define_binary_op.__func__('rdot')

    def reinterpret(self, vocab=None):
        return as_ast_node(self).reinterpret(vocab)

    def translate(self, vocab, populate=None, keys=None, solver=None):
        return as_ast_node(self).translate(vocab, populate, keys, solver)


def ifmax(name, condition, *actions):
    """Defines a potential action within an `ActionSelection` context.

    This implementation allows Nengo objects in addition to AST nodes as
    condition argument.

    Parameters
    ----------
    name : str, optional
        Name for the action. Can be omitted.
    condition : nengo_spa.ast.base.Node or NengoObject
        The utility value for the given actions.
    actions : sequence of `RoutedConnection`
        The actions to activate if the given utility is the highest.

    Returns
    -------
    NengoObject
        Nengo object that can be connected to, to provide additional input to
        the utility value.
    """
    if not is_string(name):
        actions = (condition,) + actions
        condition = name
        name = None

    return actions_ifmax(name, as_ast_node(condition), *actions)


class Network(nengo.Network, SupportDefaultsMixin, SpaOperatorMixin):
    """Base class for SPA networks or modules.

    SPA modules are networks that declare their inputs and outputs with
    associated `.Vocabulary` instances. These inputs and outputs can then be
    be used in the SPA syntax, for example ``module1.output >> module2.input``.
    Inputs and outputs named `default` can be omitted in the SPA syntax so that
    one can write ``module1 >> module2``.

    Furthermore, SPA modules allow to configure parameters of contained SPA
    modules, for example::

        with spa.Network() as net:
            net.config[spa.State].vocab = 32
            state = spa.State()  # Will now have a 32-dimensional vocabulary

    Parameters
    ----------
    label : str, optional
        Name of the network.
    seed : int, optional
        Random number seed for the network.
    add_to_container : bool, optional
        Determines if this network will be added to the current container.
    vocabs : VocabularyMap, optional
        Maps from integer dimensionalities to the associated default
        vocabularies.

    Attributes
    ----------
    vocabs : VocabularyMap
        Maps from integer dimensionalities to the associated default
        vocabularies.
    """

    vocabs = VocabularyMapParam('vocabs', default=None, optional=False)

    _input_types = {}
    _output_types = {}

    def __init__(
            self, label=None, seed=None, add_to_container=None, vocabs=None):
        super(Network, self).__init__(label, seed, add_to_container)
        self.config.configures(Network)

        if vocabs is None:
            vocabs = Config.default(Network, 'vocabs')
            if vocabs is None:
                if seed is not None:
                    rng = np.random.RandomState(seed)
                else:
                    rng = None
                vocabs = VocabularyMap(rng=rng)
        self.vocabs = vocabs
        self.config[Network].vocabs = vocabs

        self._stimuli = None

    @property
    def config(self):
        return _AutoConfig(self._config)

    @classmethod
    def get_input_vocab(cls, obj):
        """Get the vocabulary associated with an network input *obj*."""
        return input_vocab_registry[obj]

    @classmethod
    def get_output_vocab(cls, obj):
        """Get the vocabulary associated with an network output *obj*."""
        return output_vocab_registry[obj]

    def declare_input(self, obj, vocab):
        """Declares a network input.

        Parameters
        ----------
        obj : nengo.base.NengoObject
            Nengo object to use as an input to the network.
        vocab: Vocabulary
            Vocabulary to assign to the input.
        """
        try:
            extended_type = self._input_types[obj.__class__]
        except KeyError:
            extended_type = type(
                'SpaInput<%s>' % obj.__class__.__name__,
                (obj.__class__, SpaOperatorMixin), {})
            self._input_types[obj.__class__] = extended_type
        obj.__class__ = extended_type
        input_vocab_registry[obj] = vocab
        input_network_registry[obj] = self

    def declare_output(self, obj, vocab):
        """ Declares a network output.

        Parameters
        ----------
        obj : nengo.base.NengoObject
            Nengo object to use as an output of the network.
        vocab : Vocabulary
            Vocabulary to assign to the output.
        """
        try:
            extended_type = self._output_types[obj.__class__]
        except KeyError:
            extended_type = type(
                'SpaOutput<%s>' % obj.__class__.__name__,
                (obj.__class__, SpaOperatorMixin), {})
            self._output_types[obj.__class__] = extended_type
        obj.__class__ = extended_type
        output_vocab_registry[obj] = vocab
        return obj


def create_inhibit_node(net, strength=2., **kwargs):
    """Creates a node that inhibits all ensembles in a network.

    Parameters
    ----------
    net : nengo.Network
        Network to inhibit.
    strength : float
        Strength of the inhibition.
    kwargs : dict
        Additional keyword arguments for the created connections from the node
        to the inhibited ensemble neurons.

    Returns
    -------
    nengo.Node
        Node that can be connected to, to provide an inhibitory signal to the
        network.
    """
    inhibit_node = nengo.Node(size_in=1)
    for e in net.all_ensembles:
        nengo.Connection(
            inhibit_node, e.neurons,
            transform=-strength * np.ones((e.n_neurons, 1), **kwargs))
    return inhibit_node
