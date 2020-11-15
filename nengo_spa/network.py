import inspect
import weakref

import nengo
import numpy as np
from nengo.config import Config, SupportDefaultsMixin

from nengo_spa.action_selection import ifmax as actions_ifmax
from nengo_spa.ast.base import Noop
from nengo_spa.connectors import (
    SpaOperatorMixin,
    as_ast_node,
    input_vocab_registry,
    output_vocab_registry,
)
from nengo_spa.types import TScalar
from nengo_spa.vocabulary import VocabularyMap, VocabularyMapParam


class _AutoConfig:
    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, name):
        return getattr(self._cfg, name)

    def __getitem__(self, key):
        if inspect.isclass(key) and key not in self._cfg.params:
            self._cfg.configures(key)
        return self._cfg[key]


def ifmax(name, condition=None, *actions):
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
        the utility value. It is possible (but not necessary) to use SPA style
        connections of the form ``scalar >> utility`` to this object.
    """
    if not isinstance(name, str):
        if condition is not None:
            actions = (condition,) + actions
        condition = name
        name = None

    if condition is None:
        raise ValueError("Must provide `condition` (though it may be 0).")
    elif condition == 0:
        condition = Noop(TScalar)
    else:
        condition = as_ast_node(condition)

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

    _master_vocabs = weakref.WeakKeyDictionary()
    vocabs = VocabularyMapParam("vocabs", default=None, optional=False)

    _input_types = {}
    _output_types = {}

    def __init__(self, label=None, seed=None, add_to_container=None, vocabs=None):
        super(Network, self).__init__(label, seed, add_to_container)
        self.config.configures(Network)

        if vocabs is None:
            vocabs = Config.default(Network, "vocabs")
            if vocabs is None and len(Network.context) > 0:
                vocabs = self._master_vocabs.get(Network.context[0], None)
            if vocabs is None:
                if seed is not None:
                    rng = np.random.RandomState(seed)
                else:
                    rng = None
                vocabs = VocabularyMap(rng=rng)
                if len(Network.context) > 0:
                    self.__class__._master_vocabs[Network.context[0]] = vocabs
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
        return input_vocab_registry.declare_connector(obj, vocab)

    def declare_output(self, obj, vocab):
        """Declares a network output.

        Parameters
        ----------
        obj : nengo.base.NengoObject
            Nengo object to use as an output of the network.
        vocab : Vocabulary
            Vocabulary to assign to the output.
        """
        return output_vocab_registry.declare_connector(obj, vocab)


def create_inhibit_node(net, strength=2.0, **kwargs):
    """Creates a node that inhibits all ensembles in a network.

    Parameters
    ----------
    net : nengo.Network
        Network to inhibit.
    strength : float
        Strength of the inhibition.
    **kwargs : dict
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
            inhibit_node,
            e.neurons,
            transform=-strength * np.ones((e.n_neurons, 1), **kwargs),
        )
    return inhibit_node
