import nengo
from nengo.config import Config, SupportDefaultsMixin
import numpy as np

from nengo_spa.ast import (
    input_network_registry, input_vocab_registry, output_vocab_registry,
    SpaOperatorMixin)
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


class Network(nengo.Network, SupportDefaultsMixin, SpaOperatorMixin):
    """Base class for SPA networks.

    SPA networks are networks that also have a list of inputs and outputs,
    each with an associated `.Vocabulary` (or a desired dimensionality for
    the vocabulary).

    The inputs and outputs are dictionaries that map a name to an
    (object, Vocabulary) pair. The object can be a `.Node` or an `.Ensemble`.
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
        return input_vocab_registry[obj]

    @classmethod
    def get_output_vocab(cls, obj):
        return output_vocab_registry[obj]

    def declare_input(self, obj, vocab):
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
        Node that can be connected to to provide an inhibitory signal to the
        network.
    """
    inhibit_node = nengo.Node(size_in=1)
    for e in net.all_ensembles:
        nengo.Connection(
            inhibit_node, e.neurons,
            transform=-strength * np.ones((e.n_neurons, 1), **kwargs))
    return inhibit_node
