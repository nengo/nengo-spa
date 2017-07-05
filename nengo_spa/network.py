import weakref

import nengo
from nengo.config import Config, SupportDefaultsMixin
import numpy as np

from nengo_spa.exceptions import SpaNameError
from nengo_spa.vocab import VocabularyMap, VocabularyMapParam


class _AutoConfig(object):
    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, name):
        return getattr(self._cfg, name)

    def __getitem__(self, key):
        if key not in self._cfg.params:
            self._cfg.configures(key)
        return self._cfg[key]


class Network(nengo.Network, SupportDefaultsMixin):
    """Base class for SPA networks.

    SPA networks are networks that also have a list of inputs and outputs,
    each with an associated `.Vocabulary` (or a desired dimensionality for
    the vocabulary).

    The inputs and outputs are dictionaries that map a name to an
    (object, Vocabulary) pair. The object can be a `.Node` or an `.Ensemble`.
    """

    vocabs = VocabularyMapParam('vocabs', default=None, optional=False)

    _input_vocabs = weakref.WeakKeyDictionary()
    _input_networks = weakref.WeakKeyDictionary()
    _output_vocabs = weakref.WeakKeyDictionary()

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

    def declare_input(self, obj, vocab):
        if obj in self._input_vocabs:
            raise ValueError('Object {} already declared as input.'.format(
                obj))
        self._input_vocabs[obj] = vocab
        self._input_networks[obj] = self

    def declare_output(self, obj, vocab):
        if obj in self._output_vocabs:
            raise ValueError('Object {} already declared as output.'.format(
                obj))
        self._output_vocabs[obj] = vocab

    @classmethod
    def get_input_vocab(cls, obj):
        return cls._input_vocabs[obj]

    @classmethod
    def get_input_network(cls, obj):
        return cls._input_networks[obj]

    @classmethod
    def get_output_vocab(cls, obj):
        return cls._output_vocabs[obj]

    def get_spa_network(self, name, strip_output=False):
        """Return the SPA network for the given name.

        Raises :class:`SpaConstructionError` if the network cannot be found.

        Parameters
        ----------
        name : str
            Name of the network to retrieve.
        strip_output : bool, optional
            If ``True``, the network name is allowed to be followed by the name
            of an input or output that will be stripped (so the network with
            that input or output will be returned).

        Returns
        -------
        :class:`Network`
            Requested network.
        """
        components = name.split('.', 1)
        if len(components) > 1:
            head, tail = components
            try:
                return getattr(self, head).get_spa_network(
                    tail, strip_output=strip_output)
            except AttributeError:
                raise SpaNameError(head, 'network')
            except SpaNameError as err:
                raise SpaNameError(head + '.' + err.name, err.kind)
        else:
            if hasattr(self, name):
                return getattr(self, name)
            elif strip_output and (
                    name in self.inputs or name in self.outputs):
                return self
            else:
                raise SpaNameError(name, 'network')

    def get_network_input(self, name):
        """Return the object to connect into for the given name.

        The name will be either the same as a spa network, or of the form
        <network_name>.<input_name>.
        """
        obj = self.get_spa_network(name)
        try:
            return obj, self.get_input_vocab(obj)
        except KeyError:
            try:
                obj = getattr(obj, 'input')
                return obj, self.get_input_vocab(obj)
            except (AttributeError, KeyError):
                raise SpaNameError(name, 'network input')

    def get_network_output(self, name):
        """Return the object to connect into for the given name.

        The name will be either the same as a spa network, or of the form
        <network_name>.<output_name>.
        """
        obj = self.get_spa_network(name)
        try:
            return obj, self.get_output_vocab(obj)
        except KeyError:
            try:
                obj = getattr(obj, 'output')
                return obj, self.get_output_vocab(obj)
            except (AttributeError, KeyError):
                raise SpaNameError(name, 'network output')


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
    inhibit_node = nengo.Node(size_in=1, size_out=1)
    for e in net.all_ensembles:
        nengo.Connection(
            inhibit_node, e.neurons,
            transform=-strength * np.ones((e.n_neurons, 1), **kwargs))
    return inhibit_node
