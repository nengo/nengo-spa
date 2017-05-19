import numpy as np

import nengo
from nengo.config import Config, SupportDefaultsMixin
from nengo_spa.exceptions import SpaNetworkError
from nengo_spa.modules.input import Input
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

        self._spa_networks = {}

        self.inputs = {}
        self.outputs = {}

        self._stimuli = None

        self._initialized = True

    @property
    def config(self):
        return _AutoConfig(self._config)

    # FIXME remove?
    @property
    def stimuli(self):
        if self._stimuli is None:
            self._stimuli = Input(self)
        return self._stimuli

    def __setstate__(self, state):
        if '_initialized' in state:
            del state['_initialized']
        super(Network, self).__setstate__(state)
        setattr(self, '_initialized', True)

    def __setattr__(self, key, value):
        """A setattr that handles SPA networks being added specially.

        This is so that we can use the variable name for the Network as
        the name that all of the SPA system will use to access that network.
        """
        if not hasattr(self, '_initialized'):
            return super(Network, self).__setattr__(key, value)

        if isinstance(value, Network):
            if hasattr(self, key) and isinstance(getattr(self, key), Network):
                raise SpaNetworkError(
                    "Cannot re-assign network-attribute %s to %s. SPA "
                    "network-attributes can only be assigned once."
                    % (key, value))

        super(Network, self).__setattr__(key, value)

        if isinstance(value, Network):
            if value.label is None:
                value.label = key
            self._spa_networks[key] = value

    def get_spa_network(self, name, strip_output=False):
        """Return the SPA network for the given name.

        Raises :class:`SpaNetworkError` if the network cannot be found.

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
        try:
            components = name.split('.', 1)
            if len(components) > 1:
                head, tail = components
                return self._spa_networks[head].get_spa_network(
                    tail, strip_output=strip_output)
            else:
                if name in self._spa_networks:
                    return self._spa_networks[name]
                elif strip_output and (
                        name in self.inputs or name in self.outputs):
                    return self
                else:
                    raise KeyError
        except KeyError:
            raise SpaNetworkError("Could not find network %r." % name)

    def get_network_input(self, name):
        """Return the object to connect into for the given name.

        The name will be either the same as a spa network, or of the form
        <network_name>.<input_name>.
        """
        try:
            components = name.split('.', 1)
            if len(components) > 1:
                head, tail = components
                return self._spa_networks[head].get_network_input(tail)
            else:
                if name in self.inputs:
                    return self.inputs[name]
                elif name in self._spa_networks:
                    return self._spa_networks[name].get_network_input(
                        'default')
                else:
                    raise KeyError
        except KeyError:
            raise SpaNetworkError("Could not find network input %r." % name)

    def get_network_output(self, name):
        """Return the object to connect into for the given name.

        The name will be either the same as a spa network, or of the form
        <network_name>.<output_name>.
        """
        try:
            components = name.split('.', 1)
            if len(components) > 1:
                head, tail = components
                return self._spa_networks[head].get_network_output(tail)
            else:
                if name in self.outputs:
                    return self.outputs[name]
                elif name in self._spa_networks:
                    return self._spa_networks[name].get_network_output(
                        'default')
                else:
                    raise KeyError
        except KeyError:
            raise SpaNetworkError("Could not find network output %r." % name)
