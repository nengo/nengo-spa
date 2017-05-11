import nengo


def make_parse_func(func, vocab):
    """Create a function that calls func and parses the output in vocab."""

    def parse_func(t):
        return vocab.parse(func(t)).v

    return parse_func


class _HierachicalInputProxy(object):
    def __init__(self, parent, name):
        self.__dict__['parent'] = parent
        self.__dict__['name'] = name

    def __getattr__(self, name):
        return _HierachicalInputProxy(self.parent, self.name + '.' + name)

    def __setattr__(self, name, value):
        setattr(self.parent, self.name + '.' + name, value)


class Input(nengo.Network):
    """A SPA network for providing external inputs to other networks.

    The parameters passed to this network indicate the network input name
    and the function to execute to generate inputs to that network.
    The functions should always return strings, which will then be parsed
    by the relevant vocabulary. For example::

        def input1(t):
            if t < 0.1:
                return 'A'
            else:
                return '0'

        spa_net.input = spa.Input(vision=input1, task='X')

    will create two inputs:

    1. an input to the ``vision`` network, which for the first 0.1 seconds
       is the value associated with the ``'A'`` semantic pointer and then
       a vector of all zeros, and
    2. an input to the ``task`` network which is always the value associated
       with the ``'X'`` semantic pointer.

    Parameters
    ----------
    network : spa.Network, optional (Default: the current SPA network)
        Network that this instance provides input for.
    kwargs
        Keyword arguments passed through to ``nengo.Network``.
    """

    def __init__(self, network=None, **kwargs):
        super(Input, self).__init__(**kwargs)
        self.input_nodes = {}

        if network is None:
            from nengo_spa.network import get_current_spa_network
            network = get_current_spa_network()
        self.network = network

        self._initialized = True

    def __connect(self, name, expr):
        target, vocab = self.network.get_network_input(name)
        if callable(expr):
            val = make_parse_func(expr, vocab)
        else:
            val = vocab.parse(expr).v

        with self:
            node = nengo.Node(val, label='input_%s' % name)
        self.input_nodes[name] = node

        with self.network:
            nengo.Connection(node, target, synapse=None)

    def __setattr__(self, name, value):
        if not getattr(self, '_initialized') or name in self.__dict__:
            super(Input, self).__setattr__(name, value)
        else:
            self.__connect(name, value)

    def __getattr__(self, name):
        if name == '_initialized':
            return self.__dict__.get('_initialized', False)
        return _HierachicalInputProxy(self, name)
