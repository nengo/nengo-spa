import nengo
from nengo.params import Default, IntParam
from nengo_spa.network import Network


class Product(Network):
    """A SPA network capable of multiplying two scalars.

    Parameters
    ----------
    n_neurons : int, optional (Default: 200)
        Number of neurons to use in product computation.
    kwargs
        Keyword arguments passed through to ``spa.Network``.
    """

    n_neurons = IntParam('n_neurons', default=200, low=1, readonly=True)

    def __init__(self, n_neurons=Default, **kwargs):
        super(Product, self).__init__(**kwargs)

        self.n_neurons = n_neurons

        with self:
            self.product = nengo.networks.Product(self.n_neurons, 1)

        self.input_a = self.product.input_a
        self.input_b = self.product.input_b
        self.output = self.product.output

        self.declare_input(self.input_a, None)
        self.declare_input(self.input_b, None)
        self.declare_output(self.output, None)
