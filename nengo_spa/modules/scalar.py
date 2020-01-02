import nengo
from nengo.params import Default, IntParam

from nengo_spa.network import Network


class Scalar(Network):
    """Represents a single scalar.

    Parameters
    ----------
    n_neurons : int, optional (Default: 50)
        Number of neurons to represent the scalar.
    **kwargs : dict
        Keyword arguments passed through to `nengo_spa.Network`.

    Attributes
    ----------
    input : nengo.Node
         Input.
    output : nengo.Node
        Output.
    """

    n_neurons = IntParam("n_neurons", default=50, low=1, readonly=True)

    def __init__(self, n_neurons=Default, **kwargs):
        super(Scalar, self).__init__(**kwargs)

        self.n_neurons = n_neurons

        with self:
            self.scalar = nengo.Ensemble(self.n_neurons, 1)

        self.input = self.scalar
        self.output = self.scalar
        self.declare_input(self.input, None)
        self.declare_output(self.output, None)
