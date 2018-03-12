from nengo.params import BoolParam, Default, IntParam

import nengo_spa
from nengo_spa.network import Network
from nengo_spa.vocab import VocabularyOrDimParam


class Bind(Network):
    """Network for binding together two inputs.

    Binding is done with circular convolution. For more details on how
    this is computed, see the underlying `.CircularConvolution`
    network.

    Parameters
    ----------
    vocab : Vocabulary or int
        The vocabulary to use to interpret the vector. If an integer is given,
        the default vocabulary of that dimensionality will be used.
    neurons_per_dimension : int, optional (Default: 200)
        Number of neurons to use in each product computation.
    invert_a : bool, optional
        Whether to reverse the order of elements in first input.
    invert_b : bool, optional
        Whether to reverse the order of elements in the second input.
        Flipping exactly one input will make the network perform circular
        correlation instead of circular convolution which can be treated as an
        approximate inverse to circular convolution.
    kwargs : dict
        Keyword arguments passed through to `nengo_spa.Network`.

    Attributes
    ----------
    input_a : nengo.Node
        First input vector.
    input_b : nengo.Node
        Second input vector.
    output : nengo.Node
        Output.
    """

    vocab = VocabularyOrDimParam('vocab', default=None, readonly=True)
    neurons_per_dimension = IntParam(
        'neurons_per_dimension', default=200, low=1, readonly=True)
    invert_a = BoolParam('invert_a', default=False, readonly=True)
    invert_b = BoolParam('invert_b', default=False, readonly=True)

    def __init__(self, vocab=Default, neurons_per_dimension=Default,
                 invert_a=Default, invert_b=Default, **kwargs):
        super(Bind, self).__init__(**kwargs)

        self.vocab = vocab
        self.neurons_per_dimension = neurons_per_dimension
        self.invert_a = invert_a
        self.invert_b = invert_b

        with self:
            self.cc = nengo_spa.networks.CircularConvolution(
                self.neurons_per_dimension, self.vocab.dimensions,
                self.invert_a, self.invert_b)

        self.input_a = self.cc.input_a
        self.input_b = self.cc.input_b
        self.output = self.cc.output

        self.declare_input(self.input_a, self.vocab)
        self.declare_input(self.input_b, self.vocab)
        self.declare_output(self.output, self.vocab)
