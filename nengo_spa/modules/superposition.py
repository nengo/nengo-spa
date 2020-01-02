from nengo.params import Default, IntParam

from nengo_spa.network import Network
from nengo_spa.vocabulary import VocabularyOrDimParam


class Superposition(Network):
    """Network for superposing multiple inputs.

    Parameters
    ----------
    n_inputs : int
        Number of inputs.
    vocab : Vocabulary or int
        The vocabulary to use to interpret the vector. If an integer is given,
        the default vocabulary of that dimensionality will be used.
    neurons_per_dimension : int, optional (Default: 200)
        Number of neurons to use in each dimension.
    **kwargs : dict
        Keyword arguments passed through to `nengo_spa.Network`.

    Attributes
    ----------
    inputs : sequence
        Inputs.
    output : nengo.Node
        Output.
    """

    vocab = VocabularyOrDimParam("vocab", default=None, readonly=True)
    neurons_per_dimension = IntParam(
        "neurons_per_dimension", default=200, low=1, readonly=True
    )
    n_inputs = IntParam("n_inputs", optional=False, low=1, readonly=True)

    def __init__(
        self, n_inputs, vocab=Default, neurons_per_dimension=Default, **kwargs
    ):
        super(Superposition, self).__init__(**kwargs)

        self.vocab = vocab
        self.neurons_per_dimension = neurons_per_dimension
        self.n_inputs = n_inputs

        with self:
            (
                self.superposition_net,
                self.inputs,
                self.output,
            ) = self.vocab.algebra.implement_superposition(
                self.neurons_per_dimension, self.vocab.dimensions, self.n_inputs
            )

        for inp in self.inputs:
            self.declare_input(inp, self.vocab)
        self.declare_output(self.output, self.vocab)
