import nengo
import numpy as np
from nengo.params import Default, IntParam

from nengo_spa.network import Network
from nengo_spa.vocabulary import VocabularyOrDimParam


class Compare(Network):
    """Computes the dot product of two inputs.

    Parameters
    ----------
    vocab : Vocabulary or int
        The vocabulary to use to interpret the vector. If an integer is given,
        the default vocabulary of that dimensionality will be used.
    neurons_per_dimension : int, optional (Default: 200)
        Number of neurons to use in each product computation.
    **kwargs : dict
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

    vocab = VocabularyOrDimParam("vocab", default=None, readonly=True)
    neurons_per_dimension = IntParam(
        "neurons_per_dimension", default=200, low=1, readonly=True
    )

    def __init__(self, vocab=Default, neurons_per_dimension=Default, **kwargs):
        super(Compare, self).__init__(**kwargs)

        self.vocab = vocab
        self.neurons_per_dimension = neurons_per_dimension

        with self:
            with nengo.Config(nengo.Ensemble) as cfg:
                cfg[nengo.Ensemble].eval_points = nengo.dists.CosineSimilarity(
                    self.vocab.dimensions + 2
                )
                cfg[nengo.Ensemble].intercepts = nengo.dists.CosineSimilarity(
                    self.vocab.dimensions + 2
                )
                self.product = nengo.networks.Product(
                    self.neurons_per_dimension, self.vocab.dimensions
                )
            self.output = nengo.Node(size_in=1, label="output")
            nengo.Connection(
                self.product.output,
                self.output,
                transform=np.ones((1, self.vocab.dimensions)),
            )

        self.input_a = self.product.input_a
        self.input_b = self.product.input_b

        self.declare_input(self.input_a, self.vocab)
        self.declare_input(self.input_b, self.vocab)
        self.declare_output(self.output, None)
