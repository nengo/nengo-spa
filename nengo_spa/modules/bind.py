from nengo.params import BoolParam, Default, IntParam

from nengo_spa.network import Network
from nengo_spa.vocabulary import VocabularyOrDimParam


class Bind(Network):
    """Network for binding together two inputs.

    Parameters
    ----------
    vocab : Vocabulary or int
        The vocabulary to use to interpret the vector. If an integer is given,
        the default vocabulary of that dimensionality will be used.
    neurons_per_dimension : int, optional (Default: 200)
        Number of neurons to use in each dimension.
    unbind_left : bool, optional
        Whether to unbind the left input.
    unbind_right : bool, optional
        Whether to unbind the right input.
    **kwargs : dict
        Keyword arguments passed through to `nengo_spa.Network`.

    Attributes
    ----------
    input_left : nengo.Node
        Left input vector.
    input_right : nengo.Node
        Right input vector.
    output : nengo.Node
        Output.
    """

    vocab = VocabularyOrDimParam("vocab", default=None, readonly=True)
    neurons_per_dimension = IntParam(
        "neurons_per_dimension", default=200, low=1, readonly=True
    )
    unbind_left = BoolParam("unbind_left", default=False, readonly=True)
    unbind_right = BoolParam("unbind_right", default=False, readonly=True)

    def __init__(
        self,
        vocab=Default,
        neurons_per_dimension=Default,
        unbind_left=Default,
        unbind_right=Default,
        **kwargs
    ):
        super(Bind, self).__init__(**kwargs)

        self.vocab = vocab
        self.neurons_per_dimension = neurons_per_dimension
        self.unbind_left = unbind_left
        self.unbind_right = unbind_right

        with self:
            self.binding_net, inputs, output = self.vocab.algebra.implement_binding(
                self.neurons_per_dimension,
                self.vocab.dimensions,
                self.unbind_left,
                self.unbind_right,
            )

        self.input_left = inputs[0]
        self.input_right = inputs[1]
        self.output = output

        self.declare_input(self.input_left, self.vocab)
        self.declare_input(self.input_right, self.vocab)
        self.declare_output(self.output, self.vocab)
