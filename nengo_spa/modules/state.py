import nengo
from nengo.exceptions import ValidationError
from nengo.networks.ensemblearray import EnsembleArray
from nengo.params import BoolParam, Default, IntParam, NumberParam

from nengo_spa.network import Network
from nengo_spa.networks import IdentityEnsembleArray
from nengo_spa.vocabulary import VocabularyOrDimParam


class State(Network):
    """Represents a single vector, with optional memory.

    This is a minimal SPA network, useful for passing data along (for example,
    visual input).

    Parameters
    ----------
    vocab : Vocabulary or int
        The vocabulary to use to interpret the vector. If an integer is given,
        the default vocabulary of that dimensionality will be used.
    subdimensions : int, optional (Default: 16)
        The dimension of the individual ensembles making up the vector.
        Must divide *dimensions* evenly. The number of sub-ensembles
        will be ``dimensions // subdimensions``.
    neurons_per_dimension : int, optional (Default: 50)
        Number of neurons per dimension. Each ensemble will have
        ``neurons_per_dimension * subdimensions`` neurons, for a total of
        ``neurons_per_dimension * dimensions`` neurons.
    feedback : float, optional (Default: 0.0)
        Gain of feedback connection. Set to 1.0 for perfect memory,
        or 0.0 for no memory. Values in between will create a decaying memory.
    represent_cc_identity : bool, optional
        Whether to use optimizations to better represent the circular
        convolution identity vector. If activated, the `.IdentityEnsembleArray`
        will be used internally, otherwise a normal
        `nengo.networks.EnsembleArray` split up regularly according to
        *subdimensions*.
    feedback_synapse : float, optional (Default: 0.1)
        The synapse on the feedback connection.
    **kwargs : dict
        Keyword arguments passed through to `nengo_spa.Network`.

    Attributes
    ----------
    input : nengo.Node
        Input.
    output : nengo.Node
        Output.
    """

    vocab = VocabularyOrDimParam("vocab", default=None, readonly=True)
    subdimensions = IntParam("subdimensions", default=16, low=1, readonly=True)
    neurons_per_dimension = IntParam(
        "neurons_per_dimension", default=50, low=1, readonly=True
    )
    feedback = NumberParam("feedback", default=0.0, readonly=True)
    feedback_synapse = NumberParam("feedback_synapse", default=0.1, readonly=True)
    represent_cc_identity = BoolParam(
        "represent_cc_identity", default=True, readonly=True
    )

    def __init__(
        self,
        vocab=Default,
        subdimensions=Default,
        neurons_per_dimension=Default,
        feedback=Default,
        represent_cc_identity=Default,
        feedback_synapse=Default,
        **kwargs
    ):
        super(State, self).__init__(**kwargs)

        self.vocab = vocab
        self.subdimensions = subdimensions
        self.neurons_per_dimension = neurons_per_dimension
        self.feedback = feedback
        self.feedback_synapse = feedback_synapse
        self.represent_cc_identity = represent_cc_identity

        dimensions = self.vocab.dimensions

        if dimensions % self.subdimensions != 0:
            raise ValidationError(
                "Dimensions (%d) must be divisible by subdimensions (%d)"
                % (dimensions, self.subdimensions),
                attr="dimensions",
                obj=self,
            )

        with self:
            if self.represent_cc_identity:
                self.state_ensembles = IdentityEnsembleArray(
                    self.neurons_per_dimension,
                    dimensions,
                    self.subdimensions,
                    label="state",
                )
            else:
                self.state_ensembles = EnsembleArray(
                    self.neurons_per_dimension * self.subdimensions,
                    dimensions // self.subdimensions,
                    ens_dimensions=self.subdimensions,
                    eval_points=nengo.dists.CosineSimilarity(dimensions + 2),
                    intercepts=nengo.dists.CosineSimilarity(dimensions + 2),
                    label="state",
                )

            if self.feedback is not None and self.feedback != 0.0:
                nengo.Connection(
                    self.state_ensembles.output,
                    self.state_ensembles.input,
                    transform=self.feedback,
                    synapse=self.feedback_synapse,
                )

        self.input = self.state_ensembles.input
        self.output = self.state_ensembles.output
        self.declare_input(self.input, self.vocab)
        self.declare_output(self.output, self.vocab)
