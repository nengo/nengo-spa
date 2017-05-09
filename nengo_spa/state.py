import warnings

import nengo
from nengo.exceptions import ValidationError
from nengo.networks.ensemblearray import EnsembleArray
from nengo.utils.network import with_self
from nengo.params import BoolParam, Default, IntParam, NumberParam
from nengo_spa.network import Network
from nengo_spa.vocab import VocabularyOrDimParam


class IdentityEnsembleArray(nengo.Network):
    """An ensemble array optimized for representing the identity vector for
    circular convolution.

    The ensemble array will use ensembles with *subdimensions* dimensions,
    except for the first *subdimensions* dimensions. These will be split into
    a one-dimensional ensemble for the first dimension and a *subdimensions-1*
    dimensional ensemble.

    Parameters
    ----------
    neurons_per_dimension : int
        Neurons per dimension.
    dimensions : int
        Total number of dimensions. Must be a multiple of *subdimensions*.
    subdimensions : int
        Maximum number of dimensions per ensemble.
    kwargs : dict
        Arguments to pass through to the `nengo.Network` constructor.

    Attributes
    ----------
    input : nengo.Node
        Input node.
    output : nengo.Node
        Output node.
    """
    def __init__(
            self, neurons_per_dimension, dimensions, subdimensions, **kwargs):
        super(IdentityEnsembleArray, self).__init__(**kwargs)

        self.neurons_per_dimension = neurons_per_dimension
        self.dimensions = dimensions
        self.subdimensions = subdimensions

        self.neuron_input = None

        cos_sim_dist = nengo.dists.CosineSimilarity(dimensions + 2)
        with self:
            self.input = nengo.Node(size_in=dimensions)
            self.output = nengo.Node(size_in=dimensions)

            first = nengo.Ensemble(neurons_per_dimension, 1)
            nengo.Connection(self.input[0], first, synapse=None)
            nengo.Connection(first, self.output[0], synapse=None)

            if subdimensions > 1:
                second = nengo.Ensemble(
                    neurons_per_dimension * (subdimensions - 1),
                    subdimensions - 1,
                    eval_points=cos_sim_dist, intercepts=cos_sim_dist)
                nengo.Connection(
                    self.input[1:subdimensions], second, synapse=None)
                nengo.Connection(
                    second, self.output[1:subdimensions], synapse=None)

            if dimensions > subdimensions:
                remainder = nengo.networks.EnsembleArray(
                    neurons_per_dimension * subdimensions,
                    dimensions // subdimensions - 1, subdimensions,
                    eval_points=cos_sim_dist, intercepts=cos_sim_dist)
                nengo.Connection(
                    self.input[subdimensions:], remainder.input, synapse=None)
                nengo.Connection(
                    remainder.output, self.output[subdimensions:],
                    synapse=None)

    @with_self
    def add_neuron_input(self):
        """Adds a node that provides input to the neurons of all ensembles.

        This node is accessible through the *neuron_input* attribute.

        Returns
        -------
        nengo.Node
            The added node.
        """
        if self.neuron_input is not None:
            warnings.warn("neuron_input already exists. Returning.")
            return self.neuron_input

        if any(isinstance(e.neuron_type, nengo.Direct)
               for e in self.all_ensembles):
            raise ValidationError(
                "Ensembles use Direct neuron type. "
                "Cannot give neuron input to Direct neurons.",
                attr='all_ensembles[i].neuron_type', obj=self)

        self.neuron_input = nengo.Node(
            size_in=self.neurons_per_dimension * self.dimensions,
            label="neuron_input")

        i = 0
        for ens in self.all_ensembles:
            nengo.Connection(
                self.neuron_input[i:(i + ens.n_neurons)], ens.neurons,
                synapse=None)
            i += ens.n_neurons

        return self.neuron_input


class State(Network):
    """A SPA network capable of representing a single vector, with optional
    memory.

    This is a minimal SPA network, useful for passing data along (for example,
    visual input).

    Parameters
    ----------
    vocab : Vocabulary or int
        The vocabulary to use to interpret the vector. If an integer is given,
        the default vocabulary of that dimensionality will be used.
    subdimensions : int, optional (Default: 16)
        Size of the individual ensembles making up the vector.
        Must divide ``dimensions`` evenly.
    neurons_per_dimensions : int, optional (Default: 50)
        Number of neurons in an ensemble will be
        ``neurons_per_dimensions * subdimensions``.
    feedback : float, optional (Default: 0.0)
        Gain of feedback connection. Set to 1.0 for perfect memory,
        or 0.0 for no memory. Values in between will create a decaying memory.
    feedback_synapse : float, optional (Default: 0.1)
        The synapse on the feedback connection.
    kwargs
        Keyword arguments passed through to ``spa.Network``.
    """

    vocab = VocabularyOrDimParam('vocab', default=None, readonly=True)
    subdimensions = IntParam('subdimensions', default=16, low=1, readonly=True)
    neurons_per_dimension = IntParam(
        'neurons_per_dimension', default=50, low=1, readonly=True)
    feedback = NumberParam('feedback', default=.0, readonly=True)
    feedback_synapse = NumberParam(
        'feedback_synapse', default=.1, readonly=True)
    represent_identity = BoolParam(
        'represent_identity', default=True, readonly=True)

    def __init__(self, vocab=Default, subdimensions=Default,
                 neurons_per_dimension=Default, feedback=Default,
                 represent_identity=Default,
                 feedback_synapse=Default, **kwargs):
        super(State, self).__init__(**kwargs)

        self.vocab = vocab
        self.subdimensions = subdimensions
        self.neurons_per_dimension = neurons_per_dimension
        self.feedback = feedback
        self.feedback_synapse = feedback_synapse
        self.represent_identity = represent_identity

        dimensions = self.vocab.dimensions

        if dimensions % self.subdimensions != 0:
            raise ValidationError(
                "Dimensions (%d) must be divisible by subdimensions (%d)" % (
                    dimensions, self.subdimensions),
                attr='dimensions', obj=self)

        with self:
            if self.represent_identity:
                self.state_ensembles = IdentityEnsembleArray(
                    self.neurons_per_dimension, dimensions, self.subdimensions,
                    label='state')
            else:
                self.state_ensembles = EnsembleArray(
                    self.neurons_per_dimension * self.subdimensions,
                    dimensions // self.subdimensions,
                    ens_dimensions=self.subdimensions,
                    eval_points=nengo.dists.CosineSimilarity(dimensions + 2),
                    intercepts=nengo.dists.CosineSimilarity(dimensions + 2),
                    label='state')

            if self.feedback is not None and self.feedback != 0.0:
                nengo.Connection(
                    self.state_ensembles.output, self.state_ensembles.input,
                    transform=self.feedback, synapse=self.feedback_synapse)

        self.input = self.state_ensembles.input
        self.output = self.state_ensembles.output
        self.inputs = dict(default=(self.input, self.vocab))
        self.outputs = dict(default=(self.output, self.vocab))
