import warnings

import nengo
from nengo.exceptions import ValidationError
from nengo.utils.network import with_self


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
