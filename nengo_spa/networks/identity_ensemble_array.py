import warnings

import nengo
import numpy as np
from nengo.exceptions import ValidationError
from nengo.utils.network import with_self

from nengo_spa.typechecks import is_iterable


class IdentityEnsembleArray(nengo.Network):
    """An ensemble array optimized for representing the identity circular
    convolution vector besides random unit vectors.

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
    **kwargs : dict
        Keyword arguments to pass through to the `nengo.Network` constructor.

    Attributes
    ----------
    input : nengo.Node
        Input node.
    output : nengo.Node
        Output node.
    """

    def __init__(self, neurons_per_dimension, dimensions, subdimensions, **kwargs):
        super(IdentityEnsembleArray, self).__init__(**kwargs)

        self.neurons_per_dimension = neurons_per_dimension
        self.dimensions = dimensions
        self.subdimensions = subdimensions

        self.neuron_input = None
        self.neuron_output = None

        cos_sim_dist = nengo.dists.CosineSimilarity(dimensions + 2)
        with self:
            self.input = nengo.Node(size_in=dimensions)
            self.output = nengo.Node(size_in=dimensions)

            self.first = nengo.Ensemble(neurons_per_dimension, 1)
            nengo.Connection(self.input[0], self.first, synapse=None)
            nengo.Connection(self.first, self.output[0], synapse=None)

            if subdimensions > 1:
                self.second = nengo.Ensemble(
                    neurons_per_dimension * (subdimensions - 1),
                    subdimensions - 1,
                    eval_points=cos_sim_dist,
                    intercepts=cos_sim_dist,
                )
                nengo.Connection(self.input[1:subdimensions], self.second, synapse=None)
                nengo.Connection(
                    self.second, self.output[1:subdimensions], synapse=None
                )

            if dimensions > subdimensions:
                self.remainder = nengo.networks.EnsembleArray(
                    neurons_per_dimension * subdimensions,
                    dimensions // subdimensions - 1,
                    subdimensions,
                    eval_points=cos_sim_dist,
                    intercepts=cos_sim_dist,
                )
                nengo.Connection(
                    self.input[subdimensions:], self.remainder.input, synapse=None
                )
                nengo.Connection(
                    self.remainder.output, self.output[subdimensions:], synapse=None
                )

    @with_self
    def add_neuron_input(self):
        """Adds a node providing input to the neurons of all ensembles.

        This node will be accessible through the *neuron_input* attribute.

        Returns
        -------
        nengo.Node
            The added node.
        """
        if self.neuron_input is not None:
            warnings.warn("neuron_input already exists. Returning.")
            return self.neuron_input

        if any(isinstance(e.neuron_type, nengo.Direct) for e in self.all_ensembles):
            raise ValidationError(
                "Ensembles use Direct neuron type. "
                "Cannot give neuron input to Direct neurons.",
                attr="all_ensembles[i].neuron_type",
                obj=self,
            )

        self.neuron_input = nengo.Node(
            size_in=self.neurons_per_dimension * self.dimensions, label="neuron_input"
        )

        i = 0
        for ens in self.all_ensembles:
            nengo.Connection(
                self.neuron_input[i : (i + ens.n_neurons)], ens.neurons, synapse=None
            )
            i += ens.n_neurons

        return self.neuron_input

    @with_self
    def add_neuron_output(self):
        """Adds a node providing neuron (non-decoded) output of all ensembles.

        This node will be accessible through the *neuron_output* attribute.

        Returns
        -------
        nengo.Node
            The added node.
        """
        if self.neuron_output is not None:
            warnings.warn("neuron_output already exists. Returning.")
            return self.neuron_output

        if any(isinstance(e.neuron_type, nengo.Direct) for e in self.all_ensembles):
            raise ValidationError(
                "Ensembles use Direct neuron type. "
                "Cannot get neuron output from Direct neurons.",
                attr="all_ensembles[i].neuron_type",
                obj=self,
            )

        self.neuron_output = nengo.Node(
            size_in=self.neurons_per_dimension * self.dimensions, label="neuron_output"
        )

        i = 0
        for ens in self.all_ensembles:
            nengo.Connection(
                ens.neurons, self.neuron_output[i : (i + ens.n_neurons)], synapse=None
            )
            i += ens.n_neurons

        return self.neuron_output

    @with_self
    def add_output(self, name, function, synapse=None, **conn_kwargs):
        """Adds a new decoded output.

        This will add the attribute named *name* to the object.

        Parameters
        ----------
        name : str
            Name of output. Must be a valid Python attribute name.
        function : func
            Function to decode.
        synapse : float or nengo.Lowpass
            Synapse to apply to the decoded connection to the returned output
            node.
        conn_kwargs : dict
            Additional keywword arguments to apply to the decoded connection.

        Returns
        -------
        nengo.Node
            Node providing the decoded output.
        """
        if is_iterable(function):
            function = list(function)
            if len(function) != 3 and len(function) != self.remainder.n_ensembles + 2:
                raise ValidationError(
                    "Must provide one function per ensemble or one function "
                    "each for the first ensemble, the second ensembles, and "
                    "all remaining ensembles."
                )
            first_fn = function[0]
            second_fn = function[1]
            remainder_fn = function[2:]
        else:
            first_fn = second_fn = remainder_fn = function

        first_size = np.asarray(first_fn(np.zeros(self.first.dimensions))).size
        second_size = np.asarray(second_fn(np.zeros(self.second.dimensions))).size
        remainder_start = first_size + second_size

        remainder_fn_out = self.remainder.add_output(name, remainder_fn)
        remainder_size = remainder_fn_out.size_out

        output = nengo.Node(
            size_in=first_size + second_size + remainder_size, label=name
        )
        setattr(self, name, output)

        nengo.Connection(
            self.first,
            output[:first_size],
            function=first_fn,
            synapse=synapse,
            **conn_kwargs
        )
        nengo.Connection(
            self.second,
            output[first_size:remainder_start],
            function=second_fn,
            synapse=synapse,
            **conn_kwargs
        )
        nengo.Connection(
            remainder_fn_out, output[remainder_start:], synapse=synapse, **conn_kwargs
        )

        return output
