"""Associative memory implementations.

See :doc:`examples/associative_memory` for an introduction and examples.
"""
import nengo
from nengo.exceptions import ValidationError
from nengo.utils.compat import is_string
from nengo.utils.network import with_self
import numpy as np

from nengo_spa.network import Network
from nengo_spa.networks.selection import IA, Thresholding, WTA
from nengo_spa.vocab import VocabularyOrDimParam


class AssociativeMemory(Network):
    """General associative memory network.

    This provides a low-level selection network with the necessary interface
    to include it within the SPA system.

    Parameters
    ----------
    selection_net : Network
        The network that is used to select the response. It needs to accept
        the arguments *n_neurons* (number of neurons to use to represent each
        possible choice) and *n_ensembles* (number of choices). The returned
        network needs to have an *input* attribute to which the utilities for
        each choice are connected and an *output* attribute from which a
        connection will be created to read the selected output(s).
    input_vocab: Vocabulary
        The vocabulary to match.
    output_vocab: Vocabulary, optional
        The vocabulary to be produced for each match. If
        None, the associative memory will act like an autoassociative memory
        (cleanup memory).
    mapping: dict or str, optional
        A dictionary that defines the mapping from Semantic Pointers in the
        input vocabulary to Semantic Pointers in the output vocabulary. If set
        to the string ``'by-key'``, the mapping will be done based on the keys
        of the to vocabularies. If *None*, the associative memory will be
        autoassociative (cleanup memory).
    n_neurons : int
        Number of neurons to represent each choice, passed on to the
        *selection_net*.
    label : str, optional
        A name for the ensemble. Used for debugging and visualization.
    seed : int, optional
        The seed used for random number generation.
    add_to_container : bool, optional
        Determines if this Network will be added to the current container.
        See `nengo.Network` for more details.
    vocabs : VocabularyMap, optional
        Maps dimensionalities to the corresponding default vocabularies.
    """
    input_vocab = VocabularyOrDimParam(
        'input_vocab', default=None, readonly=True)
    output_vocab = VocabularyOrDimParam(
        'output_vocab', default=None, readonly=True)

    def __init__(
            self, selection_net, input_vocab, output_vocab=None, mapping=None,
            n_neurons=50, label=None, seed=None, add_to_container=None,
            vocabs=None, **selection_net_args):
        super(AssociativeMemory, self).__init__(
            label=label, seed=seed, add_to_container=add_to_container,
            vocabs=vocabs)

        if output_vocab is None:
            output_vocab = input_vocab
        elif mapping is None:
            raise ValidationError(
                "The mapping argument needs to be provided if an output "
                "vocabulary is given.", attr='mapping', obj=self)
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

        if mapping is None or mapping == 'by-key':
            mapping = {k: k for k in self.input_vocab.keys()}
        elif is_string(mapping):
            raise ValidationError(
                "The mapping argument must be a dictionary, the string "
                "'by-key' or None.", attr='mapping', obj=self)

        input_keys = mapping.keys()
        input_vectors = [input_vocab.parse(key).v for key in input_keys]
        output_keys = [mapping[k] for k in input_keys]
        output_vectors = [output_vocab.parse(key).v for key in output_keys]

        input_vectors = np.asarray(input_vectors)
        output_vectors = np.asarray(output_vectors)

        with self:
            self.selection = selection_net(
                n_neurons=n_neurons, n_ensembles=len(input_vectors),
                label="selection", **selection_net_args)
            self.input = nengo.Node(size_in=self.input_vocab.dimensions,
                                    label="input")
            self.output = nengo.Node(size_in=self.output_vocab.dimensions,
                                     label="output")

            nengo.Connection(
                self.input, self.selection.input, transform=input_vectors)
            nengo.Connection(
                self.selection.output, self.output, transform=output_vectors.T)

        self.declare_input(self.input, self.input_vocab)
        self.declare_output(self.output, self.output_vocab)

    @with_self
    def add_default_output(self, key, min_activation_value, n_neurons=50):
        """Adds a Semantic Pointer to output when no other pointer is active.

        Parameters
        ----------
        key : str
            Semantic Pointer to output.
        min_activation_value : float
            Minimum output of another Semantic Pointer to deactivate the
            default output.
        n_neurons : int, optional
            Number of neurons used to represent the default Semantic Pointer.
        """
        assert not hasattr(self, 'default_ens'), \
            "Can add default output only once."

        with nengo.presets.ThresholdingEnsembles(0.):
            setattr(self, 'default_ens',
                    nengo.Ensemble(n_neurons, 1, label="default"))
        setattr(self, 'bias', nengo.Node(1., label="bias"))
        nengo.Connection(self.bias, self.default_ens)
        nengo.Connection(
            self.default_ens, self.output,
            transform=np.atleast_2d(self.output_vocab.parse(key).v).T)
        nengo.Connection(
            self.selection.output, self.default_ens,
            transform=-np.ones(
                (1, self.selection.output.size_out)) / min_activation_value)


class IAAssocMem(AssociativeMemory):
    """Associative memory based on the `.IA` network.

    See `AssociativeMemory` and `.IA` for more information.
    """
    def __init__(
            self, input_vocab, output_vocab=None, mapping=None,
            n_neurons=50, label=None, seed=None, add_to_container=None,
            vocabs=None, **selection_net_args):
        super(IAAssocMem, self).__init__(
            selection_net=IA,
            input_vocab=input_vocab, output_vocab=output_vocab,
            mapping=mapping, n_neurons=n_neurons, label=label, seed=seed,
            add_to_container=add_to_container, vocabs=vocabs,
            **selection_net_args)
        self.input_reset = self.selection.input_reset
        self.declare_input(self.input_reset, None)


class ThresholdingAssocMem(AssociativeMemory):
    """Associative memory based on `.Thresholding`.

    See `AssociativeMemory` and `.Thresholding` for more information.
    """
    def __init__(
            self, threshold, input_vocab, output_vocab=None, mapping=None,
            n_neurons=50, label=None, seed=None, add_to_container=None,
            vocabs=None, **selection_net_args):
        selection_net_args['threshold'] = threshold
        super(ThresholdingAssocMem, self).__init__(
            selection_net=Thresholding,
            input_vocab=input_vocab, output_vocab=output_vocab,
            mapping=mapping, n_neurons=n_neurons, label=label, seed=seed,
            add_to_container=add_to_container, vocabs=vocabs,
            **selection_net_args)


class WTAAssocMem(AssociativeMemory):
    """Associative memory based on the `.WTA` network.

    See `AssociativeMemory` and `.WTA` for more information.
    """
    def __init__(
            self, threshold, input_vocab, output_vocab=None, mapping=None,
            n_neurons=50, label=None, seed=None, add_to_container=None,
            vocabs=None, **selection_net_args):
        selection_net_args['threshold'] = threshold
        super(WTAAssocMem, self).__init__(
            selection_net=WTA,
            input_vocab=input_vocab, output_vocab=output_vocab,
            mapping=mapping, n_neurons=n_neurons, label=label, seed=seed,
            add_to_container=add_to_container, vocabs=vocabs,
            **selection_net_args)
