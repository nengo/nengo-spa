"""Selection networks that pick one or more options among multiple choices."""
import nengo
import numpy as np


def IA(
        n_neurons, n_ensembles, accum_threshold=0.8, accum_neuron_ratio=0.7,
        accum_timescale=0.2, feedback_timescale=0.005,
        accum_synapse=0.1, ff_synapse=0.005,
        intercept_width=0.15, radius=1., **kwargs):
    """Independent accumulator (IA) winner-take-all (WTA) network.

    This is a two-layered network. The first layer consists of independent
    accumulators (integrators), whereas the second layer does a thresholding.
    Once the threshold is exceeded a feedback connection will stabilize the
    current choice and inhibit all other choices. To switch the selection,
    it is necessary to provide a transient input to *input_reset* to reset
    the accumulator states.

    This network is suited especially for accumulating evidence under noisy
    conditions and keep a stable choice selection until the processing of the
    choice has been finished.

    Further details are to be found in [gosmann2017]_.

    Parameters
    ----------
    n_neurons : int
        Number of neurons for each choice.
    n_ensembles : int
        Number of choices.
    accum_threshold : float, optional
        Accumulation threshold that needs to be reached to produce an output.
    accum_neuron_ratio: float, optional
        Portion of *n_neurons* that will be used for a layer 1 accumulator
        ensemble. The remaining neurons will be used for a layer 2 thresholding
        ensemble.
    accum_timescale : float, optional
        Evidence accumulation timescale.
    feedback_timescale : float, optional
        Timescale for the feedback connection from the thresholding layer to
        the accumulation layer.
    accum_synapse : Synapse or float, optional
        The synapse for connections to the accumulator ensembles.
    ff_synapse : Synapse or float, optional
        Synapse for feed-forward connections.
    intercept_width : float, optional
        The `nengo.presets.ThresholdingEnsembles` *intercept_width* parameter.
    radius : float, optional
        The representational radius of the ensembles.
    kwargs : dict
        Passed on to `nengo.Network`.

    Returns
    -------
    nengo.Network
        Network with attributes:

        * **input** (`nengo.Node`): The inputs to the network.
        * **input_reset** (`nengo.Node`): Input to reset the accumulators.
        * **output** (`nengo.Node`): The outputs of the network.
        * **accumulators** (`nengo.Thresholding`): The layer 1 accumulators.
        * **thresholding** (`nengo.Thresholding`): The layer 2 thresholding
          ensembles.

    References
    ----------
    .. [gosmann2017] Jan Gosmann, Aaron R. Voelker, and Chris Eliasmith. "A
       spiking independent accumulator model for winner-take-all computation."
       In Proceedings of the 39th Annual Conference of the Cognitive Science
       Society. London, UK, 2017. Cognitive Science Society.
    """
    n_accum_neurons = int(accum_neuron_ratio * n_neurons)
    n_thresholding_neurons = n_neurons - n_accum_neurons

    bar_beta = 1. + radius * feedback_timescale / accum_timescale
    feedback_tr = (
        np.eye(n_ensembles) - bar_beta * (1. - np.eye(n_ensembles)) /
        feedback_timescale)

    with nengo.Network(**kwargs) as net:
        net.accumulators = Thresholding(
            n_accum_neurons, n_ensembles, threshold=0.,
            intercept_width=intercept_width, radius=radius)
        net.thresholding = Thresholding(
            n_thresholding_neurons, n_ensembles, threshold=accum_threshold,
            intercept_width=intercept_width, radius=radius,
            function=lambda x: x > accum_threshold)

        nengo.Connection(
            net.accumulators.output, net.accumulators.input,
            synapse=accum_synapse)
        nengo.Connection(
            net.accumulators.output, net.thresholding.input,
            synapse=ff_synapse)
        nengo.Connection(
            net.thresholding.output, net.accumulators.input,
            synapse=accum_synapse, transform=accum_synapse * feedback_tr)

        net.input_reset = nengo.Node(size_in=1)
        nengo.Connection(
            net.input_reset, net.accumulators.input, synapse=None,
            transform=-radius * np.ones((n_ensembles, 1)) / accum_synapse)

        net.input = nengo.Node(size_in=n_ensembles)
        nengo.Connection(net.input, net.accumulators.input, synapse=None,
                         transform=1. / accum_timescale)
        net.output = net.thresholding.output
    return net


def Thresholding(
        n_neurons, n_ensembles, threshold, intercept_width=0.15, function=None,
        radius=1., **kwargs):
    """Array of thresholding ensembles.

    All inputs below the threshold will produce an output of 0, whereas inputs
    above the threshold produce an output of equal value.

    Parameters
    ----------
    n_neurons : int
        Number of neurons for each ensemble.
    n_ensembles : int
        Number of ensembles.
    threshold : float
        The thresholding value.
    intercept_width : float, optional
        The `nengo.presets.ThresholdingEnsembles` *intercept_width* parameter.
    function : function, optional
        Function to apply to the thresholded values.
    radius : float, optional
        The representational radius of the ensembles.
    kwargs : dict
        Arguments passed on to `nengo.Network`.

    Returns
    -------
    nengo.Network
        Network with attributes:

        * **input** (`nengo.Node`): The inputs to the network.
        * **output** (`nengo.Node`): The outputs of the network.
        * **thresholded** (`nengo.Node`): The raw thresholded value (before
          applying *function* or correcting for the shift produced by the
          thresholding).
    """
    with nengo.Network(**kwargs) as net:
        with nengo.presets.ThresholdingEnsembles(
                0., intercept_width, radius=radius):
            net.thresholding = nengo.networks.EnsembleArray(
                n_neurons, n_ensembles)

        net.bias = nengo.Node(1.)
        nengo.Connection(net.bias, net.thresholding.input,
                         transform=-threshold * np.ones((n_ensembles, 1)))

        net.input = net.thresholding.input
        net.thresholded = net.thresholding.output

        if function is None:
            function = lambda x: x
        function = lambda x, function=function: function(x + threshold)
        net.output = net.thresholding.add_output('function', function)
    return net


def WTA(n_neurons, n_ensembles, inhibit_scale=1.0, inhibit_synapse=0.005,
        **kwargs):
    """Winner-take-all (WTA) network with lateral inhibition.

    Parameters
    ----------
    n_neurons : int
        Number of neurons for each ensemble.
    n_ensembles : int
        Number of ensembles.
    inhibit_scale : float, optional
        Scaling of the lateral inhibition.
    inhibit_synapse : Synapse or float, optional
        Synapse on the recurrent connection for lateral inhibition.
    kwargs : dict
        Arguments passed on to `Thresholding`.

    Returns
    -------
    nengo.Network
        Network with attributes:

        * **input** (`nengo.Node`): The inputs to the network.
        * **output** (`nengo.Node`): The outputs of the network.
        * **thresholded** (`nengo.Node`): The raw thresholded value (before
          applying *function* or correcting for the shift produced by the
          thresholding).
    """
    net = Thresholding(n_neurons, n_ensembles, **kwargs)
    with net:
        nengo.Connection(
            net.thresholded, net.input,
            transform=inhibit_scale * (np.eye(n_ensembles) - 1.),
            synapse=inhibit_synapse)
    return net
