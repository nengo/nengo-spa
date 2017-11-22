import nengo
import numpy as np


def create_inhibit_node(net, strength=2., **kwargs):
    """Creates a node that inhibits all ensembles in a network.

    Parameters
    ----------
    net : nengo.Network
        Network to inhibit.
    strength : float
        Strength of the inhibition.
    kwargs : dict
        Additional keyword arguments for the created connections from the node
        to the inhibited ensemble neurons.

    Returns
    -------
    nengo.Node
        Node that can be connected to to provide an inhibitory signal to the
        network.
    """
    inhibit_node = nengo.Node(size_in=1)
    for e in net.all_ensembles:
        nengo.Connection(
            inhibit_node, e.neurons,
            transform=-strength * np.ones((e.n_neurons, 1), **kwargs))
    return inhibit_node
