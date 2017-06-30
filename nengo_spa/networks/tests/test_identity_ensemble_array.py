import nengo
import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from nengo_spa.networks.identity_ensemble_array import IdentityEnsembleArray
from nengo_spa.pointer import Identity, SemanticPointer
from nengo_spa.testing import sp_close


@pytest.mark.parametrize('pointer', [Identity, SemanticPointer])
def test_identity_ensemble_array(Simulator, seed, rng, pointer):
    d = 64
    try:
        v = pointer(d, rng=rng).v
    except TypeError:
        v = pointer(d).v

    with nengo.Network(seed=seed) as model:
        ea = IdentityEnsembleArray(15, d, 4)
        input_v = nengo.Node(v)
        nengo.Connection(input_v, ea.input)
        p = nengo.Probe(ea.output, synapse=0.01)

    with Simulator(model) as sim:
        sim.run(0.3)

    actual = np.mean(sim.data[p][sim.trange() > 0.2], axis=0)
    assert np.abs(v[0] - actual[0]) < 0.1
    assert np.linalg.norm(v - actual) < 0.2


def test_add_output(Simulator, seed, rng, plt):
    d = 8
    pointer = SemanticPointer(d, rng=rng)

    with nengo.Network(seed=seed) as model:
        ea = IdentityEnsembleArray(15, d, 4)
        input_node = nengo.Node(pointer.v)
        nengo.Connection(input_node, ea.input)
        out = ea.add_output('const', lambda x: -x)
        assert ea.const is out
        p = nengo.Probe(out, synapse=0.01)

    with Simulator(model) as sim:
        sim.run(0.3)

    plt.plot(sim.trange(), np.dot(sim.data[p], -pointer.v))
    assert sp_close(sim.trange(), sim.data[p], -pointer, skip=0.2, atol=0.3)


def test_add_output_multiple_fn(Simulator, seed, rng, plt):
    d = 8
    pointer = SemanticPointer(d, rng=rng)

    with nengo.Network(seed=seed) as model:
        ea = IdentityEnsembleArray(15, d, 4)
        input_node = nengo.Node(pointer.v)
        nengo.Connection(input_node, ea.input)
        out = ea.add_output(
            'const', (lambda x: -x, lambda x: .5 * x, lambda x: x))
        assert ea.const is out
        p = nengo.Probe(out, synapse=0.01)

    with Simulator(model) as sim:
        sim.run(0.3)

    v = np.array(pointer.v)
    v[0] *= -1.
    v[1:4] *= .5
    expected = SemanticPointer(v)

    plt.plot(sim.trange(), np.dot(sim.data[p], expected.v))
    assert sp_close(sim.trange(), sim.data[p], expected, skip=0.2, atol=0.3)


def test_neuron_connections(Simulator, seed, rng):
    d = 8
    pointer = SemanticPointer(d, rng=rng)

    with nengo.Network(seed=seed) as model:
        ea = IdentityEnsembleArray(15, d, 4)
        input_node = nengo.Node(pointer.v)
        nengo.Connection(input_node, ea.input)

        bias = nengo.Node(1)
        neuron_in = ea.add_neuron_input()
        assert ea.neuron_input is neuron_in
        nengo.Connection(
            bias, neuron_in, transform=-3. * np.ones((neuron_in.size_in, 1)))

        neuron_out = ea.add_neuron_output()
        assert ea.neuron_output is neuron_out
        p = nengo.Probe(neuron_out)

    with Simulator(model) as sim:
        sim.run(0.3)

    assert_almost_equal(sim.data[p][sim.trange() > 0.1], 0.)
