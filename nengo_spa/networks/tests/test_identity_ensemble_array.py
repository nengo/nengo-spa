import nengo
import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from nengo_spa.networks.identity_ensemble_array import IdentityEnsembleArray
from nengo_spa.semantic_pointer import Identity, SemanticPointer
from nengo_spa.testing import assert_sp_close
from nengo_spa.vector_generation import UnitLengthVectors


@pytest.mark.parametrize("pointer", [Identity, UnitLengthVectors])
def test_identity_ensemble_array(Simulator, seed, rng, pointer):
    d = 64
    if issubclass(pointer, SemanticPointer):
        v = pointer(d).v
    else:
        v = next(pointer(d, rng=rng))

    with nengo.Network(seed=seed) as model:
        ea = IdentityEnsembleArray(15, d, 4)
        input_v = nengo.Node(v)
        nengo.Connection(input_v, ea.input)
        p = nengo.Probe(ea.output, synapse=0.01)

    with Simulator(model) as sim:
        sim.run(0.3)

    actual = np.mean(sim.data[p][sim.trange() > 0.2], axis=0)
    assert np.abs(v[0] - actual[0]) < 0.15
    assert np.linalg.norm(v - actual) < 0.2


def test_add_output(Simulator, seed, rng, plt):
    d = 8
    pointer = next(UnitLengthVectors(d, rng=rng))

    with nengo.Network(seed=seed) as model:
        ea = IdentityEnsembleArray(15, d, 4)
        input_node = nengo.Node(pointer)
        nengo.Connection(input_node, ea.input)
        out = ea.add_output("const", lambda x: -x)
        assert ea.const is out
        p = nengo.Probe(out, synapse=0.01)

    with Simulator(model) as sim:
        sim.run(0.3)

    plt.plot(sim.trange(), np.dot(sim.data[p], -pointer))
    assert_sp_close(
        sim.trange(), sim.data[p], SemanticPointer(-pointer), skip=0.2, atol=0.3
    )


def test_add_output_multiple_fn(Simulator, seed, rng, plt):
    d = 8
    pointer = next(UnitLengthVectors(d, rng=rng))

    with nengo.Network(seed=seed) as model:
        ea = IdentityEnsembleArray(15, d, 4)
        input_node = nengo.Node(pointer)
        nengo.Connection(input_node, ea.input)
        out = ea.add_output("const", (lambda x: -x, lambda x: 0.5 * x, lambda x: x))
        assert ea.const is out
        p = nengo.Probe(out, synapse=0.01)

    with Simulator(model) as sim:
        sim.run(0.3)

    expected = np.array(pointer)
    expected[0] *= -1.0
    expected[1:4] *= 0.5

    plt.plot(sim.trange(), np.dot(sim.data[p], expected))
    assert_sp_close(
        sim.trange(), sim.data[p], SemanticPointer(expected), skip=0.2, atol=0.3
    )


def test_neuron_connections(Simulator, seed, rng):
    d = 8
    pointer = next(UnitLengthVectors(d, rng=rng))

    with nengo.Network(seed=seed) as model:
        ea = IdentityEnsembleArray(15, d, 4)
        input_node = nengo.Node(pointer)
        nengo.Connection(input_node, ea.input)

        bias = nengo.Node(1)
        neuron_in = ea.add_neuron_input()
        assert ea.neuron_input is neuron_in
        nengo.Connection(
            bias, neuron_in, transform=-3.0 * np.ones((neuron_in.size_in, 1))
        )

        neuron_out = ea.add_neuron_output()
        assert ea.neuron_output is neuron_out
        p = nengo.Probe(neuron_out)

    with Simulator(model) as sim:
        sim.run(0.3)

    assert_almost_equal(sim.data[p][sim.trange() > 0.1], 0.0)
