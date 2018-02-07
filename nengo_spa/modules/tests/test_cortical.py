import numpy as np
import pytest

import nengo
import nengo_spa as spa


def test_connect(Simulator, seed):
    with spa.Network(seed=seed) as model:
        model.buffer1 = spa.State(vocab=16)
        model.buffer2 = spa.State(vocab=16)
        model.buffer3 = spa.State(vocab=16)

        spa.sym.A >> model.buffer1
        model.buffer1 >> model.buffer2
        ~model.buffer1 >> model.buffer3

    with model:
        p2 = nengo.Probe(model.buffer2.output, synapse=0.03)
        p3 = nengo.Probe(model.buffer3.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.2)

    match = np.dot(sim.data[p2], model.buffer2.vocab.parse('A').v)
    assert match[199] > 0.8
    match = np.dot(sim.data[p3], model.buffer3.vocab.parse('~A').v)
    assert match[199] > 0.8


def test_transform(Simulator, seed):
    with spa.Network(seed=seed) as model:
        model.buffer1 = spa.State(vocab=32)
        model.buffer2 = spa.State(vocab=32)

        spa.sym.A >> model.buffer1
        model.buffer1 * spa.sym.B >> model.buffer2

    with model:
        p = nengo.Probe(model.buffer2.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.2)

    match = np.dot(sim.data[p], model.buffer2.vocab.parse('A*B').v)
    assert match[199] > 0.7


# FIXME test different populate arguments
def test_translate(Simulator, seed):
    with spa.Network(seed=seed) as model:
        model.buffer1 = spa.State(vocab=16)
        model.buffer2 = spa.State(vocab=32)

        spa.sym.A >> model.buffer1
        spa.translate(
            model.buffer1, model.buffer2.vocab, populate=True) >> model.buffer2

    with model:
        p = nengo.Probe(model.buffer2.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.2)

    match = np.dot(sim.data[p], model.buffer2.vocab.parse('A').v)
    assert match[199] > 0.8


def test_errors():
    # buffer2 does not exist
    with pytest.raises(AttributeError):
        with spa.Network() as model:
            model.buffer1 = spa.State(vocab=16)
            model.buffer1 >> model.buffer2


def test_direct(Simulator, seed):
    with spa.Network(seed=seed) as model:
        model.buffer1 = spa.State(vocab=16)
        model.buffer1.vocab.populate('A; B; C')
        model.buffer2 = spa.State(vocab=32)
        model.buffer2.vocab.populate('A; B; C')

        spa.sym.A >> model.buffer1
        spa.sym.B >> model.buffer2
        spa.sym.C >> model.buffer1
        spa.sym.C >> model.buffer2

    with model:
        p1 = nengo.Probe(model.buffer1.output, synapse=0.03)
        p2 = nengo.Probe(model.buffer2.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.2)

    match1 = np.dot(sim.data[p1], model.buffer1.vocab.parse('A+C').v)
    match2 = np.dot(sim.data[p2], model.buffer2.vocab.parse('B+C').v)
    # both values should be near 1.0 since buffer1 is driven to both A and C
    # and buffer2 is driven to both B and C.
    assert match1[199] > 0.75
    assert match2[199] > 0.75


def test_convolution(Simulator, plt, seed):
    D = 5
    with spa.Network(seed=seed) as model:
        model.config[spa.State].vocab = D
        model.config[spa.State].subdimensions = D
        model.inA = spa.State()
        model.inB = spa.State()
        model.outAB = spa.State()
        model.outABinv = spa.State()
        model.outAinvB = spa.State()
        model.outAinvBinv = spa.State()

        model.inA * model.inB >> model.outAB
        model.inA * ~model.inB >> model.outABinv
        ~model.inA * model.inB >> model.outAinvB
        ~model.inA * ~model.inB >> model.outAinvBinv
        nengo.Connection(nengo.Node([0, 1, 0, 0, 0]), model.inA.input)
        nengo.Connection(nengo.Node([0, 0, 1, 0, 0]), model.inB.input)

        pAB = nengo.Probe(model.outAB.output, synapse=0.03)
        pABinv = nengo.Probe(model.outABinv.output, synapse=0.03)
        pAinvB = nengo.Probe(model.outAinvB.output, synapse=0.03)
        pAinvBinv = nengo.Probe(model.outAinvBinv.output, synapse=0.03)

    for state in [
            model.inA,
            model.inB,
            model.outAB,
            model.outABinv,
            model.outAinvB,
            model.outAinvBinv]:
        for e in state.all_ensembles:
            e.radius = 1.

    with Simulator(model) as sim:
        sim.run(0.2)

    t = sim.trange()
    plt.subplot(4, 1, 1)
    plt.ylabel('A*B')
    plt.axhline(0.85, c='k')
    plt.plot(t, sim.data[pAB])
    plt.subplot(4, 1, 2)
    plt.ylabel('A*~B')
    plt.axhline(0.85, c='k')
    plt.plot(t, sim.data[pABinv])
    plt.subplot(4, 1, 3)
    plt.ylabel('~A*B')
    plt.axhline(0.85, c='k')
    plt.plot(t, sim.data[pAinvB])
    plt.subplot(4, 1, 4)
    plt.ylabel('~A*~B')
    plt.axhline(0.85, c='k')
    plt.plot(t, sim.data[pAinvBinv])

    # Check results.  Since A is [0,1,0,0,0] and B is [0,0,1,0,0], this means:
    #    ~A = [0,0,0,0,1]
    #    ~B = [0,0,0,1,0]
    #   A*B = [0,0,0,1,0]
    #  A*~B = [0,0,0,0,1]
    #  ~A*B = [0,1,0,0,0]
    # ~A*~B = [0,0,1,0,0]
    # (Remember that X*[1,0,0,0,0]=X (identity transform) and X*[0,1,0,0,0]
    #  is X rotated to the right once)

    # Ideal answer: A*B = [0,0,0,1,0]
    assert np.allclose(np.mean(sim.data[pAB][-10:], axis=0),
                       np.array([0, 0, 0, 1, 0]), atol=0.25)

    # Ideal answer: A*~B = [0,0,0,0,1]
    assert np.allclose(np.mean(sim.data[pABinv][-10:], axis=0),
                       np.array([0, 0, 0, 0, 1]), atol=0.25)

    # Ideal answer: ~A*B = [0,1,0,0,0]
    assert np.allclose(np.mean(sim.data[pAinvB][-10:], axis=0),
                       np.array([0, 1, 0, 0, 0]), atol=0.25)

    # Ideal answer: ~A*~B = [0,0,1,0,0]
    assert np.allclose(np.mean(sim.data[pAinvBinv][-10:], axis=0),
                       np.array([0, 0, 1, 0, 0]), atol=0.25)
