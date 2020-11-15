import nengo
import numpy as np
import pytest

import nengo_spa as spa


def test_basic(Simulator, plt, seed):
    with nengo.Network(seed=seed) as model:
        bg = spa.BasalGanglia(action_count=5, seed=seed)
        input = nengo.Node([0.8, 0.4, 0.4, 0.4, 0.4], label="input")
        nengo.Connection(input, bg.input, synapse=None)
        p = nengo.Probe(bg.output, synapse=0.01)

    with Simulator(model) as sim:
        sim.run(0.2)

    t = sim.trange()
    output = np.mean(sim.data[p][t > 0.1], axis=0)

    plt.plot(t, sim.data[p])
    plt.ylabel("Output")

    assert output[0] > -0.1
    assert np.all(output[1:] < -0.8)


@pytest.mark.slow
def test_basal_ganglia(Simulator, seed, plt):
    d = 64
    with spa.Network(seed=seed) as m:
        m.vision = spa.State(vocab=d)
        m.motor = spa.State(vocab=d)
        m.compare = spa.Compare(vocab=d)

        def input(t):
            if t < 0.1:
                return "0"
            elif t < 0.2:
                return "CAT"
            elif t < 0.3:
                return "DOG*~CAT"
            elif t < 0.4:
                return "PARROT"
            elif t < 0.5:
                return "MOUSE"
            else:
                return "0"

        m.encode = spa.Transcode(input, output_vocab=d)

        # test all acceptable condition formats
        with spa.ActionSelection() as actions:
            spa.ifmax(0.5, spa.sym.A >> m.motor)
            spa.ifmax(spa.dot(m.vision, spa.sym.CAT), spa.sym.B >> m.motor)
            spa.ifmax(
                spa.dot(m.vision * spa.sym.CAT, spa.sym.DOG), spa.sym.C >> m.motor
            )
            spa.ifmax(2 * spa.dot(m.vision, spa.sym.CAT * 0.5), spa.sym.D >> m.motor)
            spa.ifmax(
                spa.dot(m.vision, spa.sym.CAT) + 0.5 - spa.dot(m.vision, spa.sym.CAT),
                spa.sym.E >> m.motor,
            )
            spa.ifmax(
                spa.dot(m.vision, spa.sym.PARROT) + m.compare, spa.sym.F >> m.motor
            )
            spa.ifmax(
                0.5 * spa.dot(m.vision, spa.sym.MOUSE) + 0.5 * m.compare,
                spa.sym.G >> m.motor,
            )
            spa.ifmax(
                (spa.dot(m.vision, spa.sym.MOUSE) - m.compare) * 0.5,
                spa.sym.H >> m.motor,
            )

        m.encode >> m.vision
        spa.sym.SHOOP >> m.compare.input_a
        spa.sym.SHOOP >> m.compare.input_b
        bg = actions.bg

        p = nengo.Probe(bg.input, "output", synapse=0.03)

    with Simulator(m) as sim:
        sim.run(0.5)
    t = sim.trange()

    plt.plot(t, sim.data[p])
    plt.legend(["A", "B", "C", "D", "E", "F", "G", "H"])
    plt.title("Basal Ganglia output")

    # assert the basal ganglia is prioritizing things correctly
    # Motor F
    assert sim.data[p][t == 0.4, 5] > 0.8
    # Motor G
    assert sim.data[p][t == 0.5, 6] > 0.8
    # Motor A
    assert 0.6 > sim.data[p][t == 0.1, 0] > 0.4
    # Motor B
    assert sim.data[p][t == 0.2, 1] > 0.8
    # Motor C
    assert sim.data[p][t == 0.3, 2] > 0.5

    # Motor B should be the same as Motor D
    assert np.allclose(sim.data[p][:, 1], sim.data[p][:, 3], atol=0.2)
    # Motor A should be the same as Motor E
    assert np.allclose(sim.data[p][:, 0], sim.data[p][:, 4], atol=0.2)


def test_scalar_product():
    with spa.Network() as model:
        model.scalar = spa.Scalar()
        with spa.ActionSelection():
            spa.ifmax(model.scalar * model.scalar, 1 >> model.scalar)
    # just testing network construction without exception here


def test_constructed_input_connections_are_accessible():
    with spa.Network() as model:
        model.config[spa.State].vocab = 16
        model.state1 = spa.State()
        model.state2 = spa.State()

        with spa.ActionSelection() as actions:
            spa.ifmax(spa.dot(model.state1, spa.sym.A), spa.sym.A >> model.state2)
        bg = actions.bg

        assert isinstance(bg.input_connections[0], nengo.Connection)
