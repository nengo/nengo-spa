import nengo
import numpy as np
import pytest

import nengo_spa as spa


def test_thalamus_basic(Simulator, plt, seed):
    with nengo.Network(seed=seed) as net:
        bg = spa.BasalGanglia(action_count=5)
        input = nengo.Node([0.8, 0.4, 0.4, 0.4, 0.4], label="input")
        nengo.Connection(input, bg.input, synapse=None)

        thal = spa.Thalamus(action_count=5)
        nengo.Connection(bg.output, thal.input)

        p = nengo.Probe(thal.output, synapse=0.01)

    with Simulator(net) as sim:
        sim.run(0.2)

    t = sim.trange()
    output = np.mean(sim.data[p][t > 0.1], axis=0)

    plt.plot(t, sim.data[p])
    plt.ylabel("Output")

    assert output[0] > 0.8
    assert np.all(output[1:] < 0.01)


@pytest.mark.slow
def test_thalamus(Simulator, plt, seed):
    with spa.Network(seed=seed) as m:
        m.vision = spa.State(vocab=16, neurons_per_dimension=80)
        m.motor = spa.State(vocab=16, neurons_per_dimension=80)

        with spa.ActionSelection():
            spa.ifmax(spa.dot(m.vision, spa.sym.A), spa.sym.A >> m.motor)
            spa.ifmax(spa.dot(m.vision, spa.sym.B), m.vision >> m.motor)
            spa.ifmax(spa.dot(m.vision, ~spa.sym.A), ~m.vision >> m.motor)

        def input_f(t):
            if t < 0.1:
                return "A"
            elif t < 0.3:
                return "B"
            elif t < 0.5:
                return "~A"
            else:
                return "0"

        m.input = spa.Transcode(input_f, output_vocab=16)
        m.input >> m.vision

        p = nengo.Probe(m.motor.output, synapse=0.03)

    with Simulator(m) as sim:
        sim.run(0.5)

    t = sim.trange()
    data = m.motor.vocab.dot(sim.data[p].T)

    plt.plot(t, data.T)

    # Action 1
    assert data[0, t == 0.1] > 0.8
    assert data[1, t == 0.1] < 0.2
    # Action 2
    assert data[0, t == 0.3] < 0.2
    assert data[1, t == 0.3] > 0.8
    # Action 3
    assert data[0, t == 0.5] > 0.8
    assert data[1, t == 0.5] < 0.2


def test_routing(Simulator, seed, plt):
    model = spa.Network(seed=seed)
    model.config[spa.State].vocab = 3
    model.config[spa.State].subdimensions = 3
    with model:
        model.ctrl = spa.State(16, subdimensions=16, label="ctrl")

        def input_func(t):
            if t < 0.2:
                return "A"
            elif t < 0.4:
                return "B"
            else:
                return "C"

        model.input = spa.Transcode(input_func, output_vocab=16)

        model.buff1 = spa.State(label="buff1")
        model.buff2 = spa.State(label="buff2")
        model.buff3 = spa.State(label="buff3")

        node1 = nengo.Node([0, 1, 0])
        node2 = nengo.Node([0, 0, 1])

        nengo.Connection(node1, model.buff1.input)
        nengo.Connection(node2, model.buff2.input)

        model.input >> model.ctrl
        with spa.ActionSelection():
            spa.ifmax(spa.dot(model.ctrl, spa.sym.A), model.buff1 >> model.buff3)
            spa.ifmax(spa.dot(model.ctrl, spa.sym.B), model.buff2 >> model.buff3)
            spa.ifmax(
                spa.dot(model.ctrl, spa.sym.C), model.buff1 * model.buff2 >> model.buff3
            )

        buff3_probe = nengo.Probe(model.buff3.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.6)

    data = sim.data[buff3_probe]

    plt.plot(sim.trange(), data)

    valueA = np.mean(data[150:200], axis=0)  # should be [0, 1, 0]
    valueB = np.mean(data[350:400], axis=0)  # should be [0, 0, 1]
    valueC = np.mean(data[550:600], axis=0)  # should be [1, 0, 0]

    assert valueA[0] < 0.2
    assert valueA[1] > 0.7
    assert valueA[2] < 0.2

    assert valueB[0] < 0.2
    assert valueB[1] < 0.2
    assert valueB[2] > 0.7

    assert valueC[0] > 0.7
    assert valueC[1] < 0.2
    assert valueC[2] < 0.2


def test_routing_recurrency_compilation(Simulator, seed):
    model = spa.Network(seed=seed)
    model.config[spa.State].vocab = 2
    model.config[spa.State].subdimensions = 2
    with model:
        model.buff1 = spa.State(label="buff1")
        model.buff2 = spa.State(label="buff2")
        with spa.ActionSelection():
            spa.ifmax(0.5, model.buff1 >> model.buff2, model.buff2 >> model.buff1)

    with Simulator(model) as sim:
        assert sim


def test_nondefault_routing(Simulator, seed):
    m = spa.Network(seed=seed)
    m.config[spa.State].vocab = 3
    m.config[spa.State].subdimensions = 3
    with m:
        m.ctrl = spa.State(16, subdimensions=16, label="ctrl")

        def input_func(t):
            if t < 0.2:
                return "A"
            elif t < 0.4:
                return "B"
            else:
                return "C"

        m.input = spa.Transcode(input_func, output_vocab=16)

        m.buff1 = spa.State(label="buff1")
        m.buff2 = spa.State(label="buff2")
        m.cmp = spa.Compare(3)

        node1 = nengo.Node([0, 1, 0])
        node2 = nengo.Node([0, 0, 1])

        nengo.Connection(node1, m.buff1.input)
        nengo.Connection(node2, m.buff2.input)

        m.input >> m.ctrl
        with spa.ActionSelection():
            spa.ifmax(
                spa.dot(m.ctrl, spa.sym.A),
                m.buff1 >> m.cmp.input_a,
                m.buff1 >> m.cmp.input_b,
            )
            spa.ifmax(
                spa.dot(m.ctrl, spa.sym.B),
                m.buff1 >> m.cmp.input_a,
                m.buff2 >> m.cmp.input_b,
            )
            spa.ifmax(
                spa.dot(m.ctrl, spa.sym.C),
                m.buff2 >> m.cmp.input_a,
                m.buff2 >> m.cmp.input_b,
            )

        compare_probe = nengo.Probe(m.cmp.output, synapse=0.03)

    with Simulator(m) as sim:
        sim.run(0.6)

    similarity = sim.data[compare_probe]

    valueA = np.mean(similarity[150:200], axis=0)  # should be [1]
    valueB = np.mean(similarity[350:400], axis=0)  # should be [0]
    valueC = np.mean(similarity[550:600], axis=0)  # should be [1]

    assert valueA > 0.6
    assert valueB < 0.3
    assert valueC > 0.6


def test_errors():
    # motor does not exist
    with pytest.raises(AttributeError):
        with spa.Network() as model:
            model.vision = spa.State(vocab=16)
            with spa.ActionSelection:
                spa.ifmax(0.5, spa.sym.A >> model.motor)


def test_constructed_objects_are_accessible():
    with spa.Network() as model:
        model.config[spa.State].vocab = 16
        model.state1 = spa.State()
        model.state2 = spa.State()
        model.state3 = spa.State()

        with spa.ActionSelection() as actions:
            spa.ifmax(spa.dot(model.state1, spa.sym.A), model.state3 >> model.state2)
            spa.ifmax(0.5, spa.sym.B >> model.state2)
        bg = actions.bg
        thalamus = actions.thalamus

        assert isinstance(thalamus.gates[0], nengo.Ensemble)
        assert isinstance(thalamus.gate_in_connections[0], nengo.Connection)
        assert isinstance(thalamus.gate_out_connections[0], nengo.Connection)
        assert isinstance(thalamus.channels[0], spa.State)
        assert isinstance(thalamus.channel_out_connections[0], nengo.Connection)

        assert isinstance(thalamus.fixed_connections[1], nengo.Connection)

        assert thalamus.bg_connection.pre is bg.output
        assert thalamus.bg_connection.post is thalamus.input
