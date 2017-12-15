import numpy as np
import pytest

import nengo
import nengo_spa as spa
from nengo_spa.exceptions import SpaTypeError


def test_new_action_syntax(Simulator, seed, plt, rng):
    model = spa.Network(seed=seed)
    model.config[spa.State].vocab = 3
    model.config[spa.State].subdimensions = 3
    with model:
        model.ctrl = spa.State(16, subdimensions=16, label='ctrl')

        def input_func(t):
            if t < 0.2:
                return 'A'
            elif t < 0.4:
                return 'B'
            else:
                return 'C'

        model.input = spa.Transcode(input_func, output_vocab=16)

        model.state = spa.State(label='state')
        model.buff1 = spa.State(label='buff1')
        model.buff2 = spa.State(label='buff2')
        model.buff3 = spa.State(label='buff3')

        node1 = nengo.Node([0, 1, 0])
        node2 = nengo.Node([0, 0, 1])

        nengo.Connection(node1, model.buff1.input)
        nengo.Connection(node2, model.buff2.input)

        model.input >> model.ctrl
        model.buff1 >> model.state
        with spa.ActionSelection():
            spa.ifmax(
                spa.dot(model.ctrl, spa.sym.A), model.buff1 >> model.buff3)
            spa.ifmax(
                spa.dot(model.ctrl, spa.sym.B), model.buff2 >> model.buff3)
            spa.ifmax(
                spa.dot(model.ctrl, spa.sym.C),
                model.buff1 * model.buff2 >> model.buff3)

        state_probe = nengo.Probe(model.state.output, synapse=0.03)
        buff3_probe = nengo.Probe(model.buff3.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.6)

    data = sim.data[buff3_probe]

    plt.plot(sim.trange(), data)

    state_val = np.mean(sim.data[state_probe][200:], axis=0)
    assert state_val[0] < 0.2
    assert state_val[1] > 0.8
    assert state_val[2] < 0.2

    valueA = np.mean(data[150:200], axis=0)  # should be [0, 1, 0]
    valueB = np.mean(data[350:400], axis=0)  # should be [0, 0, 1]
    valueC = np.mean(data[550:600], axis=0)  # should be [1, 0, 0]

    assert valueA[0] < 0.2
    assert valueA[1] > 0.75
    assert valueA[2] < 0.2

    assert valueB[0] < 0.2
    assert valueB[1] < 0.2
    assert valueB[2] > 0.75

    assert valueC[0] > 0.75
    assert valueC[1] < 0.2
    assert valueC[2] < 0.2


def test_dot_product(Simulator, seed, plt):
    d = 16

    with spa.Network(seed=seed) as model:
        model.state_a = spa.State(d)
        model.state_b = spa.State(d)
        model.result = spa.Scalar()

        model.stimulus = spa.Transcode(
            lambda t: 'A' if t <= 0.3 else 'B', output_vocab=d)

        spa.sym.A >> model.state_a
        model.stimulus >> model.state_b
        spa.dot(model.state_a, model.state_b) >> model.result
        p = nengo.Probe(model.result.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.6)

    plt.plot(sim.trange(), sim.data[p])

    t = sim.trange()
    assert np.mean(sim.data[p][(t > 0.1) & (t <= 0.3)]) > 0.8
    assert np.mean(sim.data[p][t > 0.4]) < 0.15


class TestExceptions():
    @pytest.fixture
    def model(self):
        with spa.Network() as model:
            model.state_a = spa.State(16)
            model.state_b = spa.State(32)
            model.state_c = spa.State(32)
            model.scalar = spa.Scalar()
        return model

    def test_invalid_types_in_binary_operation(self, model):
        with model:
            with pytest.raises(SpaTypeError):
                model.state_a + model.state_b >> model.state_c

    def test_approx_inverse_of_scalar(self, model):
        with model:
            with pytest.raises(SpaTypeError):
                ~model.scalar >> model.state_c

    def test_dot_product_incompatiple_vocabs(self, model):
        with model:
            with pytest.raises(SpaTypeError):
                spa.dot(model.state_a, model.state_b) >> model.scalar

    def test_dot_product_first_argument_invalid(self, model):
        with model:
            with pytest.raises(SpaTypeError):
                spa.dot(model.scalar, model.state_b) >> model.scalar

    def test_dot_product_second_argument_invalid(self, model):
        with model:
            with pytest.raises(SpaTypeError):
                spa.dot(model.state_a, model.scalar) >> model.scalar

    def test_cast_type_inference_not_possible(self, model):
        with model:
            with pytest.raises(SpaTypeError):
                spa.dot(
                    spa.reinterpret(model.state_a), spa.sym.A) >> model.scalar

    def test_reinterpret_non_matching_dimensions(self, model):
        with model:
            with pytest.raises(SpaTypeError):
                spa.reinterpret(model.state_b) >> model.state_a


def test_access_thal_and_bg_objects():
    d = 16

    with spa.Network() as m:
        m.a = spa.Scalar()
        m.b = spa.Scalar()

        m.c = spa.Scalar()
        m.d = spa.Scalar()

        with spa.ActionSelection() as actions:
            spa.ifmax(m.a, 0 >> m.c)
            spa.ifmax(m.b, 1 >> m.c)

    assert isinstance(actions.bg, spa.BasalGanglia)
    assert isinstance(actions.thalamus, spa.Thalamus)


def test_eval(Simulator):
    with spa.Network() as net:
        a = spa.Transcode(input_vocab=16)
        0.5 * spa.sym.A >> a
        p = nengo.Probe(a.output)

    with Simulator(net) as sim:
        sim.run(1.0)

    assert np.allclose(sim.data[p][-1], net.vocabs[16].parse("0.5*A").v)
