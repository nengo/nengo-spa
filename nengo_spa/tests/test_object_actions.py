import numpy as np
import pytest

import nengo
import nengo_spa as spa
from nengo_spa import object_actions as oact
from nengo_spa.compiler import ast_nodes
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

        # spa.Actions('''
        #     model.input -> model.ctrl
        #     model.buff1 -> model.state
        #     ifmax dot(model.ctrl, A):
        #         model.buff1 -> model.buff3
        #     elifmax dot(model.ctrl, B):
        #         model.buff2 -> model.buff3
        #     elifmax dot(model.ctrl, C):
        #         model.buff1 * model.buff2 -> model.buff3
        # ''')
        with oact.Actions():
            oact.route(model.input, model.ctrl),
            oact.route(model.buff1, model.state))
            oact.cond(oact.dot(model.ctrl, "A"), 
                      oact.route(model.buff1, model.buff3))
            oact.cond(oact.dot(model.ctrl, "B"), 
                      oact.route(model.buff2, model.buff3))
            oact.cond(oact.dot(model.ctrl, "C"),
                      oact.route(model.buff1 * model.buff2, model.buff3))

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

        # spa.Actions('''
        #     A -> model.state_a
        #     model.stimulus -> model.state_b
        #     dot(model.state_a, model.state_b) -> model.result
        # ''')
        with oact.Actions():
            oact.route("A", model.state_a)
            oact.route(model.stimulus, model.state_b)
            oact.route(oact.dot(model.state_a, model.state_b), model.result)
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
                # spa.Actions('model.state_a + model.state_b -> model.state_c')
                with oact.Actions():
                    oact.route(model.state_a + model.state_b, model.state_c)

    def test_approx_inverse_of_scalar(self, model):
        with model:
            with pytest.raises(SpaTypeError):
                # spa.Actions('~model.scalar -> model.state_c')
                with oact.Actions():
                    oact.route(~model.scalar, model.state_c)

    def test_dot_product_incompatiple_vocabs(self, model):
        with model:
            with pytest.raises(SpaTypeError):
                # spa.Actions(
                #     'dot(model.state_a, model.state_b) -> model.scalar')
                with oact.Actions():
                    oact.route(oact.dot(model.state_a, model.state_b),
                               model.scalar)

    def test_dot_product_first_argument_invalid(self, model):
        with model:
            with pytest.raises(SpaTypeError):
                # spa.Actions(
                #     'dot(model.scalar, model.state_b) -> model.scalar')
                with oact.Actions():
                    oact.route(oact.dot(model.scalar, model.state_b),
                               model.scalar)

    def test_dot_product_second_argument_invalid(self, model):
        with model:
            with pytest.raises(SpaTypeError):
                # spa.Actions(
                #     'dot(model.state_a, model.scalar) -> model.scalar')
                with oact.Actions():
                    oact.route(oact.dot(model.state_a, model.scalar),
                               model.scalar)

    @pytest.mark.parametrize('method', ['reinterpret', 'translate'])
    def test_cast_type_inference_not_possible(self, model, method):
        with model:
            with pytest.raises(SpaTypeError):
                # spa.Actions('dot({}(model.state_a), A) -> model.scalar'.format(
                #     method))
                method = getattr(oact, method)
                with oact.Actions():
                    oact.route(oact.dot(method(model.state_a), "A"),
                               model.scalar)

    @pytest.mark.parametrize('method', ['reinterpret', 'translate'])
    def test_cast_scalar(self, model, method):
        with model:
            with pytest.raises(SpaTypeError):
                # spa.Actions(
                #     '{}(model.scalar) -> model.state_a'.format(method))
                method = getattr(oact, method)
                with oact.Actions():
                    oact.route(method(model.scalar), model.state_a)

    def test_reinterpret_non_matching_dimensions(self, model):
        with model:
            with pytest.raises(SpaTypeError):
                # spa.Actions('reinterpret(model.state_b) -> model.state_a')
                with oact.Actions():
                    oact.route(oact.reinterpret(model.state_b),
                               model.state_a)


def test_access_actions():
    d = 16
    with spa.Network() as m:
        m.a = spa.State(d)
        m.b = spa.State(d)
        m.c = spa.State(d)
        m.d = spa.State(d)
        # actions = spa.Actions('''
        #     m.b -> m.a
        #     m.c -> m.b
        #     always as 'named':
        #         m.d -> m.c
        #     ''')
        with oact.Actions() as actions:
            oact.route(m.b, m.a)
            oact.route(m.c, m.b)
            oact.route(m.d, m.c)

    assert len(actions) == 3
    # assert str(actions[0]) == 'm.b -> m.a'
    assert str(actions[0]) == "%s -> %s" % (m.b, m.a)
    # assert str(actions[1]) == 'm.c -> m.b'
    assert str(actions[1]) == "%s -> %s" % (m.c, m.b)

    # this is being removed
    # https://github.com/nengo/nengo_spa/pull/100
    # assert str(actions[2]) == '''always as 'named':
    # m.d -> m.c'''
    # assert str(actions['named']) == '''always as 'named':
    # m.d -> m.c'''


def test_access_thal_and_bg_objects():
    d = 16

    with spa.Network() as m:
        m.a = spa.Scalar()
        m.b = spa.Scalar()

        m.c = spa.Scalar()
        m.d = spa.Scalar()

        # actions = spa.Actions('''
        #     ifmax m.a:
        #         0 -> m.c
        #     ifmax m.b:
        #         1 -> m.c
        #     m.c -> m.d
        #     ''')
        with oact.Actions() as actions:
            oact.cond(m.a, oact.route("0", m.c))
        with actions:
            oact.cond(m.b, oact.route("1", m.c))
            oact.route(m.c, m.d)

    assert actions.all_bgs() == [actions[0].bg, actions[1].bg]
    assert actions.all_thals() == [actions[0].thalamus, actions[1].thalamus]

    with spa.Network() as m:
        m.a = spa.State(d)
        m.b = spa.State(d)

        # actions = spa.Actions('''
        #     m.a -> m.b
        #     ''')
        with oact.Actions() as actions:
            oact.route(m.a, m.b)

    assert len(actions.all_bgs()) == 0
    assert len(actions.all_thals()) == 0


def test_provides_access_to_constructed_objects_of_effect():
    with spa.Network() as model:
        model.config[spa.State].vocab = 16
        model.a = spa.State()
        model.b = spa.State()
        model.c = spa.State()

        # actions = spa.Actions('model.a * model.b -> model.c')
        with oact.Actions() as actions:
            oact.route(model.a * model.b, model.c)

        assert len(actions[0].constructed) == 1
        assert isinstance(actions[0].constructed[0], nengo.Connection)
        assert len(actions[0].source.constructed) == 3
        n_connections = 0
        n_bind = 0
        for obj in actions[0].source.constructed:
            if isinstance(obj, nengo.Connection):
                n_connections += 1
            elif isinstance(obj, spa.Bind):
                n_bind += 1
            else:
                raise AssertionError("Unexpected object constructed for Bind.")
        assert n_connections == 2 and n_bind == 1


def test_eval(Simulator):
    with spa.Network() as net:
        a = spa.Transcode(input_vocab=16)
        with oact.Actions():
            oact.route("0.5*A", a)
        p = nengo.Probe(a.output)

    with Simulator(net) as sim:
        sim.run(1.0)

    assert np.allclose(sim.data[p][-1], net.vocabs[16].parse("0.5*A").v)


def test_actions_context():
    with spa.Network():
        a = spa.State(16)
        b = "B"

        network_dict = {k: v for k, v in spa.Network.__dict__.items()}

        with oact.Actions():
            assert isinstance(~a, ast_nodes.ApproxInverse)
            assert isinstance(-a, ast_nodes.Negative)
            assert isinstance(a + b, ast_nodes.Sum)
            x = a - b
            assert isinstance(x, ast_nodes.Sum)
            assert isinstance(x.rhs, ast_nodes.Negative)
            assert isinstance(x.rhs.source, ast_nodes.Symbol)
            x = b - a
            assert isinstance(x, ast_nodes.Sum)
            assert isinstance(x.rhs, ast_nodes.Negative)
            assert isinstance(x.rhs.source, ast_nodes.Module)
            assert isinstance(a * b, ast_nodes.Product)

        # make sure that things are reset after exiting context
        with pytest.raises(TypeError):
            a + b

        assert spa.Network.__dict__ == network_dict


@pytest.mark.parametrize('d1,d2,method,lookup', [
    (16, 16, 'reinterpret(a, v2)', 'v1'),
    (16, 16, 'reinterpret(a)', 'v1'),
    (16, 16, 'reinterpret(a, b)', 'v1'),
    (16, 32, 'translate(a, v2)', 'v2'),
    (16, 32, 'translate(a)', 'v2'),
    (16, 32, 'translate(a, b)', 'v2'),
    (16, 32, 'translate(a, solver=nengo.solvers.Lstsq())', 'v2')])
def test_casting_vocabs(d1, d2, method, lookup, Simulator, plt, rng):
    v1 = spa.Vocabulary(d1, rng=rng)
    v1.populate('A')
    v2 = spa.Vocabulary(d2, rng=rng)
    v2.populate('A')

    with spa.Network() as model:
        a = spa.State(vocab=v1)
        b = spa.State(vocab=v2)
        # spa.Actions(
        #     'A -> a; {} -> b'.format(method))
        with oact.Actions():
            oact.route("A", a)
            oact.route(eval("oact.%s" % method), b)
        p = nengo.Probe(b.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)

    t = sim.trange() > 0.2
    v = locals()[lookup].parse('A').v

    plt.plot(sim.trange(), spa.similarity(sim.data[p], v))
    plt.xlabel("t [s]")
    plt.ylabel("Similarity")

    assert np.mean(spa.similarity(sim.data[p][t], v)) > 0.8
