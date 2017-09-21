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

        spa.Actions((
            'model.ctrl = model.input',
            'model.state = model.buff1',
            'dot(model.ctrl, A) --> model.buff3 = model.buff1',
            'dot(model.ctrl, B) --> model.buff3 = model.buff2',
            'dot(model.ctrl, C) --> model.buff3 = model.buff1 * model.buff2'
        ))

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

        spa.Actions((
            'model.state_a = A',
            'model.state_b = model.stimulus',
            'model.result = dot(model.state_a, model.state_b)'
        ))

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
                spa.Actions(('model.state_c = model.state_a + model.state_b',))

    def test_approx_inverse_of_scalar(self, model):
        with model:
            with pytest.raises(SpaTypeError):
                spa.Actions(('model.state_c = ~model.scalar',))

    def test_dot_product_incompatiple_vocabs(self, model):
        with model:
            with pytest.raises(SpaTypeError):
                spa.Actions(
                    ('model.scalar = dot(model.state_a, model.state_b)',))

    def test_dot_product_first_argument_invalid(self, model):
        with model:
            with pytest.raises(SpaTypeError):
                spa.Actions(
                    ('model.scalar = dot(model.scalar, model.state_b)',))

    def test_dot_product_second_argument_invalid(self, model):
        with model:
            with pytest.raises(SpaTypeError):
                spa.Actions(
                    ('model.scalar = dot(model.state_a, model.scalar)',))

    @pytest.mark.parametrize('method', ['reinterpret', 'translate'])
    def test_cast_type_inference_not_possible(self, model, method):
        with model:
            with pytest.raises(SpaTypeError):
                spa.Actions(('model.scalar = dot({}(model.state_a), A)'.format(
                    method),))

    @pytest.mark.parametrize('method', ['reinterpret', 'translate'])
    def test_cast_scalar(self, model, method):
        with model:
            with pytest.raises(SpaTypeError):
                spa.Actions(
                    ('model.state_a = {}(model.scalar)'.format(method),))

    def test_reinterpret_non_matching_dimensions(self, model):
        with model:
            with pytest.raises(SpaTypeError):
                spa.Actions(('model.state_a = reinterpret(model.state_b)',))


def test_access_actions():
    d = 16
    with spa.Network() as m:
        m.a = spa.State(d)
        m.b = spa.State(d)
        m.c = spa.State(d)
        m.d = spa.State(d)
    actions = spa.Actions(
        ('m.a = m.b', 'm.b = m.c'), {'named': 'm.c = m.d'}, build=False)

    assert len(actions) == 3
    assert str(actions[0]) == 'm.b -> m.a'
    assert str(actions[1]) == 'm.c -> m.b'
    assert str(actions[2]) == 'm.d -> m.c'
    assert str(actions['named']) == 'm.d -> m.c'


def test_provides_access_to_constructed_objects_of_effect():
    with spa.Network() as model:
        model.config[spa.State].vocab = 16
        model.a = spa.State()
        model.b = spa.State()
        model.c = spa.State()

        actions = spa.Actions(('model.c = model.a * model.b',), build=False)
        bg, thalamus, constructed = actions.build()

        assert len(actions[0].effects[0].constructed) == 1
        assert isinstance(
            actions[0].effects[0].constructed[0], nengo.Connection)
        assert len(actions[0].effects[0].source.constructed) == 3
        n_connections = 0
        n_bind = 0
        for obj in actions[0].effects[0].source.constructed:
            if isinstance(obj, nengo.Connection):
                n_connections += 1
            elif isinstance(obj, spa.Bind):
                n_bind += 1
            else:
                raise AssertionError("Unexpected object constructed for Bind.")
        assert n_connections == 2 and n_bind == 1


def test_bg_and_thalamus_only_created_when_required():
    with spa.Network() as model:
        model.state1 = spa.State(16)
        model.state2 = spa.State(16)
        actions = spa.Actions(('model.state1 = model.state2',))

    assert actions.bg is None
    assert actions.thalamus is None
