import nengo
import numpy as np
import pytest
from numpy.testing import assert_allclose

import nengo_spa as spa
from nengo_spa.action_selection import ActionSelection
from nengo_spa.ast.symbolic import PointerSymbol
from nengo_spa.exceptions import SpaActionSelectionError, SpaTypeError
from nengo_spa.testing import assert_sp_close


def test_new_action_syntax(Simulator, seed, plt):
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

        model.state = spa.State(label="state")
        model.buff1 = spa.State(label="buff1")
        model.buff2 = spa.State(label="buff2")
        model.buff3 = spa.State(label="buff3")

        node1 = nengo.Node([0, 1, 0])
        node2 = nengo.Node([0, 0, 1])

        nengo.Connection(node1, model.buff1.input)
        nengo.Connection(node2, model.buff2.input)

        model.input >> model.ctrl
        model.buff1 >> model.state
        with spa.ActionSelection():
            spa.ifmax(spa.dot(model.ctrl, spa.sym.A), model.buff1 >> model.buff3)
            spa.ifmax(spa.dot(model.ctrl, spa.sym.B), model.buff2 >> model.buff3)
            spa.ifmax(
                spa.dot(model.ctrl, spa.sym.C), model.buff1 * model.buff2 >> model.buff3
            )

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


def test_dummy_action():
    with spa.Network():
        with spa.ActionSelection():
            spa.ifmax(0)
            spa.ifmax("named-dummy", 0)

            with pytest.raises(ValueError):
                spa.ifmax("must-provide-condition")


def test_dot_product(Simulator, seed, plt):
    d = 16

    with spa.Network(seed=seed) as model:
        model.state_a = spa.State(d)
        model.state_b = spa.State(d)
        model.result = spa.Scalar()

        model.stimulus = spa.Transcode(
            lambda t: "A" if t <= 0.3 else "B", output_vocab=d
        )

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


class TestExceptions:
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
                spa.dot(spa.reinterpret(model.state_a), spa.sym.A) >> model.scalar

    def test_reinterpret_non_matching_dimensions(self, model):
        with model:
            with pytest.raises(SpaTypeError):
                spa.reinterpret(model.state_b) >> model.state_a


def test_access_thal_and_bg_objects():
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


def test_eval(Simulator, seed):
    with spa.Network(seed=seed) as net:
        a = spa.Transcode(input_vocab=16)
        0.5 * spa.sym.A >> a
        p = nengo.Probe(a.output)

    with Simulator(net) as sim:
        sim.run(1.0)

    assert np.allclose(sim.data[p][-1], net.vocabs[16].parse("0.5*A").v)


def test_assignment_of_fixed_scalar(Simulator, seed):
    with spa.Network(seed=seed) as model:
        sink = spa.Scalar()
        0.5 >> sink
        p = nengo.Probe(sink.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)

    assert_allclose(sim.data[p][sim.trange() > 0.3], 0.5, atol=0.2)


def test_assignment_of_pointer_symbol(Simulator, seed, rng):
    vocab = spa.Vocabulary(16, pointer_gen=rng)
    vocab.populate("A")

    with spa.Network(seed=seed) as model:
        sink = spa.State(vocab)
        PointerSymbol("A") >> sink
        p = nengo.Probe(sink.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)

    assert_sp_close(sim.trange(), sim.data[p], vocab["A"], skip=0.3)


def test_assignment_of_dynamic_scalar(Simulator, seed):
    with spa.Network(seed=seed) as model:
        source = spa.Scalar()
        sink = spa.Scalar()
        nengo.Connection(nengo.Node(0.5), source.input)
        source >> sink
        p = nengo.Probe(sink.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)

    assert_allclose(sim.data[p][sim.trange() > 0.3], 0.5, atol=0.2)


@pytest.mark.parametrize(
    "source,use_ens",
    [
        ("nengo.Node(0.5)", False),
        ("nengo.Node([0., 0.5, 1.])[1]", False),
        ("nengo.Node(0.5)", True),
    ],
)
def test_assignment_of_nengo_scalar(source, use_ens, Simulator, seed):
    with spa.Network(seed=seed) as model:
        source = eval(source)
        if use_ens:
            ens = nengo.Ensemble(50, 1)
            nengo.Connection(source, ens)
            source = ens
        sink = spa.Scalar()
        source >> sink
        p = nengo.Probe(sink.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)

    assert_allclose(sim.data[p][sim.trange() > 0.3], 0.5, atol=0.2)


def test_assignment_of_dynamic_pointer(Simulator, seed, rng):
    vocab = spa.Vocabulary(16, pointer_gen=rng)
    vocab.populate("A")

    with spa.Network(seed=seed) as model:
        source = spa.Transcode("A", output_vocab=vocab)
        sink = spa.State(vocab)
        source >> sink
        p = nengo.Probe(sink.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)

    assert_sp_close(sim.trange(), sim.data[p], vocab["A"], skip=0.3)


def test_non_default_input_and_output(Simulator, seed, rng):
    vocab = spa.Vocabulary(32, pointer_gen=rng)
    vocab.populate("A; B")

    with spa.Network(seed=seed) as model:
        a = spa.Transcode("A", output_vocab=vocab)
        b = spa.Transcode("B", output_vocab=vocab)
        bind = spa.Bind(vocab)
        a.output >> bind.input_left
        b.output >> bind.input_right
        p = nengo.Probe(bind.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)

    assert_sp_close(sim.trange(), sim.data[p], vocab.parse("A*B"), skip=0.3, atol=0.3)


@pytest.mark.slow
def test_action_selection(Simulator, seed, rng):
    vocab = spa.Vocabulary(64, pointer_gen=rng)
    vocab.populate("A; B; C; D; E; F")

    with spa.Network(seed=seed) as model:
        state = spa.Transcode(
            lambda t: "ABCDEF"[min(5, int(t / 0.5))], output_vocab=vocab
        )
        scalar = spa.Scalar()
        pointer = spa.State(vocab)
        with ActionSelection():
            spa.ifmax(spa.dot(state, PointerSymbol("A")), 0.5 >> scalar)
            spa.ifmax(spa.dot(state, PointerSymbol("B")), PointerSymbol("B") >> pointer)
            spa.ifmax(spa.dot(state, PointerSymbol("C")), state >> pointer)
            d_utility = spa.ifmax(0, PointerSymbol("D") >> pointer)
            spa.ifmax(
                spa.dot(state, PointerSymbol("E")),
                0.25 >> scalar,
                PointerSymbol("E") >> pointer,
            )
        nengo.Connection(nengo.Node(lambda t: 1.5 < t <= 2.0), d_utility)
        p_scalar = nengo.Probe(scalar.output, synapse=0.03)
        p_pointer = nengo.Probe(pointer.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(3.0)

    t = sim.trange()
    assert_allclose(sim.data[p_scalar][(0.3 < t) & (t <= 0.5)], 0.5, atol=0.2)
    assert_sp_close(
        sim.trange(), sim.data[p_pointer], vocab["B"], skip=0.8, duration=0.2
    )
    assert_sp_close(
        sim.trange(), sim.data[p_pointer], vocab["C"], skip=1.3, duration=0.2
    )
    assert_sp_close(
        sim.trange(), sim.data[p_pointer], vocab["D"], skip=1.8, duration=0.2
    )
    assert_allclose(sim.data[p_scalar][(2.3 < t) & (t <= 2.5)], 0.25, atol=0.2)
    assert_sp_close(
        sim.trange(), sim.data[p_pointer], vocab["E"], skip=2.3, duration=0.2
    )


def test_spa_connections_to_utilities():
    with spa.Network():
        with ActionSelection():
            utility = spa.ifmax(0.0)
        0.1 >> utility


def test_does_not_allow_nesting_of_action_selection():
    with spa.Network():
        with ActionSelection():
            with pytest.raises(SpaActionSelectionError):
                with ActionSelection():
                    pass


def test_action_selection_enforces_connections_to_be_part_of_action():
    with spa.Network():
        state1 = spa.State(16)
        state2 = spa.State(16)
        with pytest.raises(SpaActionSelectionError):
            with ActionSelection():
                state1 >> state2


def test_action_selection_is_not_built_on_exception():
    with spa.Network():
        state1 = spa.State(16)
        state2 = spa.State(32)
        with pytest.raises(SpaTypeError):
            with ActionSelection() as action_sel:
                spa.ifmax(1.0, state1 >> state2)
    assert not action_sel.built


def test_naming_of_actions():
    with spa.Network():
        state1 = spa.State(16)
        state2 = spa.State(16)
        with ActionSelection() as action_sel:
            u0 = spa.ifmax("name0", 0.0, state1 >> state2)
            u1 = spa.ifmax(0.0, state1 >> state2)
            u2 = spa.ifmax("name2", 0.0, state1 >> state2)

    assert tuple(action_sel.keys()) == ("name0", 1, "name2")
    assert action_sel["name0"] is u0
    assert action_sel["name2"] is u2
    for i, u in enumerate((u0, u1, u2)):
        assert action_sel[i] is u


def test_action_selection_keys_corner_cases():
    with spa.Network():
        with ActionSelection() as action_sel:
            pass
    assert list(action_sel.keys()) == []

    with spa.Network():
        with ActionSelection() as action_sel:
            spa.ifmax(0.0)
    assert list(action_sel.keys()) == [0]


def test_action_selection_is_side_effect_free_if_exception_is_raised():
    with spa.Network():
        state_a = spa.State(32)
        state_b = spa.State(32)
        state_c = spa.State(64)

        with pytest.raises(SpaTypeError):
            with ActionSelection():
                spa.ifmax(1, state_a >> state_b, state_a >> state_c)

        with ActionSelection():
            pass
            # should not raise
