import nengo
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

import nengo_spa as spa
from nengo_spa.ast2 import coerce_types
from nengo_spa.ast_dynamic import ActionSelection, ifmax
from nengo_spa.ast_symbolic import PointerSymbol, sym
from nengo_spa.exceptions import SpaTypeError
from nengo_spa.network import create_inhibit_node
from nengo_spa.pointer import SemanticPointer
from nengo_spa.testing import sp_close
from nengo_spa.types import TInferVocab, TScalar, TVocabulary


def test_coercion():
    v1 = TVocabulary(spa.Vocabulary(16))
    v2 = TVocabulary(spa.Vocabulary(16))

    assert coerce_types(TInferVocab, TInferVocab) is TInferVocab
    assert coerce_types(TInferVocab, TScalar) is TInferVocab
    assert coerce_types(TInferVocab, v1) == v1
    assert coerce_types(TScalar, TScalar) == TScalar
    assert coerce_types(TScalar, TScalar, TScalar) == TScalar
    assert coerce_types(TScalar, v1) == v1
    assert coerce_types(v1, v1) == v1
    assert coerce_types(TInferVocab, v1, TScalar) == v1
    assert coerce_types(TScalar, TScalar, v1, TScalar, v1) == v1
    with pytest.raises(SpaTypeError):
        coerce_types(v1, v2)


def test_pointer_symbol_network_creation(rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A')

    with spa.Network() as model:
        A = PointerSymbol('A', TVocabulary(vocab))
        node = A.construct()
    assert_equal(node.output, vocab['A'].v)


def test_fixed_pointer_network_creation(rng):
    with spa.Network() as model:
        A = SemanticPointer(16)
        node = A.construct()
    assert_equal(node.output, A.v)


@pytest.mark.parametrize('op', ['-', '~'])
def test_unary_operation_on_pointer_symbol(op, rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A')

    with spa.Network() as model:
        x = eval(op + "PointerSymbol('A', TVocabulary(vocab))")
        node = x.construct()
    assert_equal(node.output, vocab.parse(op + 'A').v)


@pytest.mark.parametrize('op', ['+', '-', '*'])
def test_binary_operation_on_pointer_symbols(op, rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A; B')

    with spa.Network() as model:
        v = TVocabulary(vocab)
        x = eval("PointerSymbol('A', v)" + op + "PointerSymbol('B', v)")
        node = x.construct()
    assert_equal(node.output, vocab.parse('A' + op + 'B').v)


def test_multiply_fixed_scalar_and_pointer_symbol(rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A')

    with spa.Network() as model:
        x = 2 * PointerSymbol('A', TVocabulary(vocab))
        node = x.construct()
    assert_equal(node.output, vocab.parse('2 * A').v)


@pytest.mark.parametrize('op', ['+', '-'])
def test_additive_op_fixed_scalar_and_pointer_symbol(op, rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A')

    with spa.Network() as model:
        with pytest.raises(TypeError):
            eval("2" + op + "PointerSymbol('A')")


@pytest.mark.parametrize('op', ['-', '~'])
@pytest.mark.parametrize('suffix', ['', '.output'])
def test_unary_operation_on_module(Simulator, op, suffix, rng):
    vocab = spa.Vocabulary(64, rng=rng)
    vocab.populate('A')

    with spa.Network() as model:
        stimulus = spa.Transcode('A', output_vocab=vocab)
        x = eval(op + 'stimulus' + suffix)
        p = nengo.Probe(x.construct(), synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)

    assert sp_close(sim.trange(), sim.data[p], vocab.parse(op + 'A'), skip=0.3)


@pytest.mark.parametrize('op', ['+', '-', '*'])
@pytest.mark.parametrize('suffix', ['', '.output'])
def test_binary_operation_on_modules(Simulator, op, suffix, rng):
    vocab = spa.Vocabulary(64, rng=rng)
    vocab.populate('A; B')

    with spa.Network() as model:
        a = spa.Transcode('A', output_vocab=vocab)
        b = spa.Transcode('B', output_vocab=vocab)
        x = eval('a' + suffix + op + 'b' + suffix)
        p = nengo.Probe(x.construct(), synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)

    assert sp_close(
        sim.trange(), sim.data[p], vocab.parse('A' + op + 'B'), skip=0.3)


def test_product_of_scalars(Simulator):
    with spa.Network() as model:
        stimulus = nengo.Node(0.5)
        a = spa.Scalar()
        b = spa.Scalar()
        nengo.Connection(stimulus, a.input)
        nengo.Connection(stimulus, b.input)
        x = a * b
        p = nengo.Probe(x.construct(), synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)

    assert_allclose(sim.data[p][sim.trange() > 0.3], .25, atol=0.2)


@pytest.mark.parametrize('op', ['+', '-', '*'])
@pytest.mark.parametrize('order', ['AB', 'BA'])
def test_binary_operation_on_modules_with_pointer_symbol(
        Simulator, op, order, rng):
    vocab = spa.Vocabulary(64, rng=rng)
    vocab.populate('A; B')

    with spa.Network() as model:
        a = spa.Transcode('A', output_vocab=vocab)
        if order == 'AB':
            x = eval('a' + op + 'PointerSymbol("B")')
        elif order == 'BA':
            x = eval('PointerSymbol("B")' + op + 'a')
        else:
            raise ValueError('Invalid order argument.')
        p = nengo.Probe(x.construct(), synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)

    assert sp_close(
        sim.trange(), sim.data[p], vocab.parse(order[0] + op + order[1]),
        skip=0.3)


@pytest.mark.parametrize('op', ['+', '-', '*'])
@pytest.mark.parametrize('order', ['AB', 'BA'])
def test_binary_operation_on_modules_with_fixed_pointer(
        Simulator, op, order, rng):
    vocab = spa.Vocabulary(64, rng=rng)
    vocab.populate('A; B')
    b = SemanticPointer(vocab['B'].v)

    with spa.Network() as model:
        a = spa.Transcode('A', output_vocab=vocab)
        if order == 'AB':
            x = eval('a' + op + 'b')
        elif order == 'BA':
            x = eval('b' + op + 'a')
        else:
            raise ValueError('Invalid order argument.')
        p = nengo.Probe(x.construct(), synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)

    assert sp_close(
        sim.trange(), sim.data[p], vocab.parse(order[0] + op + order[1]),
        skip=0.3)


@pytest.mark.parametrize('op', ['+', '-', '*'])
@pytest.mark.parametrize('order', ['AB', 'BA'])
def test_binary_operation_on_fixed_pointer_with_pointer_symbol(
        Simulator, op, order, rng):
    vocab = spa.Vocabulary(64, rng=rng)
    vocab.populate('A; B')
    a = PointerSymbol('A', TVocabulary(vocab))
    b = SemanticPointer(vocab['B'].v)

    with spa.Network() as model:
        if order == 'AB':
            x = eval('a' + op + 'b')
        elif order == 'BA':
            x = eval('b' + op + 'a')
        else:
            raise ValueError('Invalid order argument.')
        p = nengo.Probe(x.construct(), synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)

    assert sp_close(
        sim.trange(), sim.data[p], vocab.parse(order[0] + op + order[1]),
        skip=0.3)


def test_complex_rule(Simulator, rng):
    vocab = spa.Vocabulary(64, rng=rng)
    vocab.populate('A; B; C; D')

    with spa.Network() as model:
        a = spa.Transcode('A', output_vocab=vocab)
        b = spa.Transcode('B', output_vocab=vocab)

        x = (0.5 * PointerSymbol('C') * a + 0.5 * PointerSymbol('D')) * (
            0.5 * b + a * 0.5)
        p = nengo.Probe(x.construct(), synapse=0.3)

    with nengo.Simulator(model) as sim:
        sim.run(0.5)

    assert sp_close(
        sim.trange(), sim.data[p],
        vocab.parse('(0.5 * C * A + 0.5 * D) * (0.5 * B + 0.5 * A)'),
        skip=0.3, normalized=True)


def test_transformed(Simulator, rng):
    vocab = spa.Vocabulary(64, rng=rng)
    vocab.populate('A; B')

    with spa.Network() as model:
        a = spa.Transcode('A', output_vocab=vocab)
        x = PointerSymbol('B') * a
        p = nengo.Probe(x.construct(), synapse=0.3)

    with nengo.Simulator(model) as sim:
        sim.run(0.5)

    assert sp_close(
        sim.trange(), sim.data[p], vocab.parse('B*A'), skip=0.3,
        normalized=True)


def test_transformed_and_pointer_symbol(Simulator, rng, plt):
    vocab = spa.Vocabulary(64, rng=rng)
    vocab.populate('A; B')

    with spa.Network() as model:
        a = spa.Transcode('A', output_vocab=vocab)
        x = PointerSymbol('~B') * (PointerSymbol('B') * a)
        p = nengo.Probe(x.construct(), synapse=0.3)

    with nengo.Simulator(model) as sim:
        sim.run(0.5)

    assert sp_close(
        sim.trange(), sim.data[p], vocab.parse('~B * B * A'), skip=0.3,
        normalized=True)


def test_transformed_and_network(Simulator, rng):
    vocab = spa.Vocabulary(64, rng=rng)
    vocab.populate('A; B.unitary()')

    with spa.Network() as model:
        a = spa.Transcode('A', output_vocab=vocab)
        b = spa.Transcode('B', output_vocab=vocab)
        x = b * (PointerSymbol('~B') * a)
        p = nengo.Probe(x.construct(), synapse=0.3)

    with nengo.Simulator(model) as sim:
        sim.run(0.5)

    assert sp_close(
        sim.trange(), sim.data[p], vocab.parse('B * ~B * A'), skip=0.3,
        normalized=True)


def test_transformed_and_transformed(Simulator, rng):
    vocab = spa.Vocabulary(64, rng=rng)
    vocab.populate('A; B.unitary(); C')

    with spa.Network() as model:
        a = spa.Transcode('A', output_vocab=vocab)
        c = spa.Transcode('C', output_vocab=vocab)
        x = (PointerSymbol('B') * a) * (PointerSymbol('~B') * c)
        p = nengo.Probe(x.construct(), synapse=0.3)

    with nengo.Simulator(model) as sim:
        sim.run(0.5)

    assert sp_close(
        sim.trange(), sim.data[p], vocab.parse('B * A * ~B * C'), skip=0.3,
        normalized=True)


def test_pointer_symbol_with_dynamic_scalar(Simulator, rng):
    with spa.Network() as model:
        scalar = spa.Scalar()
        with pytest.raises(SpaTypeError):
            PointerSymbol('A') * scalar


def test_assignment_of_fixed_scalar(Simulator, rng):
    with spa.Network() as model:
        sink = spa.Scalar()
        0.5 >> sink
        p = nengo.Probe(sink.output, synapse=0.03)

    with nengo.Simulator(model) as sim:
        sim.run(0.5)

    assert_allclose(sim.data[p][sim.trange() > 0.3], 0.5, atol=0.2)


def test_assignment_of_pointer_symbol(Simulator, rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A')

    with spa.Network() as model:
        sink = spa.State(vocab)
        PointerSymbol('A') >> sink
        p = nengo.Probe(sink.output, synapse=0.03)

    with nengo.Simulator(model) as sim:
        sim.run(0.5)

    assert sp_close(sim.trange(), sim.data[p], vocab['A'], skip=0.3)


def test_assignment_of_dynamic_scalar(Simulator, rng):
    with spa.Network() as model:
        source = spa.Scalar()
        sink = spa.Scalar()
        nengo.Connection(nengo.Node(0.5), source.input)
        source >> sink
        p = nengo.Probe(sink.output, synapse=0.03)

    with nengo.Simulator(model) as sim:
        sim.run(0.5)

    assert_allclose(sim.data[p][sim.trange() > 0.3], 0.5, atol=0.2)


def test_assignment_of_dynamic_pointer(Simulator, rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A')

    with spa.Network() as model:
        source = spa.Transcode('A', output_vocab=vocab)
        sink = spa.State(vocab)
        source >> sink
        p = nengo.Probe(sink.output, synapse=0.03)

    with nengo.Simulator(model) as sim:
        sim.run(0.5)

    assert sp_close(sim.trange(), sim.data[p], vocab['A'], skip=0.3)


def test_non_default_input_and_output(Simulator, rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A; B')

    with spa.Network() as model:
        a = spa.Transcode('A', output_vocab=vocab)
        b = spa.Transcode('B', output_vocab=vocab)
        bind = spa.Bind(vocab)
        a.output >> bind.input_a
        b.output >> bind.input_b
        p = nengo.Probe(bind.output, synapse=0.03)

    with nengo.Simulator(model) as sim:
        sim.run(0.5)

    assert sp_close(sim.trange(), sim.data[p], vocab.parse('A*B'), skip=0.3)


def test_dot(Simulator, rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A; B')

    with spa.Network() as model:
        a = spa.Transcode('A', output_vocab=vocab)
        b = spa.Transcode(
            lambda t: 'A' if t <= 0.5 else 'B', output_vocab=vocab)
        x = spa.dot(a, b)
        p = nengo.Probe(x.construct(), synapse=0.03)

    with nengo.Simulator(model) as sim:
        sim.run(1.)

    t = sim.trange()
    assert_allclose(sim.data[p][(0.3 < t) & (t <= 0.5)], 1., atol=.2)
    assert np.all(sim.data[p][0.8 < t] < 0.2)


def test_dot_with_fixed(Simulator, rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A; B')

    with spa.Network() as model:
        a = PointerSymbol('A')
        b = spa.Transcode(
            lambda t: 'A' if t <= 0.5 else 'B', output_vocab=vocab)
        x = spa.dot(a, b)
        p = nengo.Probe(x.construct(), synapse=0.03)

    with nengo.Simulator(model) as sim:
        sim.run(1.)

    t = sim.trange()
    assert_allclose(sim.data[p][(0.3 < t) & (t <= 0.5)], 1., atol=.2)
    assert np.all(sim.data[p][0.8 < t] < 0.2)


def test_fixed_dot(rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A; B')

    v = TVocabulary(vocab)
    assert_allclose(
        spa.dot(PointerSymbol('A', v), PointerSymbol('A', v)).evaluate(), 1.)
    assert spa.dot(
        PointerSymbol('A', v), PointerSymbol('B', v)).evaluate() <= 0.1


def test_translate(rng):
    v1 = spa.Vocabulary(16, rng=rng)
    v1.populate('A; B')
    v2 = spa.Vocabulary(16, rng=rng)
    v2.populate('A; B')

    assert_allclose(
        spa.translate(PointerSymbol('A', TVocabulary(v1)), v2).evaluate().dot(
            v2['A']), 1., atol=0.2)


def test_dynamic_translate(Simulator, rng):
    v1 = spa.Vocabulary(64, rng=rng)
    v1.populate('A; B')
    v2 = spa.Vocabulary(64, rng=rng)
    v2.populate('A; B')

    with spa.Network() as model:
        source = spa.Transcode('A', output_vocab=v1)
        x = spa.translate(source, v2)
        p = nengo.Probe(x.construct(), synapse=0.03)

    with nengo.Simulator(model) as sim:
        sim.run(0.5)

    assert sp_close(sim.trange(), sim.data[p], v2['A'], skip=0.3, atol=0.2)


def test_action_selection(Simulator, rng):
    vocab = spa.Vocabulary(64)
    vocab.populate('A; B; C; D; E; F')

    with spa.Network() as model:
        state = spa.Transcode(
            lambda t: 'ABCDEF'[min(5, int(t / 0.5))], output_vocab=vocab)
        scalar = spa.Scalar()
        pointer = spa.State(vocab)
        with ActionSelection() as action_sel:
            ifmax(spa.dot(state, PointerSymbol('A')), 0.5 >> scalar)
            ifmax(
                spa.dot(state, PointerSymbol('B')),
                PointerSymbol('B') >> pointer)
            ifmax(spa.dot(state, PointerSymbol('C')), state >> pointer)
            d_utility = ifmax(0, PointerSymbol('D') >> pointer)
            ifmax(
                spa.dot(state, PointerSymbol('E')),
                0.25 >> scalar, PointerSymbol('E') >> pointer)
        nengo.Connection(
            nengo.Node(lambda t: 1.5 < t <= 2.), d_utility.input)
        p_scalar = nengo.Probe(scalar.output, synapse=0.03)
        p_pointer = nengo.Probe(pointer.output, synapse=0.03)

    with nengo.Simulator(model) as sim:
        sim.run(3.)

    t = sim.trange()
    assert_allclose(sim.data[p_scalar][(0.3 < t) & (t <= 0.5)], 0.5, atol=0.2)
    assert sp_close(
        sim.trange(), sim.data[p_pointer], vocab['B'], skip=0.8, duration=0.2)
    assert sp_close(
        sim.trange(), sim.data[p_pointer], vocab['C'], skip=1.3, duration=0.2)
    assert sp_close(
        sim.trange(), sim.data[p_pointer], vocab['D'], skip=1.8, duration=0.2)
    assert_allclose(sim.data[p_scalar][(2.3 < t) & (t <= 2.5)], 0.25, atol=0.2)
    assert sp_close(
        sim.trange(), sim.data[p_pointer], vocab['E'], skip=2.3, duration=0.2)


def test_does_not_allow_nesting_of_action_selection():
    with spa.Network():
        with ActionSelection():
            with pytest.raises(RuntimeError):
                with ActionSelection():
                    pass


def test_action_selection_enforces_connections_to_be_part_of_action():
    with spa.Network():
        state1 = spa.State(16)
        state2 = spa.State(16)
        with pytest.raises(RuntimeError):
            with ActionSelection():
                    state1 >> state2


def test_pointer_symbol_factory():
    ps = sym.A
    assert isinstance(ps, PointerSymbol)
    assert ps.expr == 'A'
