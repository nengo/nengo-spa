import sys

import nengo
import numpy as np
from numpy.testing import assert_allclose
import pytest

import nengo_spa as spa
from nengo_spa.ast.symbolic import PointerSymbol
from nengo_spa.exceptions import SpaTypeError
from nengo_spa.pointer import SemanticPointer
from nengo_spa.testing import sp_close


@pytest.mark.parametrize('op', ['-', '~'])
@pytest.mark.parametrize('suffix', ['', '.output'])
def test_unary_operation_on_module(Simulator, op, suffix, rng):
    vocab = spa.Vocabulary(64, rng=rng)
    vocab.populate('A')

    with spa.Network() as model:
        stimulus = spa.Transcode('A', output_vocab=vocab)  # noqa: F841
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
        a = spa.Transcode('A', output_vocab=vocab)  # noqa: F841
        b = spa.Transcode('B', output_vocab=vocab)  # noqa: F841
        x = eval('a' + suffix + op + 'b' + suffix)
        p = nengo.Probe(x.construct(), synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)

    assert sp_close(
        sim.trange(), sim.data[p], vocab.parse('A' + op + 'B'), skip=0.3)


@pytest.mark.parametrize('op', ['+', '-', '*'])
@pytest.mark.parametrize('order', ['AB', 'BA'])
def test_binary_operation_on_modules_with_pointer_symbol(
        Simulator, op, order, rng):
    vocab = spa.Vocabulary(64, rng=rng)
    vocab.populate('A; B')

    with spa.Network() as model:
        a = spa.Transcode('A', output_vocab=vocab)  # noqa: F841
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
    b = SemanticPointer(vocab['B'].v)  # noqa: F841

    with spa.Network() as model:
        a = spa.Transcode('A', output_vocab=vocab)  # noqa: F841
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


def test_transformed_and_pointer_symbol(Simulator, rng):
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
    with spa.Network():
        scalar = spa.Scalar()
        with pytest.raises(SpaTypeError):
            PointerSymbol('A') * scalar


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


@pytest.mark.skipif(sys.version_info < (3, 5), reason="requires Python 3.5")
def test_dot_matmul(Simulator, rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A; B')

    with spa.Network() as model:
        a = spa.Transcode('A', output_vocab=vocab)  # noqa: F841
        b = spa.Transcode(  # noqa: F841
            lambda t: 'A' if t <= 0.5 else 'B', output_vocab=vocab)
        x = eval('a @ b')
        p = nengo.Probe(x.construct(), synapse=0.03)

    with nengo.Simulator(model) as sim:
        sim.run(1.)

    t = sim.trange()
    assert_allclose(sim.data[p][(0.3 < t) & (t <= 0.5)], 1., atol=.2)
    assert np.all(sim.data[p][0.8 < t] < 0.2)


@pytest.mark.skipif(sys.version_info < (3, 5), reason="requires Python 3.5")
def test_dot_with_fixed_matmul(Simulator, rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A; B')

    with spa.Network() as model:
        a = PointerSymbol('A')  # noqa: F841
        b = spa.Transcode(  # noqa: F841
            lambda t: 'A' if t <= 0.5 else 'B', output_vocab=vocab)
        x = eval('a @ b')
        p = nengo.Probe(x.construct(), synapse=0.03)

    with nengo.Simulator(model) as sim:
        sim.run(1.)

    t = sim.trange()
    assert_allclose(sim.data[p][(0.3 < t) & (t <= 0.5)], 1., atol=.2)
    assert np.all(sim.data[p][0.8 < t] < 0.2)


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
