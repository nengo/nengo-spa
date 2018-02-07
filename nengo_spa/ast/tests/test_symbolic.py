import sys

import nengo
from numpy.testing import assert_allclose, assert_equal
import pytest

import nengo_spa as spa
from nengo_spa import sym
from nengo_spa.ast.symbolic import FixedScalar, PointerSymbol
from nengo_spa.types import TVocabulary


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


def test_unary_minus_on_scalar(rng):
    assert (-FixedScalar(1.)).evaluate() == -1.


def test_pointer_symbol_network_creation(rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A')

    with spa.Network():
        A = PointerSymbol('A', TVocabulary(vocab))
        node = A.construct()
    assert_equal(node.output, vocab['A'].v)


@pytest.mark.parametrize('op', ['-', '~'])
def test_unary_operation_on_pointer_symbol(op, rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A')

    with spa.Network():
        x = eval(op + "PointerSymbol('A', TVocabulary(vocab))")
        node = x.construct()
    assert_equal(node.output, vocab.parse(op + 'A').v)


@pytest.mark.parametrize('op', ['+', '-', '*'])
def test_binary_operation_on_pointer_symbols(op, rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A; B')

    with spa.Network():
        v = TVocabulary(vocab)  # noqa: F841
        x = eval("PointerSymbol('A', v)" + op + "PointerSymbol('B', v)")
        node = x.construct()
    assert_equal(node.output, vocab.parse('A' + op + 'B').v)


@pytest.mark.parametrize('op', ['+', '-'])
def test_additive_op_fixed_scalar_and_pointer_symbol(op, rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A')

    with spa.Network():
        with pytest.raises(TypeError):
            eval("2" + op + "PointerSymbol('A')")


def test_multiply_fixed_scalar_and_pointer_symbol(rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A')

    with spa.Network():
        x = 2 * PointerSymbol('A', TVocabulary(vocab))
        node = x.construct()
    assert_equal(node.output, vocab.parse('2 * A').v)


def test_fixed_dot(rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A; B')

    v = TVocabulary(vocab)
    assert_allclose(
        spa.dot(PointerSymbol('A', v), PointerSymbol('A', v)).evaluate(), 1.)
    assert spa.dot(
        PointerSymbol('A', v), PointerSymbol('B', v)).evaluate() <= 0.1


@pytest.mark.skipif(sys.version_info < (3, 5), reason="requires Python 3.5")
def test_fixed_dot_matmul(rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A; B')

    v = TVocabulary(vocab)  # noqa: F841
    assert_allclose(
        eval("PointerSymbol('A', v) @ PointerSymbol('A', v)").evaluate(), 1.)


def test_translate(rng):
    v1 = spa.Vocabulary(16, rng=rng)
    v1.populate('A; B')
    v2 = spa.Vocabulary(16, rng=rng)
    v2.populate('A; B')

    assert_allclose(
        spa.translate(PointerSymbol('A', TVocabulary(v1)), v2).evaluate().dot(
            v2['A']), 1., atol=0.2)


def test_pointer_symbol_factory():
    ps = sym.A
    assert isinstance(ps, PointerSymbol)
    assert ps.expr == 'A'
