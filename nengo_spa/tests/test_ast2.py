import nengo
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

import nengo_spa as spa
from nengo_spa.ast2 import coerce_types, FixedPointer
from nengo_spa.exceptions import SpaTypeError
from nengo_spa.testing import sp_close
from nengo_spa.types import TScalar, TVocabulary


def test_coercion():
    v1 = TVocabulary(spa.Vocabulary(16))
    v2 = TVocabulary(spa.Vocabulary(16))

    assert coerce_types(None, None) is None
    assert coerce_types(None, TScalar) is None
    assert coerce_types(None, v1) == v1
    assert coerce_types(TScalar, TScalar) == TScalar
    assert coerce_types(TScalar, TScalar, TScalar) == TScalar
    assert coerce_types(TScalar, v1) == v1
    assert coerce_types(v1, v1) == v1
    assert coerce_types(None, v1, TScalar) == v1
    assert coerce_types(TScalar, TScalar, v1, TScalar, v1) == v1
    with pytest.raises(SpaTypeError):
        coerce_types(v1, v2)


def test_fixed_pointer_network_creation(rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A')

    with spa.Network() as model:
        A = FixedPointer('A')
        A.infer_types(TVocabulary(vocab))
        node = A.construct()
    assert_equal(node.output, vocab['A'].v)


@pytest.mark.parametrize('op', ['-', '~'])
def test_unary_operation_on_fixed_pointer(op, rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A')

    with spa.Network() as model:
        x = eval(op + "FixedPointer('A')")
        x.infer_types(TVocabulary(vocab))
        node = x.construct()
    assert_equal(node.output, vocab.parse(op + 'A').v)


@pytest.mark.parametrize('op', ['+', '-', '*'])
def test_binary_operation_on_fixed_pointers(op, rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A; B')

    with spa.Network() as model:
        x = eval("FixedPointer('A')" + op + "FixedPointer('B')")
        x.infer_types(TVocabulary(vocab))
        node = x.construct()
    assert_equal(node.output, vocab.parse('A' + op + 'B').v)


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
def test_binary_operation_on_modules_with_fixed_pointer(
        Simulator, op, order, rng):
    vocab = spa.Vocabulary(64, rng=rng)
    vocab.populate('A; B')

    with spa.Network() as model:
        a = spa.Transcode('A', output_vocab=vocab)
        if order == 'AB':
            x = eval('a' + op + 'FixedPointer("B")')
        elif order == 'BA':
            x = eval('FixedPointer("B")' + op + 'a')
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

        x = (0.5 * FixedPointer('C') * a + 0.5 * FixedPointer('D')) * (
            0.5 * b + a * 0.5)
        p = nengo.Probe(x.construct(), synapse=0.3)

    with nengo.Simulator(model) as sim:
        sim.run(0.5)

    assert sp_close(
        sim.trange(),
        sim.data[p] / np.maximum(
            1e-10, np.linalg.norm(sim.data[p], axis=1, keepdims=True)),
        vocab.parse(
            '(0.5 * C * A + 0.5 * D) * (0.5 * B + 0.5 * A)').normalized(),
        skip=0.3)


# transform
# transform and fixed pointer
# network and transform
# transform and transform
# scalar with FixedPointer
# assignment
# sums get collapsed
# identity and zero
