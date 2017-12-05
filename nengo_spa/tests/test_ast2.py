from numpy.testing import assert_equal
import pytest

import nengo_spa as spa
from nengo_spa.ast2 import FixedPointer


def test_fixed_pointer_network_creation(rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A')

    with spa.Network() as model:
        A = FixedPointer('A')
        node = A.construct(vocab)
    assert_equal(node.output, vocab['A'].v)


@pytest.mark.parametrize('op', ['-', '~'])
def test_unary_operations_on_fixed_pointer(op, rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A')

    with spa.Network() as model:
        x = eval(op + "FixedPointer('A')")
        node = x.construct(vocab)
    assert_equal(node.output, vocab.parse(op + 'A').v)


@pytest.mark.parametrize('op', ['+', '-', '*'])
def test_binary_operation_on_fixed_pointers(op, rng):
    vocab = spa.Vocabulary(16, rng=rng)
    vocab.populate('A; B')

    with spa.Network() as model:
        x = eval("FixedPointer('A')" + op + "FixedPointer('B')")
        node = x.construct(vocab)
    assert_equal(node.output, vocab.parse('A' + op + 'B').v)


# transform
# transform and fixed pointer
# network
# network and transform
# network and fixed pointer
# assignment
