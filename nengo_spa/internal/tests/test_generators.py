import pytest

from nengo_spa.internal.generators import Peekable


def test_peekable():
    gen = Peekable(iter(range(5)), maxlen=2)
    assert gen.peek(n=2) == [0, 1]
    assert next(gen) == 0
    assert gen.peek() == [1]
    assert next(gen) == 1
    assert next(gen) == 2
    assert gen.peek() == [3]
    with pytest.raises(AssertionError):
        gen.peek(n=3)
    assert next(gen) == 3
    assert gen.peek(n=2) == [4]
    assert next(gen) == 4
    assert gen.peek() == []
    with pytest.raises(StopIteration):
        next(gen)


def test_peekable_no_maxlen():
    gen = Peekable(iter(range(5)), maxlen=None)
    assert gen.peek(n=20) == [0, 1, 2, 3, 4]
    for i in range(5):
        assert next(gen) == i
    with pytest.raises(StopIteration):
        next(gen)
