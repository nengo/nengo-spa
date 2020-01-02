import pytest

import nengo_spa as spa
from nengo_spa.exceptions import SpaParseError, SpaTypeError


def test_missing_source():
    d = 16
    with spa.Network():
        a = spa.State(d)
        assert a
        with pytest.raises(NameError):
            b >> a  # noqa: F821


@pytest.mark.parametrize("sink", ("b", "B"))
def test_missing_sink(sink):
    d = 16
    with spa.Network():
        a = spa.State(d)
        assert a
        with pytest.raises(NameError):
            eval("a >> %s" % sink)


@pytest.mark.parametrize("sink", ("0.5", "0.6 * a", "a * 0.6"))
def test_invalid_sink(sink):
    d = 16
    with spa.Network():
        a = spa.State(d)
        assert a
        with pytest.raises((SyntaxError, SpaTypeError)):
            eval("a >> {}".format(sink))


def test_missing_pointer():
    vocab = spa.Vocabulary(16)
    with spa.Network():
        a = spa.State(vocab)
        assert a
        with pytest.raises(SpaParseError):
            spa.sym.A >> a
