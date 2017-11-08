import pytest

import nengo_spa as spa
from nengo_spa.exceptions import SpaParseError


def test_missing_source():
    d = 16
    with spa.Network():
        a = spa.State(d)
        assert a
        with pytest.raises(NameError):
            spa.Actions('b -> a')


@pytest.mark.parametrize('sink', ('b', 'B'))
def test_missing_sink(sink):
    d = 16
    with spa.Network():
        a = spa.State(d)
        assert a
        with pytest.raises(NameError):
            spa.Actions('a -> ' + sink)


@pytest.mark.parametrize('sink', ('0.5', '0.6 * a', 'a * 0.6'))
def test_invalid_sink(sink):
    d = 16
    with spa.Network():
        a = spa.State(d)
        assert a
        with pytest.raises(SyntaxError):
            spa.Actions('a -> {}'.format(sink))


def test_missing_pointer():
    vocab = spa.Vocabulary(16)
    with spa.Network():
        a = spa.State(vocab)
        assert a
        with pytest.raises(SpaParseError):
            spa.Actions('A -> a')


def test_non_effect():
    d = 16
    with spa.Network():
        a = spa.State(d)
        assert a
        with pytest.raises(SyntaxError) as excinfo:
            spa.Actions('a')
    # not checking line number, it will not be included in Python 2.7
    assert "expected '->', but found ''" in str(excinfo.value)


def test_missing_colon():
    d = 16
    with spa.Network():
        a = spa.State(d)
        b = spa.State(d)
        assert a
        assert b
        with pytest.raises(SyntaxError) as excinfo:
            spa.Actions('''
                ifmax 0.5
                    a -> b
            ''')
    assert str(excinfo.value) == '''(2, 25) expected ':', but found '\\n'
                ifmax 0.5
                         ^'''


def test_unclosed_parens():
    d = 16
    with spa.Network():
        a = spa.State(d)
        assert a
        with pytest.raises(SyntaxError):
            spa.Actions('''
                ifmax dot(a, A:
                    pass
                elifmax 0.5:
                    pass
            ''')


def test_unclosed_single_line_str():
    with spa.Network():
        with pytest.raises(SyntaxError) as excinfo:
            spa.Actions('''
                ifmax fn('foo:
                    pass
            ''')
    assert str(excinfo.value) == 'unexpected EOF while parsing actions'


def test_noindent():
    d = 16
    with spa.Network():
        a = spa.State(d)
        assert a
        with pytest.raises(IndentationError) as excinfo:
            spa.Actions('''
                ifmax 0.5:
                pass
            ''')
    assert str(excinfo.value) == '''(3, 16) expected an indented block
                pass
                ^'''


def test_overindent():
    d = 16
    with spa.Network():
        a = spa.State(d)
        assert a
        with pytest.raises(IndentationError) as excinfo:
            spa.Actions('''
                ifmax 0.5:
                    pass
                        pass
            ''')
    assert str(excinfo.value) == '''(4, 0) unexpected indent
                        pass
^'''
