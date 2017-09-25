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
                always
                    a -> b
            ''')
    assert str(excinfo.value) == '''(2, 22) expected ':', but found '\\n'
                always
                      ^'''


def test_non_string_name():
    d = 16
    with spa.Network():
        a = spa.State(d)
        b = spa.State(d)
        assert a
        assert b
        with pytest.raises(SyntaxError) as excinfo:
            spa.Actions('''
                always as foo:
                    a -> b
            ''')
    assert str(excinfo.value) == '''(2, 26) expected STRING, but found 'foo'
                always as foo:
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
                always as 'foo:
                    pass
            ''')
    assert str(excinfo.value) == '''(2, 31) EOL while scanning string literal
                always as 'foo:
                               ^'''


def test_noindent():
    d = 16
    with spa.Network():
        a = spa.State(d)
        assert a
        with pytest.raises(IndentationError) as excinfo:
            spa.Actions('''
                always:
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
                always:
                    pass
                        pass
            ''')
    assert str(excinfo.value) == '''(4, 0) unexpected indent
                        pass
^'''
