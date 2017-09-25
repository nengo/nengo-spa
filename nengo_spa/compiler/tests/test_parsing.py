from io import BytesIO

import pytest

from nengo_spa.internal.generators import Peekable
from nengo_spa.compiler import parsing
from nengo_spa.compiler.parsing import MatchToken
import nengo_spa.compiler.tokens as tk


encoding_terminal = parsing.Terminal(type=tk.ENCODING, string='utf-8')


def peekable_tokens(code):
    tokens = Peekable(tk.tokenize(BytesIO(code.encode()).readline))
    if encoding_terminal.accept(tokens):
        encoding_terminal.read(tokens)
    return tokens


def test_terminal():
    terminal = parsing.Terminal(type=tk.NAME, string='foobar')
    tokens = peekable_tokens('foobar')
    assert terminal.accept(tokens)
    assert terminal.read(tokens) == [
        MatchToken(type=tk.NAME, string='foobar')]

    assert not terminal.accept(tokens)
    with pytest.raises(parsing.RuleMismatchError):
        terminal.read(tokens)

    assert parsing.Terminal(string='foo') == parsing.Terminal(string='foo')
    assert parsing.Terminal(string='foo') != parsing.Terminal(string='bar')


@pytest.mark.parametrize('rule', [
    parsing.Either(
        parsing.Terminal(type=tk.NAME, string='foo'),
        parsing.Terminal(type=tk.NAME, string='bar')),
    parsing.Terminal(type=tk.NAME, string='foo') |
    parsing.Terminal(type=tk.NAME, string='bar')])
def test_either(rule):
    tokens = peekable_tokens('foo bar')
    assert rule.accept(tokens)
    assert rule.read(tokens) == [
        MatchToken(type=tk.NAME, string='foo')]

    assert rule.accept(tokens)
    assert rule.read(tokens) == [
        MatchToken(type=tk.NAME, string='bar')]

    assert not rule.accept(tokens)
    with pytest.raises(parsing.RuleMismatchError):
        rule.read(tokens)

    assert rule == parsing.Either(
        parsing.Terminal(type=tk.NAME, string='foo'),
        parsing.Terminal(type=tk.NAME, string='bar'))
    assert rule != parsing.Either(
        parsing.Terminal(type=tk.NAME, string='foo'))


def test_either_chaining_gathers_rules_in_single_object():
    rules = [parsing.Terminal() for _ in range(3)]
    either = rules[0] | rules[1] | rules[2]
    assert all(r1 is r2 for r1, r2 in zip(either.rules, rules))


@pytest.mark.parametrize('rule', [
    parsing.Chain(
        parsing.Terminal(type=tk.NAME, string='foo'),
        parsing.Terminal(type=tk.NAME, string='bar')),
    parsing.Terminal(type=tk.NAME, string='foo') +
    parsing.Terminal(type=tk.NAME, string='bar')])
def test_chain(rule):
    tokens = peekable_tokens('foo bar')
    assert rule.accept(tokens)
    assert rule.read(tokens) == [
        MatchToken(type=tk.NAME, string='foo'),
        MatchToken(type=tk.NAME, string='bar')]

    assert not rule.accept(tokens)
    with pytest.raises(parsing.RuleMismatchError):
        rule.read(tokens)

    assert rule == parsing.Chain(
        parsing.Terminal(type=tk.NAME, string='foo'),
        parsing.Terminal(type=tk.NAME, string='bar'))
    assert rule != parsing.Chain(
        parsing.Terminal(type=tk.NAME, string='foo'))


def test_chaining_gathers_rules_in_single_object():
    rules = [parsing.Terminal() for _ in range(3)]
    chain = rules[0] + rules[1] + rules[2]
    assert all(r1 is r2 for r1, r2 in zip(chain.rules, rules))


def test_maybe():
    maybe = parsing.Maybe(parsing.Terminal(type=tk.NAME, string='foobar'))
    tokens = peekable_tokens('foobar')
    assert maybe.accept(tokens)
    assert maybe.read(tokens) == [MatchToken(type=tk.NAME, string='foobar')]

    assert maybe.accept(tokens)
    assert maybe.read(tokens) == []

    assert maybe == parsing.Maybe(
        parsing.Terminal(type=tk.NAME, string='foobar'))
    assert maybe != parsing.Maybe(
        parsing.Terminal(type=tk.NAME, string='foo'))


@pytest.mark.parametrize('code', ['foobar', 'foo bar'])
def test_at_least_one(code):
    tokens = peekable_tokens(code)

    at_least_one = parsing.AtLeastOne(parsing.Terminal(type=tk.NAME))
    assert at_least_one.accept(tokens)
    parse = at_least_one.read(tokens)
    expected = code.split(' ')
    assert [x[1] for x in parse] == expected

    assert not at_least_one.accept(tokens)
    with pytest.raises(parsing.RuleMismatchError):
        at_least_one.read(tokens)

    assert at_least_one == parsing.AtLeastOne(parsing.Terminal(type=tk.NAME))
    assert at_least_one != parsing.AtLeastOne(parsing.Terminal(type=tk.STRING))


@pytest.mark.parametrize('code', ['', 'foobar', 'foo bar'])
def test_any_number(code):
    tokens = peekable_tokens(code)

    any_number = parsing.AnyNumber(parsing.Terminal(type=tk.NAME))
    assert any_number.accept(tokens)
    parse = any_number.read(tokens)
    expected = code.split(' ')
    if expected == ['']:
        expected = []
    assert [x[1] for x in parse] == expected

    assert any_number == parsing.AnyNumber(parsing.Terminal(type=tk.NAME))
    assert any_number != parsing.AnyNumber(parsing.Terminal(type=tk.STRING))


def test_group():
    tokens = peekable_tokens('foo bar')
    grammar = parsing.Group('parent', parsing.Group(
        'child', parsing.AnyNumber(parsing.Terminal(type=tk.NAME))))

    assert grammar.accept(tokens)
    parse = grammar.read(tokens)
    # we expect to get one 'parent' group
    assert len(parse) == 1
    parse = parse[0]
    assert parse[0] == 'parent'
    # the parent group has one 'child' with two tokens
    assert parse[1][0][0] == 'child'
    assert len(parse[1][0][1]) == 2

    # testing the equality operator
    assert grammar == parsing.Group('parent', parsing.Group(
        'child', parsing.AnyNumber(parsing.Terminal(type=tk.NAME))))
    assert grammar != parsing.Group('foo', parsing.Group(
        'child', parsing.AnyNumber(parsing.Terminal(type=tk.NAME))))


def test_peek():
    tokens = peekable_tokens('foo')
    grammar = parsing.Peek(parsing.Terminal(type=tk.NAME))

    assert grammar.accept(tokens)
    assert grammar.read(tokens) == []
    assert grammar.accept(tokens)
    assert grammar.read(tokens) == []

    assert grammar == parsing.Peek(parsing.Terminal(type=tk.NAME))
    assert grammar != parsing.Peek(parsing.Terminal(type=tk.STRING))
