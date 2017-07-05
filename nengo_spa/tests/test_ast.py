import pytest

import nengo_spa as spa
from nengo_spa.exceptions import SpaTypeError
from nengo_spa.actions import Parser
from nengo_spa.ast import (
    Action, ApproxInverse, DotProduct, Effect, Effects, Negative, Module,
    Product, Sink, Sum, Symbol, TAction, TEffect, TEffects, TScalar,
    TVocabulary)


def test_scalar():
    ast = Parser().parse_expr('1')
    assert ast == 1


def test_symbol():
    ast = Parser().parse_expr('A')
    assert ast == Symbol('A')
    assert str(ast) == 'A'

    vocab_type = TVocabulary(spa.Vocabulary(16, strict=False))
    ast.infer_types(None, vocab_type)
    assert ast.type == vocab_type

    ast = Parser().parse_expr('A')
    with pytest.raises(SpaTypeError):
        ast.infer_types(None, TScalar)


def test_spa_network():
    d = 16
    with spa.Network() as model:
        state = spa.State(d)

    ast = Parser().parse_expr('state')
    assert ast == Module('state', state)
    assert str(ast) == 'state'
    ast.infer_types(model, None)
    assert ast.type.vocab == state.vocabs[d]

    with spa.Network() as model:
        with spa.Network() as network:
            network.state = spa.State(d)

    ast = Parser().parse_expr('network.state')
    assert ast == Module('network.state', network.state)
    assert str(ast) == 'network.state'
    ast.infer_types(model, None)
    assert ast.type.vocab == network.state.vocabs[d]


def test_scalar_multiplication():
    ast = Parser().parse_expr('2 * A')
    assert ast == Product(2, Symbol('A'))
    assert str(ast) == '2 * A'

    vocab_type = TVocabulary(spa.Vocabulary(16, strict=False))
    ast.infer_types(None, vocab_type)
    assert ast.type == vocab_type

    ast = Parser().parse_expr('A * 2')
    assert ast == Product(Symbol('A'), 2)
    assert str(ast) == 'A * 2'

    vocab_type = TVocabulary(spa.Vocabulary(16, strict=False))
    ast.infer_types(None, vocab_type)
    assert ast.type == vocab_type

    d = 16
    with spa.Network() as model:
        state = spa.State(d)

    ast = Parser().parse_expr('2 * state')
    assert ast == Product(2, Module('state', state))
    assert str(ast) == '2 * state'

    ast.infer_types(model, None)
    assert ast.type.vocab == state.vocabs[d]

    ast = Parser().parse_expr('state * 2')
    assert ast == Product(Module('state', state), 2)
    assert str(ast) == 'state * 2'

    ast.infer_types(model, None)
    assert ast.type.vocab == state.vocabs[d]


@pytest.mark.parametrize('symbol, klass', [('+', Sum), ('*', Product)])
def test_binary_operations(symbol, klass):
    ast = Parser().parse_expr('A {} B'.format(symbol))
    assert ast == klass(Symbol('A'), Symbol('B'))
    assert str(ast) == 'A {} B'.format(symbol)

    vocab_type = TVocabulary(spa.Vocabulary(16, strict=False))
    ast.infer_types(None, vocab_type)
    assert ast.type == vocab_type

    d = 16
    with spa.Network() as model:
        state = spa.State(d)
        state2 = spa.State(d)

    ast = Parser().parse_expr('state {} B'.format(symbol))
    assert ast == klass(Module('state', state), Symbol('B'))
    assert str(ast) == 'state {} B'.format(symbol)

    ast.infer_types(model, TVocabulary(state.vocabs[d]))
    assert ast.type.vocab == state.vocabs[d]

    ast = Parser().parse_expr('A {} state'.format(symbol))
    assert ast == klass(Symbol('A'), Module('state', state))
    assert str(ast) == 'A {} state'.format(symbol)

    ast.infer_types(model, TVocabulary(state.vocabs[d]))
    assert ast.type.vocab == state.vocabs[d]

    ast = Parser().parse_expr('state {} state2'.format(symbol))
    assert ast == klass(Module('state', state), Module('state2', state2))
    assert str(ast) == 'state {} state2'.format(symbol)

    ast.infer_types(model, None)
    assert ast.type.vocab == state.vocabs[d]


@pytest.mark.parametrize('symbol, klass', [
    ('~', ApproxInverse), ('-', Negative)])
def test_unary(symbol, klass):
    ast = Parser().parse_expr(symbol + 'A')
    assert ast == klass(Symbol('A'))
    assert str(ast) == symbol + 'A'

    vocab_type = TVocabulary(spa.Vocabulary(16, strict=False))
    ast.infer_types(None, vocab_type)
    assert ast.type == vocab_type

    d = 16
    with spa.Network() as model:
        state = spa.State(d)

    ast = Parser().parse_expr(symbol + 'state')
    assert ast == klass(Module('state', state))
    assert str(ast) == symbol + 'state'

    ast.infer_types(model, None)
    assert ast.type.vocab == state.vocabs[d]


def test_dot_product():
    d = 16
    with spa.Network() as model:
        state = spa.State(d)

    ast = Parser().parse_expr('dot(A, state)')
    assert ast == DotProduct(Symbol('A'), Module('state', state))
    assert str(ast) == 'dot(A, state)'

    ast.infer_types(model, TScalar)
    assert ast.type == TScalar

    ast = Parser().parse_expr('2 * dot(A, state) + 1')
    assert ast == Sum(Product(2, DotProduct(Symbol('A'), Module(
        'state', state))), 1)
    assert str(ast) == '2 * dot(A, state) + 1'

    ast.infer_types(model, TScalar)
    assert ast.type == TScalar


def test_effect():
    d = 16
    with spa.Network() as model:
        model.state = spa.State(d)

    ast = Parser().parse_effect('model.state = A')
    assert ast == Effect(Sink('model.state', model.state), Symbol('A'))
    assert str(ast) == 'model.state = A'
    assert ast.type == TEffect


def test_effects():
    with spa.Network() as model:
        model.config[spa.State].vocab = 16
        a = spa.State()
        b = spa.State()
        x = spa.State()
        y = spa.Scalar()
        z = spa.State()
        foo = spa.Scalar()
        bar = spa.State()

    # Check that multiple lvalue=rvalue parsing is working with commas
    ast = Parser().parse_effects('x=a,y=dot(a,b),z=b')
    assert ast == Effects(
        Effect(Sink('x', x), Module('a', a)),
        Effect(Sink('y', y), DotProduct(Module('a', a), Module('b', b))),
        Effect(Sink('z', z), Module('b', b)))
    assert str(ast) == 'x = a, y = dot(a, b), z = b'
    assert ast.type == TEffects

    ast = Parser().parse_effects('  foo = dot(a, b)  , bar = b')
    assert ast == Effects(
        Effect(Sink('foo', foo), DotProduct(Module('a', a), Module('b', b))),
        Effect(Sink('bar', bar), Module('b', b)))
    assert str(ast) == 'foo = dot(a, b), bar = b'


def test_action():
    d = 16
    with spa.Network() as model:
        model.state = spa.State(d)

    ast = Parser().parse_action('dot(model.state, A) --> model.state = B')
    assert ast == Action(
        DotProduct(Module('model.state', model.state), Symbol('A')),
        Effects(Effect(
            Sink('model.state', model.state), Symbol('B'),
            channeled=True)))
    assert str(ast) == 'dot(model.state, A) --> model.state = B'
    assert ast.type == TAction


def test_complex_epressions():
    d = 16
    with spa.Network() as m:
        m.state = spa.State(d)
        m.a = spa.State(d)
        m.b = spa.State(d)
        m.x = spa.State(d)

    ast = Parser().parse_expr('~(A - B * m.state)')
    assert ast == ApproxInverse(Sum(Symbol('A'), Negative(Product(
        Symbol('B'), Module('m.state', m.state)))))
    assert str(ast) == '~(A + -(B * m.state))'

    ast.infer_types(m, TVocabulary(m.state.vocabs[d]))
    assert ast.type.vocab == m.state.vocabs[d]

    ast = Parser().parse_expr('0.5*(2*dot(m.a, A)-dot(m.b, B))-2')
    assert str(ast) == '0.5 * (2 * dot(m.a, A) + -dot(m.b, B)) + -2'

    ast = Parser().parse_expr('dot(m.x, -1) + 1')
    assert str(ast) == 'dot(m.x, -1) + 1'

    ast = Parser().parse_expr('2*dot(m.a, 1) - dot(m.b, -1) + dot(m.a, m.b)')
    assert str(ast) == '2 * dot(m.a, 1) + -dot(m.b, -1) + dot(m.a, m.b)'

    ast = Parser().parse_expr('m.a*m.b - 1 + 2*m.b')
    assert str(ast) == 'm.a * m.b + -1 + 2 * m.b'


def test_zero_vector():
    d = 16
    with spa.Network() as model:
        model.state = spa.State(d)

    ast = Parser().parse_effect('model.state = 0')
    ast.infer_types(model, None)
    assert ast.source.type.vocab == model.state.vocabs[d]


def test_vocab_transform_in_multiplication():
    d = 16
    with spa.Network() as model:
        model.state = spa.State(d)

    ast = Parser().parse_expr('2 * translate(model.state)')
    assert str(ast) == '2 * translate(model.state)'

    ast = Parser().parse_expr('translate(model.state) * 2')
    assert str(ast) == 'translate(model.state) * 2'

    ast = Parser().parse_expr('2 * reinterpret(model.state)')
    assert str(ast) == '2 * reinterpret(model.state)'

    ast = Parser().parse_expr('reinterpret(model.state) * 2')
    assert str(ast) == 'reinterpret(model.state) * 2'
