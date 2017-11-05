from io import BytesIO

import pytest

import nengo_spa as spa
from nengo_spa.exceptions import SpaTypeError
from nengo_spa.compiler import tokens
from nengo_spa.compiler.ast import AstBuilder
from nengo_spa.compiler.ast_nodes import (
    Action, ActionSet, ApproxInverse, DotProduct, Effect, Effects, Negative,
    Module, Product, Scalar, Sink, Sum, Symbol)
from nengo_spa.compiler.ast_types import (
    TActionSet, TEffect, TEffects, TScalar, TVocabulary)


def bare_tokens(expr):
    """Tokenizes *expr* without tokens for encoding and endmarker.

    Helpful for constructing parts of parse trees in the tests.
    """
    return [
        tk for tk in tokens.tokenize(BytesIO(expr.encode()).readline)
        if tk[0] != tokens.ENCODING and tk[0] != tokens.ENDMARKER]


def test_scalar():
    ast = AstBuilder().build_expr(bare_tokens('1'))
    assert ast == [1]


def test_symbol():
    ast = AstBuilder().build_expr(bare_tokens('A'))
    assert ast == [Symbol('A')]
    node = ast[0]
    assert str(node) == 'A'

    vocab_type = TVocabulary(spa.Vocabulary(16, strict=False))
    node.infer_types(vocab_type)
    assert node.type == vocab_type

    ast = AstBuilder().build_expr(bare_tokens('A'))
    with pytest.raises(SpaTypeError):
        ast[0].infer_types(TScalar)


def test_spa_network():
    d = 16
    with spa.Network():
        state = spa.State(d)

    ast = AstBuilder().build_expr(bare_tokens('state'))
    assert ast == [Module('state', state)]
    assert str(ast[0]) == 'state'
    ast[0].infer_types(None)
    assert ast[0].type.vocab == state.vocabs[d]

    with spa.Network():
        with spa.Network() as network:
            network.state = spa.State(d)

    ast = AstBuilder().build_expr(bare_tokens('network.state'))
    assert ast == [Module('network.state', network.state)]
    assert str(ast[0]) == 'network.state'
    ast[0].infer_types(None)
    assert ast[0].type.vocab == network.state.vocabs[d]


def test_scalar_multiplication():
    ast = AstBuilder().build_expr(bare_tokens('2 * A'))
    assert ast == [Product(2, Symbol('A'))]
    assert str(ast[0]) == '2 * A'

    vocab_type = TVocabulary(spa.Vocabulary(16, strict=False))
    ast[0].infer_types(vocab_type)
    assert ast[0].type == vocab_type

    ast = AstBuilder().build_expr(bare_tokens('A * 2'))
    assert ast[0] == Product(Symbol('A'), 2)
    assert str(ast[0]) == 'A * 2'

    vocab_type = TVocabulary(spa.Vocabulary(16, strict=False))
    ast[0].infer_types(vocab_type)
    assert ast[0].type == vocab_type

    d = 16
    with spa.Network():
        state = spa.State(d)

    ast = AstBuilder().build_expr(bare_tokens('2 * state'))
    assert ast == [Product(2, Module('state', state))]
    assert str(ast[0]) == '2 * state'

    ast[0].infer_types(None)
    assert ast[0].type.vocab == state.vocabs[d]

    ast = AstBuilder().build_expr(bare_tokens('state * 2'))
    assert ast == [Product(Module('state', state), 2)]
    assert str(ast[0]) == 'state * 2'

    ast[0].infer_types(None)
    assert ast[0].type.vocab == state.vocabs[d]


@pytest.mark.parametrize('symbol, klass', [('+', Sum), ('*', Product)])
def test_binary_operations(symbol, klass):
    ast = AstBuilder().build_expr(bare_tokens('A {} B'.format(symbol)))
    assert ast == [klass(Symbol('A'), Symbol('B'))]
    assert str(ast[0]) == 'A {} B'.format(symbol)

    vocab_type = TVocabulary(spa.Vocabulary(16, strict=False))
    ast[0].infer_types(vocab_type)
    assert ast[0].type == vocab_type

    d = 16
    with spa.Network():
        state = spa.State(d)
        state2 = spa.State(d)

    ast = AstBuilder().build_expr(bare_tokens('state {} B'.format(symbol)))
    assert ast == [klass(Module('state', state), Symbol('B'))]
    assert str(ast[0]) == 'state {} B'.format(symbol)

    ast[0].infer_types(TVocabulary(state.vocabs[d]))
    assert ast[0].type.vocab == state.vocabs[d]

    ast = AstBuilder().build_expr(bare_tokens('A {} state'.format(symbol)))
    assert ast == [klass(Symbol('A'), Module('state', state))]
    assert str(ast[0]) == 'A {} state'.format(symbol)

    ast[0].infer_types(TVocabulary(state.vocabs[d]))
    assert ast[0].type.vocab == state.vocabs[d]

    ast = AstBuilder().build_expr(bare_tokens(
        'state {} state2'.format(symbol)))
    assert ast == [klass(Module('state', state), Module('state2', state2))]
    assert str(ast[0]) == 'state {} state2'.format(symbol)

    ast[0].infer_types(None)
    assert ast[0].type.vocab == state.vocabs[d]


@pytest.mark.parametrize('symbol, klass', [
    ('~', ApproxInverse), ('-', Negative)])
def test_unary(symbol, klass):
    ast = AstBuilder().build_expr(bare_tokens(symbol + 'A'))
    assert ast == [klass(Symbol('A'))]
    assert str(ast[0]) == symbol + 'A'

    vocab_type = TVocabulary(spa.Vocabulary(16, strict=False))
    ast[0].infer_types(vocab_type)
    assert ast[0].type == vocab_type

    d = 16
    with spa.Network():
        state = spa.State(d)

    ast = AstBuilder().build_expr(bare_tokens(symbol + 'state'))
    assert ast == [klass(Module('state', state))]
    assert str(ast[0]) == symbol + 'state'

    ast[0].infer_types(None)
    assert ast[0].type.vocab == state.vocabs[d]


def test_dot_product():
    d = 16
    with spa.Network():
        state = spa.State(d)

    ast = AstBuilder().build_expr(bare_tokens('dot(A, state)'))
    assert ast == [DotProduct(Symbol('A'), Module('state', state))]
    assert str(ast[0]) == 'dot(A, state)'

    ast[0].infer_types(TScalar)
    assert ast[0].type == TScalar

    ast = AstBuilder().build_expr(bare_tokens('2 * dot(A, state) + 1'))
    assert ast == [Sum(Product(2, DotProduct(Symbol('A'), Module(
        'state', state))), 1)]
    assert str(ast[0]) == '2 * dot(A, state) + 1'

    ast[0].infer_types(TScalar)
    assert ast[0].type == TScalar


def test_effect():
    d = 16
    with spa.Network() as model:
        model.state = spa.State(d)

    ast = AstBuilder().build_effect([
        ('expr', bare_tokens('A')),
        ('sink', bare_tokens('model.state')),
    ])
    assert ast == [Effect(Sink('model.state', model.state), Symbol('A'))]
    assert str(ast[0]) == 'A -> model.state'
    assert ast[0].type == TEffect


def test_effects():
    with spa.Network() as model:
        model.config[spa.State].vocab = 16
        a = spa.State()
        b = spa.State()
        x = spa.State()
        y = spa.Scalar()
        z = spa.State()

    # Check that multiple lvalue -> rvalue parsing is working with semicolon
    ast = AstBuilder().build(
        [('effect', [
            ('expr', bare_tokens('a')),
            ('sink', bare_tokens('x')),
        ])] +
        bare_tokens(';') +
        [('effect', [
            ('expr', bare_tokens('dot(a,b)')),
            ('sink', bare_tokens('y')),
        ])] +
        bare_tokens('\n') +
        [('effect', [
            ('expr', bare_tokens('b')),
            ('sink', bare_tokens('z')),
        ])]
    )
    assert ast == [
        Effect(Sink('x', x), Module('a', a)),
        Effect(Sink('y', y), DotProduct(Module('a', a), Module('b', b))),
        Effect(Sink('z', z), Module('b', b))]
    assert str(ast[0]) == 'a -> x'
    assert str(ast[1]) == 'dot(a, b) -> y'
    assert str(ast[2]) == 'b -> z'
    assert all(node.type == TEffect for node in ast)


def test_always():
    with spa.Network() as model:
        model.config[spa.State].vocab = 16
        a = spa.State()
        b = spa.State()
        x = spa.State()
        y = spa.Scalar()

    ast = AstBuilder().build([(
        'always',
        [('effect', [
            ('expr', bare_tokens('a')),
            ('sink', bare_tokens('x')),
        ])] +
        bare_tokens(';') +
        [('effect', [
            ('expr', bare_tokens('dot(a,b)')),
            ('sink', bare_tokens('y')),
        ])]
    )])
    assert ast == [Effects([
        Effect(Sink('x', x), Module('a', a)),
        Effect(Sink('y', y), DotProduct(Module('a', a), Module('b', b)))])]
    assert str(ast[0]) == 'a -> x\ndot(a, b) -> y'
    assert ast[0].type == TEffects


def test_max_action():
    with spa.Network() as model:
        model.config[spa.State].vocab = 16
        a = spa.State()
        b = spa.State()
        x = spa.State()
        y = spa.Scalar()

    ast = AstBuilder().build([
        ('max_action', [
            ('ifmax', [
                ('utility', [('expr', bare_tokens('dot(a, b)'))]),
                ('effect', [
                    ('expr', bare_tokens('a')),
                    ('sink', bare_tokens('x')),
                ]),
            ]),
            ('elifmax', [
                ('utility', [('expr', bare_tokens('0.5'))]),
                ('effect', [
                    ('expr', bare_tokens('dot(a,b)')),
                    ('sink', bare_tokens('y')),
                ]),
            ]),
        ])
    ])
    assert ast == [ActionSet([
        Action(
            DotProduct(Module('a', a), Module('b', b)),
            Effects([Effect(Sink('x', x), Module('a', a), channeled=True)]),
            index=0),
        Action(
            Scalar(0.5),
            Effects([Effect(
                Sink('y', y),
                DotProduct(Module('a', a), Module('b', b)), channeled=True)]),
            index=1)
        ])
    ]
    assert str(ast[0]) == '''ifmax dot(a, b):
    a -> x
elifmax 0.5:
    dot(a, b) -> y
'''
    assert ast[0].type == TActionSet


def test_always_name():
    with spa.Network() as model:
        model.config[spa.State].vocab = 16
        a = spa.State()
        b = spa.State()

    ast = AstBuilder().build(
        [('always', [
            ('as', bare_tokens("as 'name'")),
            ('effect', [
                ('expr', bare_tokens('a')),
                ('sink', bare_tokens('b')),
            ])
        ])]
    )
    assert ast == [Effects([
        Effect(Sink('b', b), Module('a', a)),
        ], name='name')]
    assert str(ast[0]) == '''always as 'name':
    a -> b'''


def test_max_action_name():
    with spa.Network() as model:
        model.config[spa.State].vocab = 16
        a = spa.State()
        b = spa.State()
        x = spa.State()
        y = spa.Scalar()

    ast = AstBuilder().build([
        ('max_action', [
            ('ifmax', [
                ('utility', [('expr', bare_tokens('dot(a, b)'))]),
                ('as', bare_tokens("as 'name1'")),
                ('effect', [
                    ('expr', bare_tokens('a')),
                    ('sink', bare_tokens('x')),
                ]),
            ]),
            ('elifmax', [
                ('utility', [('expr', bare_tokens('0.5'))]),
                ('as', bare_tokens("as 'name2'")),
                ('effect', [
                    ('expr', bare_tokens('dot(a,b)')),
                    ('sink', bare_tokens('y')),
                ]),
            ]),
        ])
    ])
    assert ast == [ActionSet([
        Action(
            DotProduct(Module('a', a), Module('b', b)),
            Effects([Effect(Sink('x', x), Module('a', a), channeled=True)]),
            index=0, name='name1'),
        Action(
            Scalar(0.5),
            Effects([Effect(
                Sink('y', y),
                DotProduct(Module('a', a), Module('b', b)), channeled=True)]),
            index=1, name='name2')
        ])
    ]
    assert str(ast[0]) == '''ifmax dot(a, b) as 'name1':
    a -> x
elifmax 0.5 as 'name2':
    dot(a, b) -> y
'''
    assert ast[0].type == TActionSet


def test_complex_epressions():
    d = 16
    with spa.Network() as m:
        m.state = spa.State(d)
        m.a = spa.State(d)
        m.b = spa.State(d)
        m.x = spa.State(d)

    ast = AstBuilder().build_expr(bare_tokens('~(A - B * m.state)'))
    assert ast == [ApproxInverse(Sum(Symbol('A'), Negative(Product(
        Symbol('B'), Module('m.state', m.state)))))]
    assert str(ast[0]) == '~(A + -(B * m.state))'

    ast[0].infer_types(TVocabulary(m.state.vocabs[d]))
    assert ast[0].type.vocab == m.state.vocabs[d]

    ast = AstBuilder().build_expr(bare_tokens(
        '0.5*(2*dot(m.a, A)-dot(m.b, B))-2'))
    assert str(ast[0]) == '0.5 * (2 * dot(m.a, A) + -dot(m.b, B)) + -2'

    ast = AstBuilder().build_expr(bare_tokens('dot(m.x, -1) + 1'))
    assert str(ast[0]) == 'dot(m.x, -1) + 1'

    ast = AstBuilder().build_expr(bare_tokens(
        '2*dot(m.a, 1) - dot(m.b, -1) + dot(m.a, m.b)'))
    assert str(ast[0]) == '2 * dot(m.a, 1) + -dot(m.b, -1) + dot(m.a, m.b)'

    ast = AstBuilder().build_expr(bare_tokens('m.a*m.b - 1 + 2*m.b'))
    assert str(ast[0]) == 'm.a * m.b + -1 + 2 * m.b'


def test_zero_vector():
    d = 16
    with spa.Network() as model:
        model.state = spa.State(d)

    ast = AstBuilder().build_effect([
        ('expr', bare_tokens('0')),
        ('sink', bare_tokens('model.state')),
    ])
    ast[0].infer_types(None)
    assert ast[0].source.type.vocab == model.state.vocabs[d]


def test_vocab_transform_in_multiplication():
    d = 16
    with spa.Network() as model:
        model.state = spa.State(d)

    ast = AstBuilder().build_expr(bare_tokens('2 * translate(model.state)'))
    assert str(ast[0]) == '2 * translate(model.state)'

    ast = AstBuilder().build_expr(bare_tokens('translate(model.state) * 2'))
    assert str(ast[0]) == 'translate(model.state) * 2'

    ast = AstBuilder().build_expr(bare_tokens('2 * reinterpret(model.state)'))
    assert str(ast[0]) == '2 * reinterpret(model.state)'

    ast = AstBuilder().build_expr(bare_tokens('reinterpret(model.state) * 2'))
    assert str(ast[0]) == 'reinterpret(model.state) * 2'


def test_missing_operator():
    d = 16
    with spa.Network() as model:
        model.state = spa.State(d)

    with pytest.raises(SyntaxError):
        AstBuilder().build_expr(bare_tokens('A B'))
