import pytest

import nengo_spa as spa
from nengo_spa.exceptions import SpaTypeError

from nengo_spa.ast import (
    Action, ActionSet, ApproxInverse, DotProduct, Effect, Effects, Negative,
    Module, Product, Scalar, Sum, Symbol, Sink, as_node)
from nengo_spa.types import (
    TActionSet, TEffect, TScalar, TVocabulary)


def test_scalar():
    assert as_node(1) == Scalar(1)


def test_symbol():
    node = as_node('A')
    assert node == Symbol('A')
    assert str(node) == 'A'

    vocab_type = TVocabulary(spa.Vocabulary(16, strict=False))
    node.infer_types(vocab_type)
    assert node.type == vocab_type

    ast = as_node('A')
    with pytest.raises(SpaTypeError):
        ast.infer_types(TScalar)


def test_spa_network():
    d = 16
    with spa.Network():
        state = spa.State(d)

    ast = as_node(state)
    assert ast == Module(str(state), state)
    assert str(ast) == str(state)
    ast.infer_types(None)
    assert ast.type.vocab == state.vocabs[d]

    with spa.Network():
        with spa.Network() as network:
            network.state = spa.State(d)

    ast = as_node(network.state)
    assert ast == Module(str(network.state), network.state)
    assert str(ast) == str(network.state)
    ast.infer_types(None)
    assert ast.type.vocab == network.state.vocabs[d]


def test_scalar_multiplication():
    ast = 2 * spa.sym("A")
    assert ast == Product(2, Symbol('A'))
    assert str(ast) == '2 * A'

    vocab_type = TVocabulary(spa.Vocabulary(16, strict=False))
    ast.infer_types(vocab_type)
    assert ast.type == vocab_type

    ast = spa.sym('A') * 2
    assert ast == Product(Symbol('A'), 2)
    assert str(ast) == 'A * 2'

    vocab_type = TVocabulary(spa.Vocabulary(16, strict=False))
    ast.infer_types(vocab_type)
    assert ast.type == vocab_type

    d = 16
    with spa.Network():
        state = spa.State(d)

    ast = 2 * as_node(state)
    assert ast == Product(2, Module(str(state), state))
    assert str(ast) == '2 * %s' % state

    ast.infer_types(None)
    assert ast.type.vocab == state.vocabs[d]

    ast = as_node(state) * 2
    assert ast == Product(Module(str(state), state), 2)
    assert str(ast) == '%s * 2' % state

    ast.infer_types(None)
    assert ast.type.vocab == state.vocabs[d]


@pytest.mark.parametrize('symbol, klass', [('+', Sum), ('*', Product)])
def test_binary_operations(symbol, klass):
    ast = klass('A', 'B')
    assert ast == klass(Symbol('A'), Symbol('B'))
    assert str(ast) == 'A {} B'.format(symbol)

    vocab_type = TVocabulary(spa.Vocabulary(16, strict=False))
    ast.infer_types(vocab_type)
    assert ast.type == vocab_type

    d = 16
    with spa.Network():
        state = spa.State(d)
        state2 = spa.State(d)

    ast = klass(state, 'B')
    assert ast == klass(Module(str(state), state), Symbol('B'))
    assert str(ast) == '{} {} B'.format(state, symbol)

    ast.infer_types(TVocabulary(state.vocabs[d]))
    assert ast.type.vocab == state.vocabs[d]

    ast = klass('A', state)
    assert ast == klass(Symbol('A'), Module(str(state), state))
    assert str(ast) == 'A {} {}'.format(symbol, state)

    ast.infer_types(TVocabulary(state.vocabs[d]))
    assert ast.type.vocab == state.vocabs[d]

    ast = klass(state, state2)
    assert ast == klass(Module(str(state), state), Module(str(state2), state2))
    assert str(ast) == '{} {} {}'.format(state, symbol, state2)

    ast.infer_types(None)
    assert ast.type.vocab == state.vocabs[d]


@pytest.mark.parametrize('symbol, klass', [
    ('~', ApproxInverse), ('-', Negative)])
def test_unary(symbol, klass):
    ast = klass('A')
    assert ast == klass(Symbol('A'))
    assert str(ast) == symbol + 'A'

    vocab_type = TVocabulary(spa.Vocabulary(16, strict=False))
    ast.infer_types(vocab_type)
    assert ast.type == vocab_type

    d = 16
    with spa.Network():
        state = spa.State(d)

    ast = klass(state)
    assert ast == klass(Module(str(state), state))
    assert str(ast) == symbol + str(state)

    ast.infer_types(None)
    assert ast.type.vocab == state.vocabs[d]


def test_dot_product():
    d = 16
    with spa.Network():
        state = spa.State(d)

    ast = spa.dot('A', state)
    assert ast == DotProduct(Symbol('A'), Module(str(state), state))
    assert str(ast) == 'dot(A, {})'.format(state)

    ast.infer_types(TScalar)
    assert ast.type == TScalar

    ast = 2 * spa.dot('A', state) + 1
    assert ast == Sum(Product(2, DotProduct(Symbol('A'), Module(
        str(state), state))), 1)
    assert str(ast) == '2 * dot(A, {}) + 1'.format(state)

    ast.infer_types(TScalar)
    assert ast.type == TScalar


def test_effect():
    d = 16
    with spa.Network() as model:
        model.state = spa.State(d)

    ast = spa.route('A', model.state)
    assert ast == Effect(Sink(str(model.state), model.state), Symbol('A'))
    assert str(ast) == 'A -> {}'.format(model.state)
    assert ast.type == TEffect


def test_effects():
    with spa.Network() as model:
        model.config[spa.State].vocab = 16
        a = spa.State()
        b = spa.State()
        x = spa.State()
        y = spa.Scalar()
        z = spa.State()

    # Check that multiple lvalue -> rvalue parsing is working with semicolon
    ast = [spa.route(a, x), spa.dot(a, b) >> y, as_node(b) >> z]
    assert ast == [
        Effect(Sink(str(x), x), Module(str(a), a)),
        Effect(Sink(str(y), y), DotProduct(Module(str(a), a),
                                           Module(str(b), b))),
        Effect(Sink(str(z), z), Module(str(b), b))]
    assert str(ast[0]) == '{} -> {}'.format(a, x)
    assert str(ast[1]) == 'dot({}, {}) -> {}'.format(a, b, y)
    assert str(ast[2]) == '{} -> {}'.format(b, z)
    assert all(node.type == TEffect for node in ast)


# def test_always():
#     with spa.Network() as model:
#         model.config[spa.State].vocab = 16
#         a = spa.State()
#         b = spa.State()
#         x = spa.State()
#         y = spa.Scalar()
#
#     ast = AstBuilder().build([(
#         'always',
#         [('effect', [
#             ('expr', bare_tokens('a')),
#             ('sink', bare_tokens('x')),
#         ])] +
#         bare_tokens(';') +
#         [('effect', [
#             ('expr', bare_tokens('dot(a,b)')),
#             ('sink', bare_tokens('y')),
#         ])]
#     )])
#     assert ast == [Effects([
#         Effect(Sink('x', x), Module('a', a)),
#         Effect(Sink('y', y), DotProduct(Module('a', a), Module('b', b)))])]
#     assert str(ast[0]) == 'a -> x\ndot(a, b) -> y'
#     assert ast[0].type == TEffects


def test_max_action():
    with spa.Network() as model:
        model.config[spa.State].vocab = 16
        a = spa.State(label="a")
        b = spa.State(label="b")
        x = spa.State(label="x")
        y = spa.Scalar(label="y")

    ast = ActionSet([
        spa.ifmax(spa.dot(a, b), spa.route(a, x)),
        spa.ifmax(0.5, spa.route(spa.dot(a, b), y))])

    assert ast == ActionSet([
        Action(
            DotProduct(Module(str(a), a), Module(str(b), b)),
            Effects((Effect(Sink(str(x), x), Module(str(a), a),
                            channeled=True),)),
            index=0),
        Action(
            Scalar(0.5),
            Effects((Effect(
                Sink(str(y), y),
                DotProduct(Module(str(a), a), Module(str(b), b)),
                channeled=True),)),
            index=0)
    ])
    assert str(ast) == '''ifmax dot({a}, {b}):
    {a} -> {x}
elifmax 0.5:
    dot({a}, {b}) -> {y}
'''.format(a=str(a), b=str(b), x=str(x), y=str(y))
    assert ast.type == TActionSet


# def test_always_name():
#     with spa.Network() as model:
#         model.config[spa.State].vocab = 16
#         a = spa.State()
#         b = spa.State()
#
#     ast = AstBuilder().build(
#         [('always', [
#             ('as', bare_tokens("as 'name'")),
#             ('effect', [
#                 ('expr', bare_tokens('a')),
#                 ('sink', bare_tokens('b')),
#             ])
#         ])]
#     )
#     assert ast == [Effects([
#         Effect(Sink('b', b), Module('a', a)),
#         ], name='name')]
#     assert str(ast[0]) == '''always as 'name':
#     a -> b'''


# def test_max_action_name():
#     with spa.Network() as model:
#         model.config[spa.State].vocab = 16
#         a = spa.State()
#         b = spa.State()
#         x = spa.State()
#         y = spa.Scalar()
#
#     ast = AstBuilder().build([
#         ('max_action', [
#             ('ifmax', [
#                 ('utility', [('expr', bare_tokens('dot(a, b)'))]),
#                 ('as', bare_tokens("as 'name1'")),
#                 ('effect', [
#                     ('expr', bare_tokens('a')),
#                     ('sink', bare_tokens('x')),
#                 ]),
#             ]),
#             ('elifmax', [
#                 ('utility', [('expr', bare_tokens('0.5'))]),
#                 ('as', bare_tokens("as 'name2'")),
#                 ('effect', [
#                     ('expr', bare_tokens('dot(a,b)')),
#                     ('sink', bare_tokens('y')),
#                 ]),
#             ]),
#         ])
#     ])
#     assert ast == [ActionSet([
#         Action(
#             DotProduct(Module('a', a), Module('b', b)),
#             Effects([Effect(Sink('x', x), Module('a', a), channeled=True)]),
#             index=0, name='name1'),
#         Action(
#             Scalar(0.5),
#             Effects([Effect(
#                 Sink('y', y),
#                 DotProduct(Module('a', a), Module('b', b)), channeled=True)]),
#             index=1, name='name2')
#         ])
#     ]
#     assert str(ast[0]) == '''ifmax dot(a, b) as 'name1':
#     a -> x
# elifmax 0.5 as 'name2':
#     dot(a, b) -> y
# '''
#     assert ast[0].type == TActionSet


def test_complex_epressions():
    d = 16
    with spa.Network() as m:
        m.state = spa.State(d)
        m.a = spa.State(d)
        m.b = spa.State(d)
        m.x = spa.State(d)

    ast = ~('A' - 'B' * as_node(m.state))
    assert ast == ApproxInverse(Sum(Symbol('A'), Negative(Product(
        Symbol('B'), Module(str(m.state), m.state)))))
    assert str(ast) == '~(A + -(B * {}))'.format(m.state)

    ast.infer_types(TVocabulary(m.state.vocabs[d]))
    assert ast.type.vocab == m.state.vocabs[d]

    ast = 0.5 * (2 * spa.dot(m.a, "A") - spa.dot(m.b, "B")) - 2
    assert str(ast) == '0.5 * (2 * dot({}, A) + -dot({}, B)) + -2'.format(
        m.a, m.b)

    ast = spa.dot(m.x, -1) + 1
    assert str(ast) == 'dot({}, -1) + 1'.format(m.x)

    ast = 2 * spa.dot(m.a, 1) - spa.dot(m.b, -1) + spa.dot(m.a, m.b)
    assert str(ast) == '2 * dot({a}, 1) + -dot({b}, -1) + ' \
                       'dot({a}, {b})'.format(
        a=m.a, b=m.b)

    ast = as_node(m.a) * m.b - 1 + 2 * as_node(m.b)
    assert str(ast) == '{a} * {b} + -1 + 2 * {b}'.format(a=m.a, b=m.b)


def test_zero_vector():
    d = 16
    with spa.Network() as model:
        model.state = spa.State(d)

    ast = spa.route("0", model.state)
    ast.infer_types(None)
    assert ast.source.type.vocab == model.state.vocabs[d]


def test_vocab_transform_in_multiplication():
    d = 16
    with spa.Network() as model:
        model.state = spa.State(d)

    ast = 2 * spa.translate(model.state)
    assert str(ast) == '2 * translate({})'.format(model.state)

    ast = spa.translate(model.state) * 2
    assert str(ast) == 'translate({}) * 2'.format(model.state)

    ast = 2 * spa.reinterpret(model.state)
    assert str(ast) == '2 * reinterpret({})'.format(model.state)

    ast = spa.reinterpret(model.state) * 2
    assert str(ast) == 'reinterpret({}) * 2'.format(model.state)
