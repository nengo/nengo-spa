"""Abstract syntax trees for SPA actions.

SPA actions are first parsed into a parse tree which is then converted into an
abstract syntax tree (AST) with the `AstBuilder` which allows to infer types of
individual nodes in the tree and to construct the required SPA modules for
implementing it in a Nengo network.

For the construction of the AST of basic expressions the Python eval function is
used (by modifiying the name lookup in the globals dictionary). Because of this
all Python code not involving identifiers will be statically evaluated before
insertion into the syntax tree (e.g., '2 * 3 + 1' will be inserted as
``Scalar(7)``).

Each node in the syntax tree will evaluate to a specific type. The most
important types are ``TScalar`` for expressions evaluating to a scalar and
:class:`TVocabulary` for expressions evaluating to a semantic pointer. This
latter type describes the vocabulary the semantic pointer belongs to and
different vocabularies give different types. This ensures that only semantic
pointers of a matching vocabulary are assigned.

To determine the type of each node the complete AST is required. This is
because names of semantic pointers are not associated with a vocabulary and it
needs to be inferred from some actual SPA network for which we have to be able
to resolve the names of those networks. There are a few basic rules for this
type inference:

1. If something with unknown vocabulary is assigned to a network, that
   network's vocabulary provides the type.
2. If a binary operation has an operand with unknown vocabulary it is
   determined from the other operand.

Once all the types have been determined, the AST can be used to construct
Nengo objects to perform the operations represented with the AST. In this
process each node in the syntax tree can create :class:`Artifact`s. These
give generated Nengo objects to be connected to the appropriate places
including the transform that should be used on the connection. This is
necessary because at the time most objects are constructed we only know this
constructed object and the transform, but not what it is supposed to connect
to. So the final connection will be done by some other node in the syntax tree.

To avoid confusion with the normal Nengo build process, we use the term
'construct' here. (Also, we use 'build' to refer to building the AST based on
the parse tree.)

Note that `Difference` ``a - b`` will be represented as `a + (-b)` in the AST.

Operator precedence is defined as follows from highest to lowest priority:

``
0 Scalar, Symbol, Zero, One, Module, DotProduct, Reinterpret, Translate
1 UnaryOperation
2 Product
3 Sum
``
"""

import inspect
import re

from nengo_spa.compiler import tokens
from nengo_spa.compiler.ast_nodes import (
    Action, ActionSet, DotProduct, Effect, Effects, Module, Reinterpret, Sink,
    Symbol, Translate)
from nengo_spa.compiler.parsing import MatchToken
from nengo_spa.network import Network as SpaNetwork


def build_ast(parse_tree, stacklevel=1):
    """Build the AST from given parse tree.

    Arguments
    ---------
    parse_tree : list
        Parse tree obtained with `nengo_spa.compiler.grammar.parse`.
    stacklevel : int
        What stack level the names in the parsed code refer to.

    Returns
    -------
    list
        List with root-level nodes in the AST.
    """
    return AstBuilder(ActionsScope(stacklevel + 1)).build(parse_tree)


class ConstructionContext(object):
    """Context in which SPA actions are constructed.

    This primarily provides the SPA networks used to construct certain
    components. All attributes except `root_network` may be ``None`` if these
    are not provided in the current construction context.

    Attributes
    ----------
    root_network : :class:`spa.Module`
        The root network the encapsulated all of the constructed structures.
    bg : :class:`spa.BasalGanglia`
        Module to manage the basal ganglia part of action selection.
    thalamus : :class:`spa.Thalamus`
        Module to manage the thalamus part of action selection.
    sink : :class:`Sink`
        Node in the AST where some result will be send to.
    active_net : class:`nengo.Network`
        Network to add constructed components to.
    """
    __slots__ = [
        'root_network', 'bg', 'thalamus', 'bias', 'sink', 'active_net']

    def __init__(
            self, root_network, bg=None, thalamus=None,
            sink=None, active_net=None):
        self.root_network = root_network
        self.bg = bg
        self.thalamus = thalamus
        self.bias = None
        self.sink = sink
        if active_net is None:
            active_net = root_network
        self.active_net = active_net

    def subcontext(self, bg=None, thalamus=None, sink=None, active_net=None):
        """Creates a subcontext.

        All omitted arguments will be initialized from the parent context.
        """
        if bg is None:
            bg = self.bg
        if thalamus is None:
            thalamus = self.thalamus
        if sink is None:
            sink = self.sink
        if active_net is None:
            active_net = self.active_net
        return self.__class__(
            root_network=self.root_network, bg=bg,
            thalamus=thalamus, sink=sink, active_net=active_net)

    @property
    def sink_network(self):
        return SpaNetwork.get_input_network(self.sink.obj)

    @property
    def sink_input(self):
        return self.sink.obj, SpaNetwork.get_input_vocab(self.sink.obj)


class ActionsScope(object):
    """Captures the scope that names in action rules refer to."""

    builtins = {
        'dot': DotProduct,
        'reinterpret': Reinterpret,
        'translate': Translate}

    def __init__(self, stacklevel=1):
        frame = inspect.currentframe()
        for _ in range(stacklevel):
            frame = frame.f_back
        self.locals = frame.f_locals
        self.globals = frame.f_globals
        self.py_builtins = frame.f_builtins

    def __getitem__(self, key):
        if key == '__tracebackhide__':  # gives better tracebacks in py.test
            item = False
        elif key in self.builtins:
            item = self.builtins[key]
        elif key in self.locals:
            item = self.locals[key]
        elif key in self.globals:
            item = self.globals[key]
        elif key in self.py_builtins:
            item = self.py_builtins[key]
        elif key[0].isupper():
            item = Symbol(key)
        else:
            raise KeyError(key)

        if isinstance(item, SpaNetwork):
            item = Module(key, item)
        return item


class AstBuilder(object):
    """Builds the AST from a parse tree.

    Parameters
    ----------
    scope : ActionsScope, optional
        Used to resolve names in the action rules. If not given, it defaults to
        the scope of the calling function.
    """

    def __init__(self, scope=None):
        if scope is None:
            scope = ActionsScope(stacklevel=2)
        self.scope = scope
        self._channeled = False
        self._encoding = None
        self._name = None
        self._index = 0

    def build(self, parse_tree):
        """Builds a parse tree (or subtree) into an AST.

        Returns
        -------
        list
            List of built top-level AST nodes.
        """
        ast_nodes = []
        for child in parse_tree:
            if tokens.is_token_info(child):
                ast_nodes.extend(self.build_token(child))
            else:
                build_fn_name = 'build_' + child[0]
                if hasattr(self, build_fn_name):
                    ast_nodes.extend(getattr(self, build_fn_name)(child[1]))
                else:
                    ast_nodes.extend(self.build(child[1]))
        return ast_nodes

    def build_token(self, token):
        if token[0] == tokens.ENCODING:
            self._encoding = token.string
        return []

    def build_expr(self, parse_tree):
        expr = self._untokenize_expr(parse_tree)
        return [eval(expr, {}, self.scope)]

    def build_sink(self, parse_tree):
        expr = self._untokenize_expr(parse_tree)
        obj = eval(expr, dict(self.scope.locals), dict(self.scope.globals))
        return [Sink(expr.strip(), obj)]

    def _untokenize_expr(self, parse_tree):
        """Converts a sequence of tokens back into Python code.

        Parameters
        ----------
        parse_tree : sequence
            A list of tokens (a leaf in the parse tree).

        Returns
        -------
        str
            String of Python code constructed from the tokens.
        """
        expr = tokens.untokenize(
            ([] if self._encoding is None else [
                (tokens.ENCODING, self._encoding)]) +
            parse_tree +
            [(tokens.ENDMARKER, '')])
        if self._encoding is not None:
            expr = expr.decode(self._encoding)
        expr = re.sub(r'\\?\s+', ' ', expr)
        expr = re.sub(r'\s*\.\s*', '.', expr)
        return expr

    def build_effect(self, parse_tree):
        try:
            source, sink = self.build(parse_tree)
        except ValueError:
            raise ValueError("Invalid parse tree")
        return [Effect(sink, source, channeled=self._channeled)]

    def build_as(self, parse_tree):
        assert parse_tree == [
            MatchToken(type=tokens.NAME, string='as'),
            MatchToken(type=tokens.STRING)]
        self._name = eval(parse_tree[1][1])  # eval to remove quotes
        return []

    def build_always(self, parse_tree):
        self._name = None
        return [Effects(self.build(parse_tree), name=self._name)]

    def build_max_action(self, parse_tree):
        self._index = 0
        return [ActionSet(self.build(parse_tree))]

    def build_ifmax(self, parse_tree):
        return self.build_max_choice(parse_tree)

    def build_elifmax(self, parse_tree):
        return self.build_max_choice(parse_tree)

    def build_max_choice(self, parse_tree):
        try:
            self._channeled = True
            self._name = None
            children = self.build(parse_tree)
            if len(children) <= 0:
                raise ValueError("Invalid parse tree.")
            ast_node = Action(
                children[0], Effects(children[1:]),
                self._index, name=self._name)
            self._index += 1
            self._channeled = False
            return [ast_node]
        finally:
            self._channeled = False
