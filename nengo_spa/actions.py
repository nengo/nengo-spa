"""Parsing of SPA actions."""

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

from nengo.exceptions import NetworkContextError
from nengo.network import Network as NengoNetwork
from nengo.utils.compat import is_integer

from nengo_spa.compiler.ast import build_ast, ConstructionContext
from nengo_spa.compiler.grammar import parse, Terminal
from nengo_spa.compiler.parsing import RuleMismatchError
from nengo_spa.compiler import tokens


class AstAccessor(Sequence):
    """Provides access to the root AST nodes of build action rules.

    Nodes can either be accessed by their ordinal position in the action rules
    or by the name provided in the action rules.
    """
    def __init__(self, ast):
        self.ast = ast
        self.by_name = {
            node.name: node for node in ast if hasattr(node, 'name')}

    def __len__(self):
        return len(self.ast)

    def __getitem__(self, key):
        if is_integer(key):
            return self.ast[key]
        else:
            return self.by_name[key]

    def __contains__(self, key):
        return key in self.ast or key in self.by_name

    def keys(self):
        return self.by_name.keys()


def Actions(actions, stacklevel=1):
    """Compiles action rules in a SPA network.

    Arguments
    ---------
    actions : str
        Action rules to build.
    stacklevel : int, optional
        Defines the stack level that names in the action rules refer to. For
        example, a stack level of 1 will use the scope of the calling function.

    Returns
    -------
    AstAccessor
        Provides access to the abstract syntax tree (AST) of the actions by
        integer indices corresponding to the order of the actions and by string
        keys if names where gives in the actions. The AST nodes provide access
        to the built Nengo objects.
    """

    if len(NengoNetwork.context) <= 0:
        raise NetworkContextError(
            "actions can only be called inside a ``with network:`` block.")
    root_network = NengoNetwork.context[-1]

    try:
        parse_tree = parse(actions)
    except RuleMismatchError as err:
        raise _rule_mismatch_err_to_syntax_err(actions, err)
    except tokens.TokenError:
        raise SyntaxError("unexpected EOF while parsing actions")

    ast = build_ast(parse_tree, stacklevel + 1)

    for top_node in ast:
        top_node.infer_types(None)
    construction_context = ConstructionContext(root_network)
    for top_node in ast:
        top_node.construct(construction_context)

    return AstAccessor(ast)


def _rule_mismatch_err_to_syntax_err(actions, err):
    err_type = SyntaxError
    pos = err.found[2]
    line = None
    if pos is not None:
        try:
            line = actions.split('\n')[pos[0] - 1]
        except IndexError:
            pass

    if err.found[0] == tokens.ERRORTOKEN:
        pos = (pos[0], len(line))
        msg = "EOL while scanning string literal"
    elif err.expected == Terminal(type=tokens.INDENT):
        msg = "expected an indented block"
        err_type = IndentationError
    elif err.found[0] == tokens.INDENT:
        msg = 'unexpected indent'
        err_type = IndentationError
    else:
        if err.found[0] == tokens.ENDMARKER and pos is not None:
            line = actions.split('\n')[-1]
            pos = (pos[0] - 1, len(line))

        msg = "expected {}, but found {!r}".format(err.expected, err.found[1])

    if pos is None:
        return err_type(msg)
    else:
        marker = pos[1] * ' ' + '^'
        return err_type("{} {}\n{}\n{}".format(pos, msg, line, marker))
