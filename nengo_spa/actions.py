"""Parsing of SPA actions."""

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

from nengo.exceptions import NetworkContextError
from nengo.network import Network as NengoNetwork
from nengo.utils.compat import is_integer

from nengo_spa.compiler.grammar import parse
from nengo_spa.compiler.ast import build_ast, ConstructionContext


class AstAccessor(Sequence):
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

    parse_tree = parse(actions)
    ast = build_ast(parse_tree, stacklevel + 1)

    for top_node in ast:
        top_node.infer_types(root_network, None)
    construction_context = ConstructionContext(root_network)
    for top_node in ast:
        top_node.construct(construction_context)

    return AstAccessor(ast)
