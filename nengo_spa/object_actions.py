import weakref

from nengo.network import Network
from nengo.exceptions import NetworkContextError
from nengo_spa import Vocabulary
from nengo_spa.exceptions import SpaConstructionError
from nengo_spa.network import Network as SPANetwork
from nengo_spa.actions import AstAccessor
from nengo_spa.compiler import ast_nodes as nodes
from nengo_spa.compiler.ast import ConstructionContext, ActionsScope


def to_node(obj, as_sink=False):
    if isinstance(obj, nodes.Node):
        return obj
    elif obj is None or obj == "0":
        return nodes.Zero()
    elif obj == "1":
        return nodes.One()
    elif isinstance(obj, str):
        return eval(obj, {}, ActionsScope(stacklevel=2))
    elif isinstance(obj, (int, float)):
        return nodes.Scalar(obj)
    else:
        name = str(obj)
        return (nodes.Sink(name, obj) if as_sink else
                nodes.Module(name, obj))


action_ops = {
    "__invert__": lambda x: inv(x),
    "__neg__": lambda x: neg(x),
    "__add__": lambda x, y: add(x, y),
    "__radd__": lambda x, y: add(x, y),
    "__sub__": lambda x, y: add(x, neg(y)),
    "__rsub__": lambda x, y: add(y, neg(x)),
    "__mul__": lambda x, y: bind(x, y),
    "__rmul__": lambda x, y: bind(x, y),
    "__rshift__": lambda x, y: route(x, y),
    "__rrshift__": lambda x, y: route(y, x),
}
saved_network_ops = {k: getattr(SPANetwork, k, None) for k in action_ops}


def dot(a, b):
    return nodes.DotProduct(to_node(a), to_node(b))


def bind(a, b):
    return nodes.Product(to_node(a), to_node(b))


def add(a, b):
    return nodes.Sum(to_node(a), to_node(b))


def inv(a):
    return nodes.ApproxInverse(to_node(a))


def neg(a):
    return nodes.Negative(to_node(a))


def reinterpret(a, vocab=None):
    if vocab is not None and not isinstance(vocab, Vocabulary):
        vocab = to_node(vocab)
    return nodes.Reinterpret(to_node(a), vocab=vocab)


def translate(a, vocab=None, populate=None, solver=None):
    if vocab is not None and not isinstance(vocab, Vocabulary):
        vocab = to_node(vocab)
    return nodes.Translate(to_node(a), vocab=vocab, populate=populate,
                           solver=solver)


def route(a, b):
    eff = nodes.Effect(to_node(b, as_sink=True), to_node(a))
    Actions.context.add_effect(eff)
    return eff


def cond(cond, *effects):
    Actions.context.add_rule(cond, effects)


class Actions(AstAccessor):
    context = None

    def __init__(self):
        self.blocks = []
        self.effects = []
        self.rules = []

        super(Actions, self).__init__(self.blocks)

    def add_block(self, actions, mode="bg"):
        if len(Network.context) <= 0:
            raise NetworkContextError(
                "actions can only be called inside a ``with network:`` block.")
        root_network = Network.context[-1]

        if mode == "bg":
            blocks = self._add_bg_block(actions)
        else:
            blocks = self._add_cortical_block(actions)

        for block in blocks:
            block.infer_types(None)
        construction_context = ConstructionContext(root_network)
        for block in blocks:
            block.construct(construction_context)
        self.blocks += blocks

    def _add_cortical_block(self, effects):
        return [to_node(e) for e in effects]

    def _add_bg_block(self, actions):
        action_nodes = []
        for i, (cond, effects) in enumerate(actions):
            if not isinstance(effects, (tuple, list)):
                effects = [effects]
            effects = nodes.Effects([to_node(e) for e in effects])
            for e in effects.effects:
                e.channeled = True
            action_nodes.append(
                nodes.Action(to_node(cond), effects, index=i))
        return [nodes.ActionSet(action_nodes)]

    def add_effect(self, effect):
        self.effects.append(effect)

    def add_rule(self, cond, effects):
        self.effects = [e for e in self.effects if e not in effects]
        self.rules.append((cond, effects))

    def __enter__(self):
        if Actions.context is not None:
            try:
                # TODO: is there a better way to check if it's alive?
                Actions.context.test
                raise SpaConstructionError("Nesting of Action contexts is not "
                                           "supported")
            except ReferenceError:
                # dead context object
                pass
        Actions.context = weakref.proxy(self)

        for k, v in action_ops.items():
            setattr(SPANetwork, k, v)
        return self

    def __exit__(self, *args):
        if args[0] is not None:
            return

        if len(self.rules) > 0:
            self.add_block(self.rules, mode="bg")
            self.rules = []

        if len(self.effects) > 0:
            self.add_block(self.effects, mode="cortical")
            self.effects = []

        Actions.context = None

        for k, v in saved_network_ops.items():
            if v is None:
                delattr(SPANetwork, k)
            else:
                setattr(SPANetwork, k, v)
