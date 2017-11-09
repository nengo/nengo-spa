from nengo.network import Network
from nengo.exceptions import NetworkContextError
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


def route(a, b):
    return nodes.Effect(to_node(b, as_sink=True), to_node(a))


def reinterpret(a):
    return nodes.Reinterpret(a)


def translate(a):
    return nodes.Translate(a)


class Actions(object):
    def __init__(self, *blocks):
        self.blocks = []
        self.data = AstAccessor(self.blocks)
        for b in blocks:
            if not isinstance(b, list):
                b = [b]
            self.add_block(*b)

    def add_block(self, *actions):
        if len(Network.context) <= 0:
            raise NetworkContextError(
                "actions can only be called inside a ``with network:`` block.")
        root_network = Network.context[-1]

        if isinstance(actions[0], tuple):
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
