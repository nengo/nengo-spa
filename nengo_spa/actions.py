try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence
import weakref

from nengo.exceptions import NetworkContextError
from nengo.utils.compat import is_integer

from nengo_spa.ast import ActionSet, ConstructionContext
from nengo_spa.exceptions import SpaConstructionError
from nengo_spa.network import Network


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

    def _attr_list(self, attr):
        objs = []
        for act_set in self.ast:
            if hasattr(act_set, attr):
                objs.append(getattr(act_set, attr))
        return objs

    def all_bgs(self):
        return self._attr_list("bg")

    def all_thals(self):
        return self._attr_list("thalamus")


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
            for i, rule in enumerate(actions):
                rule.index = i
            actions = [ActionSet(actions)]

        for block in actions:
            block.infer_types(None)
        construction_context = ConstructionContext(root_network)
        for block in actions:
            block.construct(construction_context)
        self.blocks += actions

    def add_effect(self, effect):
        self.effects.append(effect)

    def add_rule(self, rule):
        self.effects = [
            e for e in self.effects if e not in rule.effects.effects]
        self.rules.append(rule)

    def __enter__(self):
        if Actions.context is not None and Actions.context() is not None:
            raise SpaConstructionError("Nesting of Action contexts is not "
                                       "supported")
        Actions.context = weakref.ref(self)

        return self

    def __exit__(self, *args):
        Actions.context = None

        if args[0] is not None:
            return

        if len(self.rules) > 0:
            self.add_block(self.rules, mode="bg")
            self.rules = []

        if len(self.effects) > 0:
            self.add_block(self.effects, mode="cortical")
            self.effects = []
