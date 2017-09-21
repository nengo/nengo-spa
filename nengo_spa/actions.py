"""Parsing of SPA actions."""

import inspect
from itertools import chain

from nengo.exceptions import NetworkContextError
from nengo.network import Network as NengoNetwork
from nengo.utils.compat import is_integer

from nengo_spa.compiler.ast import ConstructionContext
from nengo_spa.compiler.ast_nodes import (
    Action, DotProduct, Effect, Effects, Module,
    Reinterpret, Sink, Symbol, Translate)
from nengo_spa.exceptions import SpaNameError, SpaParseError
from nengo_spa.modules.basalganglia import BasalGanglia
from nengo_spa.modules.thalamus import Thalamus
from nengo_spa.network import Network as SpaNetwork


class Parser(object):
    """Parser for SPA actions.

    Parameters
    ----------
    locals_ : int or dict, optional
        Dictionary of locals to resolve names during parsing. When given an
        integer instead of a dictionary, the locals dictionary that many frames
        up the stack will be used. For example, when given ``1``, the locals
        dictionary of the calling function will be used.
    """

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

    def parse_action(self, action, index=0, name=None, strict=True):
        """Parse an SPA action.

        Parameters
        ----------
        action : str
            Action to parse.
        index : int, optional
            Index of the action for identification by basal ganglia and
            thalamus.
        name : str, optional
            Name of the action.
        strict : bool, optional
            If ``True`` only actual conditional actions are allowed and an
            exception will be raised for anything else. If ``False``, allows
            also the parsing of effects without the conditional part.

        Returns
        -------
        :class:`spa_ast.Action` or :class:`spa_ast.Effects`
        """
        try:
            condition, effects = action.split('-->', 1)
        except ValueError:
            if strict:
                raise SpaParseError("Not an action, '-->' missing.")
            else:
                return self.parse_effects(action, channeled=False)
        else:
            return Action(
                self.parse_expr(condition),
                self.parse_effects(effects, channeled=True), index, name=name)

    def parse_effects(self, effects, channeled=False):  # noqa: C901
        """Pares SPA effects.

        Parameters
        ----------
        effects : str
            Effects to parse.
        channeled : bool, optional
            Whether the effects should be passed through channels when
            constructed.

        Returns
        -------
        :class:`spa_ast.Effects`
        """
        parsed = []
        symbol_stack = []
        start = 0
        for i, c in enumerate(effects):
            top = symbol_stack[-1] if len(symbol_stack) > 0 else None
            if top == '\\':  # escaped character, ignore
                symbol_stack.pop()
            elif top is not None and top in '\'"':  # in a string
                if c == '\\':  # escape
                    symbol_stack.append(c)
                elif c == top:  # end string
                    symbol_stack.pop()
            else:
                if c in '\'"':  # start string
                    symbol_stack.append(c)
                elif c in '([':  # start parens/brackets
                    symbol_stack.append(c)
                elif c in ')]':  # end parens/brackets
                    if (top == '(' and c != ')') or (top == '[' and c != ']'):
                        raise SpaParseError("Parenthesis mismatch.")
                    symbol_stack.pop()
                elif c == ',' and len(symbol_stack) == 0:  # effect delimiter
                    parsed.append(effects[start:i])
                    start = i + 1
        parsed.append(effects[start:])

        if len(symbol_stack) != 0:
            top = symbol_stack.pop()
            if top in '([':
                raise SpaParseError("Parenthesis mismatch.")
            elif top in '\'"':
                raise SpaParseError("Unclosed string.")
            else:
                raise SpaParseError("Unmatched: " + top)

        return Effects([self.parse_effect(effect, channeled=channeled)
                        for effect in parsed])

    def parse_effect(self, effect, channeled=False):
        """Parse single SPA effect.

        Parameters
        ----------
        effect : str
            Effect to parse.
        channeled : bool, optional
            Whether the effect should be passed through a channel when
            constructed.

        Returns
        -------
        :class:`spa_ast.Effect`
        """
        try:
            sink, source = effect.split('=', 1)
        except ValueError:
            raise SpaParseError("Not an effect; assignment missing")

        sink = sink.strip()
        try:
            obj = eval(sink, dict(self.locals), globals())
        except (AttributeError, NameError):
            raise SpaNameError(sink, 'network input')
        return Effect(
            Sink(sink, obj), self.parse_expr(source), channeled=channeled)

    def parse_expr(self, expr):
        """Parse an SPA expression.

        Parameters
        ----------
        expr : str
            Expression to parse.

        Returns
        -------
        :class:`spa_ast.Source`
        """
        return eval(expr, {}, self)

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
            raise SpaNameError(key, 'name')

        if isinstance(item, SpaNetwork):
            item = Module(key, item)
        return item


class Actions(object):
    """A collection of Action objects.

    The *args and **kwargs are treated as unnamed and named Actions,
    respectively.

    The keyword argument `vocabs` is special in that it provides a dictionary
    mapping names to vocabularies. The vocabularies can then be used with those
    names in the action rules.
    """

    def __init__(
            self, actions=None, named_actions=None, stacklevel=1, build=True):
        super(Actions, self).__init__()

        if actions is None:
            actions = []
        if named_actions is None:
            named_actions = {}

        self.actions = []
        self.named_actions = {}

        self.bg = None
        self.thalamus = None
        self.connstructed = None

        self.construction_context = None

        self.parse(actions, named_actions, stacklevel=stacklevel + 1)
        if build:
            self.bg, self.thalamus, self.connstructed = self.build()

    def parse(self, actions, named_actions, stacklevel=1):
        named_actions = sorted(named_actions.items())

        parser = Parser(stacklevel=stacklevel + 1)
        for action in actions:
            self._parse_and_add(parser, action)
        for name, action in named_actions:
            self._parse_and_add(parser, action, name=name)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, key):
        if is_integer(key):
            return self.actions[key]
        else:
            return self.named_actions[key]

    def _parse_and_add(self, parser, action, name=None):
        ast = parser.parse_action(
            action, len(self.actions), strict=False, name=name)
        self.actions.append(ast)
        if name is not None:
            self.named_actions[name] = ast

    @property
    def bg_actions(self):
        return [a for a in chain(self.actions, self.named_actions.values())
                if isinstance(a, Action)]

    def build(self):
        needs_bg = len(self.bg_actions) > 0

        if len(NengoNetwork.context) <= 0:
            raise NetworkContextError(
                "Actions.build can only be called inside a ``with network:`` "
                "block.")
        root_network = NengoNetwork.context[-1]

        with root_network:
            if needs_bg:
                bg = BasalGanglia(action_count=len(self.actions))
                thalamus = Thalamus(action_count=len(self.actions))
                for i, a in enumerate(self.actions):
                    thalamus.actions.ensembles[i].label = (
                        'action[{}]: {}'.format(i, a.effects))
                thalamus.connect_bg(bg)
            else:
                bg = thalamus = None

        self.construction_context = ConstructionContext(
            root_network, bg=bg, thalamus=thalamus)
        with root_network:
            for action in self.actions:
                action.infer_types(root_network, None)
            # Infer types for all actions before doing any construction, so
            # that all semantic pointers are added to the respective
            # vocabularies so that the translate transform are identical.
            for action in self.actions:
                action.construct(self.construction_context)

        return bg, thalamus, self.actions
