"""Definition of the Nengo SPA action rule grammar.

All interaction with this module should use the `parse` function. The exact
rules defined in this module may change.
"""

from io import BytesIO
import sys

from nengo_spa.compiler.parsing import (
    AnyNumber, AtLeastOne, Group, Maybe, Peek, Rule, RuleMismatchError,
    Terminal)
import nengo_spa.compiler.tokens as tk
from nengo_spa.internal.generators import Peekable


def parse(code):
    """Parses a block of code with the Nengo SPA action rule grammar.

    Arguments
    ---------
    code : str
        Code to parse.

    Returns
    -------
    list
        Represents the parse tree. Each non-terminal node is either represented
        by a list of child nodes or a 2-tuple *(node_type, children)* with a
        string *node_type* specifying the node type and a list of *children*.
        The terminal nodes are `tokenize.TokenInfo` objects.
    """
    return actions.read(Peekable(tk.tokenize(
        BytesIO(code.encode()).readline)))


effect_line_delimiter = Terminal(type=tk.NEWLINE) | Terminal(type=tk.ENDMARKER)
if sys.version_info[0] < 3:
    class ConnectL2R(Rule):
        """Rule accepting the -> operator.

        Consists out of two tokens in Python 2 and thus needs a special
        implementation with a look-ahead of 2.
        """
        def accept(self, tokens):
            return [t[1] for t in tokens.peek(n=2)] == ['-', '>']

        def read(self, tokens):
            try:
                return [next(tokens), next(tokens)]
            except StopIteration:
                raise RuleMismatchError(self, self, (tk.ENDMARKER, '', None))

        def __str__(self):
            return "'->'"

        def __repr__(self):
            return "ConnectL2R()"

    connect_l2r = ConnectL2R()
else:
    # In Python 3 '->' is actually an operator (for type annotations)
    connect_l2r = Terminal(type=tk.OP, string='->')


class _ExprRule(Rule):
    """Accepts an expression.

    An expression is regarded as a sequence of tokens that can be converted
    into a string that can be passed to the Python `eval` function.
    """

    DELIMETERS = (
        connect_l2r, Terminal(string=';'), effect_line_delimiter,
        Terminal(type=tk.INDENT), Terminal(type=tk.DEDENT),
        Terminal(string=':'), Terminal(type=tk.NAME, string='as'))
    NESTING = {'{': '}', '(': ')', '[': ']'}

    def accept(self, tokens):
        return len(tokens.peek()) > 0 and not any(
            delim.accept(tokens) for delim in self.DELIMETERS)

    def read(self, tokens):
        expr_tokens = []
        stack = []
        while len(stack) > 0 or not any(
                delim.accept(tokens) for delim in self.DELIMETERS):
            expr_tokens.append(next(tokens))
            tp = expr_tokens[-1][1]
            if tp in self.NESTING.keys():
                stack.append(self.NESTING[tp])
            elif len(stack) > 0 and stack[-1] == tp:
                stack.pop()
        return [('expr', expr_tokens)]

    def __str__(self):
        return 'expr'


expr = _ExprRule()
sink = Group('sink', Terminal(type=tk.NAME) + AnyNumber(
    Terminal(string='.') + Terminal(type=tk.NAME)))

nl = Terminal(type=tk.NEWLINE) + AnyNumber(Terminal(type=tk.NL))

effect = (
    Terminal(type=tk.NAME, string='pass') |
    Group('effect', expr + connect_l2r + sink))
effect_line = (
    effect +
    AnyNumber(Terminal(string=';') + effect) +
    Maybe(Terminal(type=tk.COMMENT)) +
    Peek(nl | Terminal(type=tk.ENDMARKER) | Terminal(type=tk.DEDENT)) +
    Maybe(nl))
effect_lines = AtLeastOne(effect_line)

block = (
    Terminal(type=tk.INDENT) + effect_lines +
    Terminal(type=tk.DEDENT))

name = Group(
    'as', Terminal(type=tk.NAME, string='as') + Terminal(type=tk.STRING))

nameable = (
    Maybe(name) +
    Terminal(string=':') +
    ((nl + block) | effect_line))

ifmax = Group(
    'ifmax',
    Terminal(type=tk.NAME, string='ifmax') + Group('utility', expr) + nameable)
elifmax = Group(
    'elifmax',
    Terminal(type=tk.NAME, string='elifmax') + Group('utility', expr) +
    nameable)
max_action = Group('max_action', ifmax + AnyNumber(elifmax))
always_action = Group(
    'always', Terminal(type=tk.NAME, string='always') + nameable)
keyword_action = always_action | max_action
action = keyword_action | effect_line


class MainBlock(Rule):
    """Accepts the whole indented block of action rules.

    Does not accept potential leading encoding or newline tokens.
    """

    def accept(self, tokens):
        return Terminal(type=tk.INDENT).accept(tokens) or action.accept(tokens)

    def read(self, tokens):
        match = []
        if Terminal(type=tk.INDENT).accept(tokens):
            # Main block is a block of multiple lines of Python
            match.extend(Terminal(type=tk.INDENT).read(tokens))
            while action.accept(tokens):
                match.extend(action.read(tokens))
                if Terminal(type=tk.NEWLINE).accept(tokens):
                    Terminal(type=tk.NEWLINE).read(tokens)
            match.extend(Terminal(type=tk.DEDENT).read(tokens))
        else:
            # Main block is just a single line of Python without indentation
            match = AtLeastOne(action).read(tokens)
        return match

    def __str__(self):
        return (
            str(Terminal(type=tk.INDENT)) + '<self>' +
            str(Terminal(type=tk.DEDENT))) | str(AtLeastOne(action))


if sys.version_info[0] < 3:
    actions = (
        AnyNumber(Terminal(type=tk.NL) | Terminal(type=tk.COMMENT)) +
        MainBlock() + Terminal(type=tk.ENDMARKER))
else:
    actions = (
        Terminal(type=tk.ENCODING) +
        AnyNumber(Terminal(type=tk.NL) | Terminal(type=tk.COMMENT)) +
        MainBlock() + Terminal(type=tk.ENDMARKER))
