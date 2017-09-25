"""Classes to construct simple top-down parsers for non-ambiguous languages.

The classes are used to construct the rules of a context-free grammar. Each
class allows to test whether the next token(s) are accepted and to read the
tokens and produce a list of parse objects. The parsing is greedy in so far that
the first rule that accepts the upcoming tokens will be applied without any
backtracking should it fail later.
"""

from nengo_spa.compiler import tokens


class RuleMismatchError(Exception):
    """Raised when trying to apply a rule to tokens that do not match the rule.
    """
    def __init__(self, rule, expected, found):
        super(RuleMismatchError, self).__init__(
            "Could not apply rule '{}'. Expected '{}', but found '{}'.".format(
                rule, expected, found))
        self.rule = rule
        self.expected = expected
        self.found = found


class MatchToken(object):
    """Class to match `TokenInfo` objects.

    This class will be equal to any `TokenInfo` object that matches all given
    arguments with its attributes. It is allowed to have further arguments that
    are not required to match.

    Arguments
    ---------
    kwargs : dict
        Definition of attributes that need to match.
    """

    arg2idx = {arg: idx for idx, arg in enumerate(
        ('type', 'string', 'start', 'end', 'line'))}

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __eq__(self, other):
        return isinstance(other, tokens.TokenInfo) and all(
            self._match_arg(other, arg) for arg in self.kwargs)

    def _match_arg(self, other, arg):
        if hasattr(other, arg):
            return getattr(other, arg) == self.kwargs[arg]
        else:
            return other[self.arg2idx[arg]] == self.kwargs[arg]

    @classmethod
    def from_token(cls, token, match_on=None):
        """Creates an instance from given `TokenInfo` object.

        By default *type* and *string* need to match.

        Arguments
        ---------
        token : TokenInfo
            Token to create the instance from.
        match_on : sequence, optional
            Sequence of attributes that need to be matched.
        """
        if match_on is None:
            match_on = ['type', 'string']
        return cls(**{
            name: getattr(token, name)
            if hasattr(token, name) else token[cls.arg2idx[name]]
            for name in match_on})

    def __str__(self):
        return 'MatchToken({})'.format(
            ', '.join('{}={!r}'.format(k, v) for k, v in self.kwargs.items()))


class Rule(object):
    """Abstract base class for grammar rules.

    Rules can be chained with the addition operator ``+`` and alternatives can
    be given with the or operator ``|``. Note that the first alternative that
    matches the upcoming tokens will be applied without any form of
    backtracking. Thus, the order when listing alternatives may matter.
    """

    def accept(self, tokens):
        """Tests whether the upcoming tokens are accepted by the rule.

        Parameters
        ----------
        tokens : nengo.internal.generators.Peekable
            Peekable generator providing the upcoming tokens.

        Returns
        -------
        bool
        """
        raise NotImplementedError()

    def read(self, tokens):
        """Reads the tokens excepted by the rule and returns derived objects.

        Parameters
        ----------
        tokens : nengo.internal.generators.Peekable
            Peekable generator providing the upcoming tokens.

        Returns
        -------
        list
            List of read tokens, or a tuple of a node name and a list of read
            tokens or other tuples constituting a parse tree.
        """
        raise NotImplementedError()

    def __add__(self, other):
        return Chain(self, other)

    def __or__(self, other):
        return Either(self, other)


class Terminal(Rule):
    """Accepts a terminal symbol.

    Parameters
    ----------
    kwargs : dict
        Attributes that must be matched on a token to be accepted (i.e.
        considered to match the terminal symbol).
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def accept(self, tokens):
        tk = tokens.peek()
        if len(tk) > 0:
            tk = tk[0]
        else:
            return False
        return MatchToken(**self.kwargs) == tk

    def read(self, tokens):
        if not self.accept(tokens):
            raise RuleMismatchError(self, self, tokens.peek()[0])
        return [next(tokens)]

    def __eq__(self, other):
        return isinstance(other, Terminal) and self.kwargs == other.kwargs

    def __repr__(self):
        return 'Terminal({{{}}})'.format(', '.join(
            k + ': ' + str(v) for k, v in self.kwargs.items()))

    def __str__(self):
        if 'string' in self.kwargs:
            return repr(self.kwargs['string'])
        elif 'type' in self.kwargs:
            return str(tokens.tok_name[self.kwargs['type']])
        else:
            return repr(self)


class Chain(Rule):
    """Accepts rules in the given order.

    Parameters
    ----------
    rules : list
        List of rules that are accepted in exactly the given order.
    """
    def __init__(self, *rules):
        self.rules = rules

    def accept(self, tokens):
        return self.rules[0].accept(tokens)

    def read(self, tokens):
        return [x for rule in self.rules for x in rule.read(tokens)]

    def __add__(self, other):
        return Chain(*(self.rules + (other,)))

    def __eq__(self, other):
        return isinstance(other, Chain) and self.rules == other.rules

    def __repr__(self):
        return '(' + ' '.join(repr(r) for r in self.rules) + ')'

    def __str__(self):
        return '(' + ' '.join(str(r) for r in self.rules) + ')'


class Either(Rule):
    """Accepts any rule out of a list of rules.

    The first accepted rule will me used. No backtracking is employed.

    Parameters
    ----------
    rules : list
        Rules that are accepted.
    """

    def __init__(self, *rules):
        self.rules = rules

    def accept(self, tokens):
        return any(rule.accept(tokens) for rule in self.rules)

    def read(self, tokens):
        for rule in self.rules:
            if rule.accept(tokens):
                return rule.read(tokens)
        else:
            raise RuleMismatchError(self, self, tokens.peek()[0])

    def __or__(self, other):
        return Either(*(self.rules + (other,)))

    def __eq__(self, other):
        return isinstance(other, Either) and self.rules == other.rules

    def __repr__(self):
        return '(' + ' | '.join(repr(r) for r in self.rules) + ')'

    def __str__(self):
        return '(' + ' | '.join(str(r) for r in self.rules) + ')'


class Maybe(Rule):
    """Accepts rule if it matches, otherwise ignores it.

    Parameters
    ----------
    rule : Rule
        Rule that may be matched.
    """

    def __init__(self, rule):
        self.rule = rule

    def accept(self, tokens):
        return True

    def read(self, tokens):
        if self.rule.accept(tokens):
            return self.rule.read(tokens)
        else:
            return []

    def __eq__(self, other):
        return isinstance(other, Maybe) and self.rule == other.rule

    def __repr__(self):
        return '({!r})?'.format(self.rule)

    def __str__(self):
        return '({})?'.format(self.rule)


class AtLeastOne(Rule):
    """Accepts a rule that is repeated at least once.

    Parameters
    ----------
    rule : Rule
        Rule to match at least once.
    """

    def __init__(self, rule):
        self.rule = rule

    def accept(self, tokens):
        return self.rule.accept(tokens)

    def read(self, tokens):
        if not self.accept(tokens):
            raise RuleMismatchError(self, self.rule, tokens.peek()[0])
        match = []
        while self.accept(tokens):
            match.extend(self.rule.read(tokens))
        return match

    def __eq__(self, other):
        return isinstance(other, AtLeastOne) and self.rule == other.rule

    def __repr__(self):
        return '({!r})+'.format(self.rule)

    def __str__(self):
        return '({})+'.format(self.rule)


class AnyNumber(Rule):
    """Accepts any number of occurrences of a rule.

    Parameters
    ----------
    rule : Rule
        Rule to match any number of times.
    """

    def __init__(self, rule):
        self.rule = rule

    def accept(self, tokens):
        return True

    def read(self, tokens):
        match = []
        while self.rule.accept(tokens):
            match.extend(self.rule.read(tokens))
        return match

    def __eq__(self, other):
        return isinstance(other, AnyNumber) and self.rule == other.rule

    def __repr__(self):
        return '({!r})*'.format(self.rule)

    def __str__(self):
        return '({})*'.format(self.rule)


class Group(Rule):
    """Allows to group parse objects and tag them with a name.

    This allows to build up an actual parse tree.
    """
    def __init__(self, name, rule):
        self.name = name
        self.rule = rule

    def accept(self, tokens):
        return self.rule.accept(tokens)

    def read(self, tokens):
        return [(self.name, self.rule.read(tokens))]

    def __eq__(self, other):
        return (
            isinstance(other, Group) and
            self.name == other.name and
            self.rule == other.rule)

    def __repr__(self):
        return 'Group({!r}, {!r})'.format(self.name, self.rule)

    def __str__(self):
        return self.name


class Peek(Rule):
    """Requires that a rule applies without consuming any tokens."""

    def __init__(self, rule):
        self.rule = rule

    def accept(self, tokens):
        return self.rule.accept(tokens)

    def read(self, tokens):
        if not self.accept(tokens):
            raise RuleMismatchError(self, self.rule, tokens.peek()[0])
        return []

    def __eq__(self, other):
        return isinstance(other, Peek) and self.rule == other.rule

    def __repr__(self):
        return 'Peek({!r})'.format(self.rule)

    def __str__(self):
        return 'Peek({})'.format(self.rule)
