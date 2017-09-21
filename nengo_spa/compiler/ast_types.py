"""Definition of types used in the AST."""


class Type(object):
    """Describes a type.

    Each part of the AST evaluates to some type.
    """
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self.name)

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.__class__) ^ hash(self.name)

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.name == other.name

    def __ne__(self, other):
        return not self == other


TAction = Type('TAction')
TActionSet = Type('TActionSet')
TScalar = Type('TScalar')
TEffect = Type('TEffect')
TEffects = Type('TEffects')


class TVocabulary(Type):
    """Each vocabulary is treated as its own type.

    All vocabulary types constitute a type class.
    """
    def __init__(self, vocab):
        super(TVocabulary, self).__init__('TVocabulary')
        self.vocab = vocab

    def __repr__(self):
        return '{}({!r}, {!r})'.format(
            self.__class__.__name__, self.name, self.vocab)

    def __str__(self):
        return '{}<{}>'.format(self.name, self.vocab)

    def __hash__(self):
        return super(TVocabulary, self).__hash__() ^ hash(self.vocab)

    def __eq__(self, other):
        return (super(TVocabulary, self).__eq__(other) and
                self.vocab is other.vocab)
