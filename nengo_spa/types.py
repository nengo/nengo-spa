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
        if not isinstance(other, Type):
            return NotImplemented
        return self.__class__ is other.__class__ and self.name == other.name

    def __lt__(self, other):
        return other.__gt__(self)

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        if not isinstance(other, Type):
            return NotImplemented
        return False

    def __ge__(self, other):
        return self > other or self == other


TAction = Type('TAction')
TActionSet = Type('TActionSet')
TEffect = Type('TEffect')
TEffects = Type('TEffects')
TScalar = Type('TScalar')


class _TInferVocab(Type):
    def __gt__(self, other):
        if not isinstance(other, Type):
            return NotImplemented
        return other == TScalar


TInferVocab = _TInferVocab('TInferVocab')


class TVocabDimensions(Type):
    def __init__(self, dimensions):
        super(TVocabDimensions, self).__init__('TVocabDimensions')
        self.dimensions = dimensions

    def __repr__(self):
        return '{}({!r}, {!r})'.format(
            self.__class__.__name__, self.name, self.dimensions)

    def __str__(self):
        return '{}<{}>'.format(self.name, self.dimensions)

    def __eq__(self, other):
        if not isinstance(other, Type):
            return NotImplemented
        return (super(TVocabDimensions, self).__eq__(other) and
                self.dimensions == other.dimensions)

    def __gt__(self, other):
        if not isinstance(other, Type):
            return NotImplemented
        return other <= TInferVocab


class TVocabulary(Type):
    """Each vocabulary is treated as its own type.

    All vocabulary types constitute a type class.
    """

    def __init__(self, vocab):
        super(TVocabulary, self).__init__('TVocabulary')
        self.vocab = vocab

    @property
    def dimensions(self):
        return self.vocab.dimensions

    def __repr__(self):
        return '{}({!r}, {!r})'.format(
            self.__class__.__name__, self.name, self.vocab)

    def __str__(self):
        return '{}<{}>'.format(self.name, self.vocab)

    def __hash__(self):
        return super(TVocabulary, self).__hash__() ^ hash(self.vocab)

    def __eq__(self, other):
        if not isinstance(other, Type):
            return NotImplemented
        return (super(TVocabulary, self).__eq__(other) and
                self.vocab is other.vocab)

    def __gt__(self, other):
        if not isinstance(other, Type):
            return NotImplemented
        return other <= TVocabDimensions(self.vocab.dimensions)
