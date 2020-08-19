"""Types used in the NengoSPA type system to verify operations."""

from nengo_spa.exceptions import SpaTypeError


class Type:
    """Describes a type.

    Types can be compared and by default two types are considered to be equal
    when their class and name match. Subclasses are allowed to overwrite
    *__eq__*.

    Furthermore, a partial ordering can be defined over types by overwriting
    the *__gt__* method. A call to ``a.__gt__(b)`` should return *True*, if
    (and only if) *b* can be cast to the type *a*. For example, the type for an
    unspecified vocabulary can be cast to the type for a specific vocabulary.

    Note, that other comparison operators will be implemented on the
    implementation of *__gt__*.

    Parameters
    ----------
    name : str
        Name of the type.
    """

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.name)

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.__class__) ^ hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Type):
            return NotImplemented
        return self.__class__ is other.__class__ and self.name == other.name

    def __ne__(self, other):
        return not (self == other)

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


TScalar = Type("TScalar")


class _TAnyVocab(Type):
    """Type that allows for any vocabulary."""

    def __gt__(self, other):
        if not isinstance(other, Type):
            return NotImplemented
        return other == TScalar


TAnyVocab = _TAnyVocab("TAnyVocab")


class TAnyVocabOfDim(Type):
    """Type that allows for any vocab of a given dimensionality."""

    def __init__(self, dimensions):
        super(TAnyVocabOfDim, self).__init__("TAnyVocabOfDim")
        self.dimensions = dimensions

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.dimensions)

    def __str__(self):
        return "{}<{}>".format(self.name, self.dimensions)

    def __eq__(self, other):
        if not isinstance(other, Type):
            return NotImplemented
        return (
            super(TAnyVocabOfDim, self).__eq__(other)
            and self.dimensions == other.dimensions
        )

    def __gt__(self, other):
        if not isinstance(other, Type):
            return NotImplemented
        return other <= TAnyVocab


class TVocabulary(Type):
    """Type for a specific vocabulary.

    All vocabulary types constitute a type class.
    """

    def __init__(self, vocab):
        super(TVocabulary, self).__init__("TVocabulary")
        self.vocab = vocab

    @property
    def dimensions(self):
        return self.vocab.dimensions

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.vocab)

    def __str__(self):
        return "{}<{}>".format(self.name, self.vocab)

    def __hash__(self):
        return super(TVocabulary, self).__hash__() ^ hash(self.vocab)

    def __eq__(self, other):
        if not isinstance(other, Type):
            return NotImplemented
        return super(TVocabulary, self).__eq__(other) and self.vocab is other.vocab

    def __gt__(self, other):
        if not isinstance(other, Type):
            return NotImplemented
        return other <= TAnyVocabOfDim(self.vocab.dimensions)


def coerce_types(*types):
    """Returns the most specific type in the argument list.

    If the types passed in the argument list are incompatible a `.SpaTypeError`
    will be raised.

    The specificity of a types is defined by their partial ordering implemented
    in the type classes.
    """
    type_ = max(types)
    if not all(t <= type_ for t in types):
        offender = next(iter(t for t in types if not t <= type_))
        if (
            hasattr(offender, "vocab")
            and hasattr(type_, "vocab")
            and offender.vocab is not type_.vocab
        ):
            reason = "Different vocabularies"
        elif (
            hasattr(offender, "dimensions")
            and hasattr(type_, "dimensions")
            and offender.dimensions != type_.dimensions
        ):
            reason = "Dimensionality mismatch"
        else:
            reason = "Incompatible types"
        raise SpaTypeError(reason + ": " + ", ".join(str(t) for t in types))
    return type_
