import re
import warnings
from collections.abc import Mapping
from keyword import iskeyword

import nengo
import numpy as np
from nengo.exceptions import NengoWarning, ValidationError

from nengo_spa import semantic_pointer
from nengo_spa.algebras.hrr_algebra import HrrAlgebra
from nengo_spa.exceptions import SpaParseError
from nengo_spa.semantic_pointer import AbsorbingElement, Identity, Zero
from nengo_spa.typechecks import is_integer, is_iterable, is_number
from nengo_spa.vector_generation import UnitLengthVectors

valid_sp_regex = re.compile("^[A-Z][_a-zA-Z0-9]*$")
special_sps = {
    "AbsorbingElement": AbsorbingElement,
    "Identity": Identity,
    "Zero": Zero,
}
reserved_sp_names = {"None", "True", "False"} | set(special_sps.keys())


class Vocabulary(Mapping):
    """A collection of semantic pointers, each with their own text label.

    The Vocabulary can also act as a dictionary, with keys as the names
    of the semantic pointers and values as the `.SemanticPointer` objects
    themselves. The names of Semantic Pointers must be valid Python 2
    identifiers starting with a capital letter.

    Every vocabulary knows the special elements *AbsorbingElement*, *Identity*,
    and *Zero*. However, these are not included in the keys returned by `.keys`
    or the vectors returned by `.vectors`.

    Parameters
    -----------
    dimensions : int
        Number of dimensions for each semantic pointer.
    strict : bool, optional
        Whether to automatically create missing semantic pointers. If a
        non-strict vocabulary is asked for a pointer that does not exist within
        the vocabulary, the missing pointer will be automatically added to the
        vocabulary. A strict vocabulary will throw an error if asked for a
        pointer that does not exist in the vocabulary.
    max_similarity : float, optional
        When randomly generating pointers, ensure that the cosine of the
        angle between the new pointer and all existing pointers is less
        than this amount. If the system is unable to find such a pointer
        after 100 tries, a warning message is printed.
    pointer_gen : generator or np.random.RandomState, optional
        Generator used to create vectors for new Semantic Pointers. Defaults to
        `.UnitLengthVectors`. If a `np.random.RandomState` is passed, it will
        be used by `.UnitLengthVectors`.
    name : str
        A name to display in the string representation of this vocabulary.
    algebra : AbstractAlgebra, optional
        Defines the vector symbolic operators used for Semantic Pointers in the
        vocabulary. Defaults to `.HrrAlgebra`.

    Attributes
    ----------
    keys : sequence
        The names of all known semantic pointers (e.g., ``['A', 'B', 'C']``).
    max_similarity : float
        When randomly generating pointers, ensure that the cosine of the
        angle between the new pointer and all existing pointers is less
        than this amount. If the system is unable to find such a pointer
        after 100 tries, a warning message is printed.
    strict : bool
        Whether to automatically create missing semantic pointers. If a
        non-strict vocabulary is asked for a pointer that does not exist within
        the vocabulary, the missing pointer will be automatically added to the
        vocabulary. A strict vocabulary will throw an error if asked for a
        pointer that does not exist in the vocabulary.
    vectors : ndarray
        All of the semantic pointer vectors in a matrix, in the same order
        as in `keys`.
    algebra : AbstractAlgebra, optional
        Defines the vector symbolic operators used for Semantic Pointers in the
        vocabulary.
    """

    def __init__(
        self,
        dimensions,
        strict=True,
        max_similarity=0.1,
        pointer_gen=None,
        name=None,
        algebra=None,
    ):
        if algebra is None:
            algebra = HrrAlgebra()
        self.algebra = algebra

        if not is_integer(dimensions) or dimensions < 1:
            raise ValidationError(
                "dimensions must be a positive integer", attr="dimensions", obj=self
            )

        if pointer_gen is None:
            pointer_gen = UnitLengthVectors(dimensions)
        elif isinstance(pointer_gen, np.random.RandomState):
            pointer_gen = UnitLengthVectors(dimensions, pointer_gen)

        if not is_iterable(pointer_gen) or isinstance(pointer_gen, str):
            raise ValidationError(
                "pointer_gen must be iterable or RandomState",
                attr="pointer_gen",
                obj=self,
            )

        self.dimensions = dimensions
        self.strict = strict
        self.max_similarity = max_similarity
        self._key2idx = {}
        self._keys = []
        self._vectors = np.zeros((0, dimensions), dtype=float)
        self.pointer_gen = pointer_gen
        self.name = name

    @property
    def vectors(self):
        v = self._vectors.view()
        v.setflags(write=False)
        return v

    def __str__(self):
        name = "" if self.name is None else '"{}" '.format(self.name)
        return "{}-dimensional vocab {}at 0x{:x}".format(
            self.dimensions, name, id(self)
        )

    def create_pointer(self, attempts=100, transform=None):
        """Create a new semantic pointer and add it to the vocabulary.

        This will take into account the `max_similarity` attribute.  If a
        pointer satisfying max_similarity is not generated after the specified
        number of attempts, the candidate pointer with lowest maximum cosine
        similarity with all existing pointers is returned.

        Parameters
        ----------
        attempts : int, optional
            Maximum number of attempts to create a Semantic Pointer not
            exceeding `max_similarity`.
        transform : str, optional
            A transform to apply to the generated vector. Needs to be the name
            of a method of `.SemanticPointer`. Currently, the only sensible
            value is 'unitary'.

        Returns
        -------
        SemanticPointer
            The generated Semantic Pointer.
        """
        best_p = None
        best_sim = np.inf
        for _ in range(attempts):
            # note: p will get its algebra from vocab.algebra
            p = semantic_pointer.SemanticPointer(next(self.pointer_gen), vocab=self)
            if transform is not None:
                p = eval("p." + transform, dict(self), {"p": p})
            if len(self) == 0:
                best_p = p
                break
            else:
                p_sim = np.max(np.dot(self._vectors, p.v))
                if p_sim < best_sim:
                    best_p = p
                    best_sim = p_sim
                    if p_sim < self.max_similarity:
                        break
        else:
            warnings.warn(
                "Could not create a semantic pointer with "
                "max_similarity=%1.2f (D=%d, M=%d, similarity=%1.2f)"
                % (self.max_similarity, self.dimensions, len(self._key2idx), best_sim)
            )
        return best_p

    def __contains__(self, key):
        return key in special_sps or key in self._key2idx

    def __len__(self):
        return len(self._vectors)

    def __iter__(self):
        return iter(self._keys)

    def __getitem__(self, key):
        """Return the semantic pointer with the requested name."""
        # __tracebackhide__ is used in py.test to hide stack frames from the
        # traceback. That means py.test might try to look up this attribute
        # in a test which will result in an exception hiding the actual
        # exception. By raising a KeyError we indicate that there is no
        # __tracebackhide__ attribute on this object and preserve the relevant
        # exception.
        if key == "__tracebackhide__":
            raise KeyError()
        if key in special_sps:
            return special_sps[key](self.dimensions, self)
        if not self.strict and key not in self:
            self.add(key, self.create_pointer())
        # note: pointer will get its algebra from vocab.algebra
        return semantic_pointer.SemanticPointer(
            self._vectors[self._key2idx[key]], vocab=self, name=key
        )

    def __hash__(self):
        return hash(id(self))

    def add(self, key, p):
        """Add the semantic pointer *p* to the vocabulary.

        Parameters
        ----------
        key : str
            Name of the Semantic Pointer. Must be a valid Python 2 identifier
            starting with a capital letter. Must not be *AbsorbingElement*,
            *Identity*, or *Zero*.
        p : SemanticPointer or array_like
            Semantic Pointer to add.
        """
        if not valid_sp_regex.match(key) or iskeyword(key) or key in reserved_sp_names:
            raise SpaParseError(
                "Invalid Semantic Pointer name {!r}. Valid names are valid "
                "Python 2 identifiers beginning with a capital letter.".format(key)
            )
        if not isinstance(p, semantic_pointer.SemanticPointer):
            # note: p will get its algebra from vocab.algebra
            p = semantic_pointer.SemanticPointer(p, vocab=self)

        if key in self._key2idx:
            raise ValidationError(
                "The semantic pointer %r already exists" % key, attr="", obj=self
            )
        isDifferentVocab = p.vocab is not None and p.vocab is not self
        isDifferentAlgebra = p.algebra is not self.algebra  # algebra never None
        if isDifferentVocab or isDifferentAlgebra:
            raise ValidationError(
                "Cannot add a semantic pointer that belongs to a different "
                "vocabulary or algebra.",
                attr="",
                obj=self,
            )

        self._key2idx[key] = len(self._key2idx)
        self._keys.append(key)
        self._vectors = np.vstack([self._vectors, p.v])

    def populate(self, pointers):
        """Populate the vocabulary with semantic pointers given an expression.

        In its most basic form *pointers* is a string of names separated with
        ``;``::

            vocab.populate('A; B; C')

        Semantic Pointers can be constructed from other Semantic Pointers::

            vocab.populate('A; B; C = 0.3 * A + 1.4 * C')

        Those constructed Semantic Pointers are **not** normalized to
        unit-length. This can be done by appending a ``normalized()`` call.
        In the same way unitary Semantic Pointers can be obtained with
        ``unitary()``::

            vocab.populate('A.unitary(); B; C = (A+B).normalized()')

        Parameters
        ----------
        pointers : string
            The expression defining the semantic pointers to
            add to the vocabulary.
        """
        if len(pointers.strip()) <= 0:
            return  # Do nothing (and don't fail) for empty string.

        for p_expr in pointers.split(";"):
            assign_split = p_expr.split("=", 1)
            modifier_split = p_expr.split(".", 1)
            if len(assign_split) > 1:
                name, value_expr = assign_split
                value = eval(value_expr.strip(), {}, self)
            elif len(modifier_split) > 1:
                name = modifier_split[0]
                value = self.create_pointer(transform=modifier_split[1])
            else:
                name = p_expr
                value = self.create_pointer()
            self.add(name.strip(), value)

    def parse(self, text):
        """Evaluate a text string and return the corresponding SemanticPointer.

        This uses the Python ``eval()`` function, so any Python operators that
        have been defined for SemanticPointers are valid (``+``, ``-``, ``*``,
        ``~``, ``()``). Valid semantic pointer terms must start
        with a capital letter.

        If the expression returns a scalar (int or float), a scaled version
        of the identity SemanticPointer will be returned.
        """

        # The following line does everything.  Note that self is being
        # passed in as the locals dictionary, and thanks to the __getitem__
        # implementation, this will automatically create new semantic
        # pointers as needed.
        try:
            value = eval(text, {}, self)
        except NameError as err:
            raise SpaParseError(
                "Error parsing expression {expr!r} with {vocab}: {msg}".format(
                    expr=text, vocab=self, msg=str(err)
                )
            )

        if is_number(value):
            value *= Identity(self.dimensions)
        elif not isinstance(value, semantic_pointer.SemanticPointer):
            raise SpaParseError(
                "The result of parsing '%s' is not a SemanticPointer." % text
            )
        return value

    def parse_n(self, *texts):
        """Applies `parse` to each item in *texts* and returns the result."""
        return [self.parse(t) for t in texts]

    def dot(self, v):
        """Returns the dot product with all terms in the Vocabulary.

        Parameters
        ----------
        v : SemanticPointer or array_like
            SemanticPointer to calculate dot product with.
        """
        if isinstance(v, semantic_pointer.SemanticPointer):
            v = v.v
        return np.dot(self._vectors, v)

    def transform_to(self, other, populate=None, keys=None, solver=None):
        """Create a linear transform from one Vocabulary to another.

        This is simply the sum of the outer products of the corresponding
        terms in each Vocabulary if no *solver* is given, otherwise a
        least-squares solution will be obtained.

        Parameters
        ----------
        other : Vocabulary
            The vocabulary to translate into.
        populate : Boolean
            Whether to add the missing keys from the original vocabulary
            to the new target vocabulary.
        keys : list, optional
            Limits the Semantic Pointers considered from the original
            vocabulary if given.
        solver: callable
            Solver to obtain least-squares solution to map one vocabulary to
            the other.
        """
        if keys is None:
            keys = self._keys
        keys = set(keys)

        missing_keys = set(k for k in keys if k not in other)

        if len(missing_keys) > 0:
            if populate is None:
                warnings.warn(
                    NengoWarning(
                        "The transform_to source vocabulary has keys not existent "
                        "in the target vocabulary. These will be ignored. Use the "
                        "`populate=False` keyword argument to silence this "
                        "warning or `populate=True` to automatically add missing "
                        "keys to the target vocabulary."
                    )
                )
            elif populate:
                other.populate(";".join(missing_keys))
                missing_keys = set()

        from_vocab = self.create_subset(keys - missing_keys).vectors
        to_vocab = other.create_subset(keys - missing_keys).vectors
        if solver is None:
            return np.dot(to_vocab.T, from_vocab)
        else:
            return solver(from_vocab, to_vocab)[0].T

    def create_subset(self, keys):
        """Returns a subset of this vocabulary.

        Creates and returns a subset of the current vocabulary that contains
        all the semantic pointers found in keys.

        Parameters
        ----------
        keys : sequence
            List or set of semantic pointer names to be copied over to the
            new vocabulary.
        """
        # Make new Vocabulary object
        subset = Vocabulary(
            self.dimensions,
            self.strict,
            self.max_similarity,
            pointer_gen=self.pointer_gen,
            algebra=self.algebra,
        )

        # Copy over the new keys
        for key in keys:
            subset.add(key, self[key].reinterpret(subset))

        return subset


class VocabularyMap(Mapping):
    """Maps dimensionalities to corresponding vocabularies.

    Acts like a Python dictionary.

    Parameters
    ----------
    vocabs : sequence of Vocabulary
        A list of vocabularies to add to the mapping. The dimensionalities
        will be determined from the vocabulary objects.
    rng : numpy.random.RandomState
        Random number generator to use for newly created vocabularies (with
        `.get_or_create`).
    """

    def __init__(self, vocabs=None, rng=None):
        if vocabs is None:
            vocabs = []
        self.rng = rng

        self._vocabs = {}
        try:
            for vo in vocabs:
                self.add(vo)
        except (AttributeError, TypeError):
            raise ValueError(
                "The `vocabs` argument requires a list of Vocabulary "
                "instances or `None`."
            )

    def add(self, vocab):
        """Add a vocabulary to the map.

        The dimensionality will be determined from the vocabulary.

        Parameters
        ----------
        vocab : Vocaublary
            Vocabulary to add.
        """
        if vocab.dimensions in self._vocabs:
            warnings.warn(
                "Duplicate vocabularies with dimension %d. "
                "Using the last entry in the vocab list with "
                "that dimensionality." % vocab.dimensions
            )
        self._vocabs[vocab.dimensions] = vocab

    def __delitem__(self, dimensions):
        del self._vocabs[dimensions]

    def discard(self, vocab):
        """Discard (remove) a vocabulary from the mapping.

        Parameters
        ----------
        vocab : int or Vocabulary
            If an integer is given, the vocabulary associated to the
            dimensionality will be discarded. If a `.Vocabulary` is given, that
            specific instance will be discarded.
        """
        if isinstance(vocab, int):
            del self._vocabs[vocab]
        elif self._vocabs.get(vocab.dimensions, None) is vocab:
            del self._vocabs[vocab.dimensions]

    def __getitem__(self, dimensions):
        return self._vocabs[dimensions]

    def get_or_create(self, dimensions):
        """Gets or creates a vocabulary of given dimensionality.

        If the mapping already maps the given dimensionality to a vocabulary,
        it will be returned. Otherwise, a new vocabulary will be created,
        added to the mapping, and returned.

        Parameters
        ----------
        dimensions : int
            Dimensionality of vocabulary to return.

        Returns
        -------
        Vocabulary
            Vocabulary of given dimensionality.
        """
        if dimensions not in self._vocabs:
            self._vocabs[dimensions] = Vocabulary(
                dimensions,
                strict=False,
                pointer_gen=UnitLengthVectors(dimensions, self.rng),
            )
        return self._vocabs[dimensions]

    def __iter__(self):
        return iter(self._vocabs)

    def __len__(self):
        return len(self._vocabs)

    def __contains__(self, vocab):
        if isinstance(vocab, int):
            return vocab in self._vocabs
        else:
            return (
                vocab.dimensions in self._vocabs
                and self._vocabs[vocab.dimensions] is vocab
            )


class VocabularyMapParam(nengo.params.Parameter):
    """Nengo parameter that accepts `.VocabularyMap` instances.

    Sequences of `.Vocabulary` will be coerced to `.VocabularyMap`.
    """

    def coerce(self, instance, vocab_set):
        vocab_set = super(VocabularyMapParam, self).coerce(instance, vocab_set)

        if vocab_set is not None and not isinstance(vocab_set, VocabularyMap):
            try:
                vocab_set = VocabularyMap(vocab_set)
            except ValueError:
                raise ValidationError(
                    "Must be of type 'VocabularyMap' or compatible "
                    "(got type %r)." % type(vocab_set).__name__,
                    attr=self.name,
                    obj=instance,
                )

        return vocab_set


class VocabularyOrDimParam(nengo.params.Parameter):
    """Nengo parameter that accepts `.Vocabulary` or integer dimensionality.

    If an integer is assigned, the vocabulary will retrieved from the
    instance's *vocabs* attribute with *vocabs.get_or_create(dimensions)*.
    Thus, a class using *VocabularyOrDimParam* should also have an attribute
    *vocabs* of type `VocabularyMap`.
    """

    coerce_defaults = False

    def coerce(self, instance, value):
        value = super(VocabularyOrDimParam, self).coerce(instance, value)

        if value is not None:
            if is_integer(value):
                if value < 1:
                    raise ValidationError(
                        "Vocabulary dimensionality must be at least 1.",
                        attr=self.name,
                        obj=instance,
                    )
                value = instance.vocabs.get_or_create(value)
            elif not isinstance(value, Vocabulary):
                raise ValidationError(
                    "Must be of type 'Vocabulary' or an integer (got type %r)."
                    % type(value).__name__,
                    attr=self.name,
                    obj=instance,
                )
        return value
