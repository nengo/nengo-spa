"""Operator-like SPA functions.

The operator functions work similar to other overloaded operators in Python.
They call the class implementation on the operand, e.g. ``reinterpret(a, ...)``
calls ``a.reinterpret(...)`` like ``-a`` calls ``a.__neg__()``. For binary
operators, this will be tried for the left operand first. If the left operand
does not have the corresponding method or if it returns `NotImplemented`, the
same method prefixed with an ``r`` will be called on the right operand. For
example, ``dot(a, b)`` tries ``a.dot(b)`` first and then ``b.rdot(a)``. This is
equivalent to for example ``a + b`` trying ``a.__add__(b)`` first and then
``b.__radd__(a)``.
"""


def dot(a, b):
    """Dot-product between *a* and *b*."""
    result = NotImplemented
    if hasattr(a, "dot"):
        result = a.dot(b)
    if result is NotImplemented and hasattr(b, "dot"):
        result = b.rdot(a)
    if result is NotImplemented:
        raise TypeError(
            "'dot' not supported between instances of '{}' and '{}'".format(
                type(a), type(b)
            )
        )
    return result


def reinterpret(source, vocab=None):
    """Reinterpret *source* Semantic Pointer as part of vocabulary *vocab*.

    The *vocab* parameter can be set to *None* to clear the associated
    vocabulary and allow the *source* to be interpreted as part of the
    vocabulary of any Semantic Pointer it is combined with.
    """
    if hasattr(source, "reinterpret"):
        return source.reinterpret(vocab)
    else:
        raise TypeError("bad operand type for 'reinterpret'")


def translate(source, vocab, populate=None, keys=None, solver=None):
    """Translate *source* Semantic Pointer to vocabulary *vocab*.

    The translation of a Semantic Pointer uses some form of projection to
    convert the source Semantic Pointer to a Semantic Pointer of another
    vocabulary. By default the outer products of terms in the source and
    target vocabulary are used, but if *solver* is given, it is used to find
    a least squares solution for this projection.

    Parameters
    ----------
    source : object
        Source of Semantic Pointer.
    vocab : Vocabulary
        Target vocabulary.
    populate : bool, optional
        Whether the target vocabulary should be populated with missing keys.
        This is done by default, but with a warning. Set this explicitly to
        *True* or *False* to silence the warning or raise an error.
    keys : list, optional
        All keys to translate. If *None*, all keys in the source vocabulary
        will be translated.
    solver : nengo.Solver, optional
        If given, the solver will be used to solve the least squares problem to
        provide a better projection for the translation.
    """
    if hasattr(source, "translate"):
        return source.translate(vocab, populate, keys, solver)
    else:
        raise TypeError("bad operand type for 'translate'")
