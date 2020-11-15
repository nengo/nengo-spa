"""Basic classes for abstract syntax trees (ASTs) in NengoSPA."""

from nengo_spa.typechecks import is_array
from nengo_spa.types import TAnyVocab, TVocabulary, coerce_types


def infer_types(*nodes):
    """Infers the most specific type for given nodes, sets end returns it.

    This determines the most specific type for given nodes. If it is a specific
    vocabulary, this vocabulary will be set for all less specific vocabulary
    types (but not scalar types) on the given node. Then the type will be
    returned.
    """
    type_ = coerce_types(*[n.type for n in nodes])
    if isinstance(type_, TVocabulary):
        for n in nodes:
            if TAnyVocab <= n.type < type_:
                n.type = type_
    return type_


class Node:
    """Base class for nodes in the AST for NengoSPA operations.

    Parameters
    ----------
    type_ : nengo_spa.types.Type
        Type that the node evaluates to.
    """

    def __init__(self, type_):
        self.type = type_

    def connect_to(self, sink, **kwargs):
        """Implement the computation represented by the node and connect it.

        Parameters
        ----------
        sink : NengoObject
            Nengo object to connect to and transmit the result to.
        **kwargs : dict
            Additional keyword arguments to pass to `nengo.Connection`.
        """
        raise NotImplementedError()

    def construct(self):
        """Implement the computation represented by the node.

        Returns
        -------
        NengoObject
            Usually the object providing the computation result as output, but
            can be something else in certain cases.
        """
        raise NotImplementedError()


class Noop(Node):
    """Node that has no effect."""

    def connect_to(self, sink, **kwargs):
        pass

    def construct(self):
        raise NotImplementedError("Noop nodes cannot be constructed.")


class Fixed(Node):
    """Base class for AST nodes that provide a fixed value."""

    def evaluate(self):
        """Return the fixed value that the node evaluates to."""
        raise NotImplementedError()


class TypeCheckedBinaryOp:
    """Decorator to check the type of the *other* parameter of an operator.

    If the *other* parameter is not an instance of *expected_type*,
    *NotImplemented* will be returned from the decorated method. If
    *conversion* is given it will be applied first.

    Parameters
    ----------
    expected_type : class
        Type for which the operator is implemented.
    conversion : function, optional
        Used to convert *other* before checking its type.
    """

    __slots__ = ["expected_type", "conversion"]

    def __init__(self, expected_type, conversion=None):
        self.expected_type = expected_type
        self.conversion = conversion

    def __call__(self, fn):
        def checked(inst, other):
            if self.conversion is not None:
                other = self.conversion(other)
            if is_array(other):
                raise TypeError("Bare array not allowed in SPA operation.")
            elif not isinstance(other, self.expected_type):
                return NotImplemented
            return fn(inst, other)

        return checked
