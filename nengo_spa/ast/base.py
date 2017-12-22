"""Basic classes for abstract syntax trees (ASTs) in Nengo SPA."""

from nengo.config import Default

from nengo_spa.types import coerce_types, TAnyVocab, TVocabulary


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


class Node(object):
    """Base class for nodes in the AST for Nengo SPA operations.

    Parameters
    ----------
    type_ : nengo_spa.types.Type
        Type that the node evaluates to.
    """
    def __init__(self, type_):
        self.type = type_

    def connect_to(self, sink, transform=Default):
        """Implement the computation represented by the node and connect it.

        Parameters
        ----------
        sink : NengoObject
            Nengo object to connect to and transmit the result to.
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


class Fixed(Node):
    """Base class for AST nodes that provide a fixed value."""

    def evaluate(self):
        """Return the fixed value that the node evaluates to."""
        raise NotImplementedError()
