"""Basic classes for abstract syntax trees (ASTs) in Nengo SPA."""

from nengo_spa.types import coerce_types, TAnyVocab, TVocabulary


def infer_types(*nodes):
    type_ = coerce_types(*[n.type for n in nodes])
    if isinstance(type_, TVocabulary):
        for n in nodes:
            if TAnyVocab <= n.type < type_:
                n.type = type_
    return type_


class Node(object):
    def __init__(self, type_):
        self.type = type_

    def connect_to(self, sink):
        raise NotImplementedError()

    def construct(self):
        raise NotImplementedError()


class Fixed(Node):
    def evaluate(self):
        raise NotImplementedError()
