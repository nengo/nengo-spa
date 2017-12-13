import weakref

from nengo_spa.exceptions import SpaTypeError
from nengo_spa.types import TInferVocab, TScalar, TVocabulary


input_network_registry = weakref.WeakKeyDictionary()
input_vocab_registry = weakref.WeakKeyDictionary()
output_vocab_registry = weakref.WeakKeyDictionary()


def coerce_types(*types):
    if all(t == TScalar for t in types):
        return TScalar

    defined = [t for t in types if isinstance(t, TVocabulary)]
    if len(defined) > 0:
        if all(t == defined[0] for t in defined):
            return defined[0]
        else:
            raise SpaTypeError("Vocabulary mismatch.")
    else:
        return TInferVocab


def infer_types(*nodes):
    type_ = coerce_types(*[n.type for n in nodes])
    if isinstance(type_, TVocabulary):
        for n in (n for n in nodes if n.type == TInferVocab):
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
