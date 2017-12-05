import weakref

import nengo

from nengo_spa.exceptions import SpaTypeError


input_network_registry = weakref.WeakKeyDictionary()
input_vocab_registry = weakref.WeakKeyDictionary()
output_vocab_registry = weakref.WeakKeyDictionary()


class FixedPointer(object):
    def __init__(self, expr, vocab=None):
        self.expr = expr
        self.vocab = vocab

    def add_transform(self, tr):
        pass

    def construct(self, vocab):
        if self.vocab is not None:
            vocab = self.vocab
        return nengo.Node(vocab.parse(self.expr).v, label=self.expr)

    def __invert__(self):
        return FixedPointer('~' + self.expr, self.vocab)

    def __neg__(self):
        return FixedPointer('-' + self.expr, self.vocab)

    def __add__(self, other):
        vocab = self._coerce_vocab(other)
        return FixedPointer(self.expr + '+' + other.expr, vocab)

    def __sub__(self, other):
        vocab = self._coerce_vocab(other)
        return FixedPointer(self.expr + '-' + other.expr, vocab)

    def __mul__(self, other):
        vocab = self._coerce_vocab(other)
        return FixedPointer(self.expr + '*' + other.expr, vocab)

    def _coerce_vocab(self, other):
        if self.vocab is None:
            return other.vocab
        elif other.vocab is None:
            return self.vocab
        elif self.vocab is other.vocab:
            return self.vocab
        else:
            raise SpaTypeError("Vocabulary mismatch.")




# class Transformed(object):
    # def __init__(self, sources, transform):
        # self.sources = sources
        # self.transform = transform
