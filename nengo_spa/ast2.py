import weakref

import nengo
import numpy as np

from nengo_spa.exceptions import SpaTypeError


input_network_registry = weakref.WeakKeyDictionary()
input_vocab_registry = weakref.WeakKeyDictionary()
output_vocab_registry = weakref.WeakKeyDictionary()

BindRealization = None
ProductRealization = None


def coerce_vocabs(*vocabs):
    defined = [v for v in vocabs if v is not None]
    if len(defined) > 0:
        if all(v is defined[0] for v in defined):
            return defined[0]
        else:
            raise SpaTypeError("Vocabulary mismatch.")
    else:
        return None


class FixedPointer(object):
    def __init__(self, expr, vocab=None):
        self.expr = expr
        self.vocab = vocab

    def construct(self, vocab):
        if self.vocab is not None:
            vocab = self.vocab
        return nengo.Node(vocab.parse(self.expr).v, label=self.expr)

    def __invert__(self):
        return FixedPointer('~' + self.expr, self.vocab)

    def __neg__(self):
        return FixedPointer('-' + self.expr, self.vocab)

    def __add__(self, other):
        vocab = coerce_vocabs(self.vocab, other.vocab)
        return FixedPointer(self.expr + '+' + other.expr, vocab)

    def __sub__(self, other):
        vocab = coerce_vocabs(self.vocab, other.vocab)
        return FixedPointer(self.expr + '-' + other.expr, vocab)

    def __mul__(self, other):
        vocab = coerce_vocabs(self.vocab, other.vocab)
        return FixedPointer(self.expr + '*' + other.expr, vocab)


class Transformed(object):
    def __init__(self, source, transform, vocab):
        self.source = source
        self.transform = transform
        self.vocab = vocab

    def connect_to(self, sink):
        # FIXME connection params
        return nengo.Connection(self.source, sink, transform=self.transform)

    def construct(self):
        node = nengo.Node(size_in=self.vocab.dimensions)
        self.connect_to(node)
        return node


class Summed(object):
    def __init__(self, sources, vocab):
        self.sources = sources
        self.vocab = vocab

    def connect_to(self, sink):
        for s in self.sources:
            s.connect_to(sink)

    def construct(self):
        node = nengo.Node(size_in=self.vocab.dimensions)
        self.connect_to(node)
        return node


class ModuleOutput(object):
    def __init__(self, output, vocab):
        self.output = output
        self.vocab = vocab

    def construct(self):
        return self.output

    def connect_to(self, sink):
        nengo.Connection(self.output, sink)

    def __invert__(self):
        # FIXME alternate binding operators
        vocab = self.vocab
        transform = np.eye(vocab.dimensions)[-np.arange(vocab.dimensions)]
        return Transformed(self.output, transform, vocab)

    def __neg__(self):
        return Transformed(self.output, transform=-1, vocab=self.vocab)

    def __add__(self, other):
        vocab = coerce_vocabs(self.vocab, other.vocab)
        return Summed((self, other), vocab)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if self.vocab is None and other.vocab is None:
            mul = ProductRealization()
        elif self.vocab is other.vocab:
            mul = BindRealization(self.vocab)
        else:
            raise NotImplementedError(
                "Dynamic scaling of semantic pointer not implemented.")

        self.connect_to(mul.input_a)
        other.connect_to(mul.input_b)
        return ModuleOutput(mul.output, self.vocab)


# class Transformed(object):
    # def __init__(self, sources, transform):
        # self.sources = sources
        # self.transform = transform
