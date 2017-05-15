import nengo
from nengo.config import Default
from nengo.exceptions import ValidationError
from nengo.params import IntParam, Parameter
from nengo.utils.stdlib import checked_call

from nengo_spa.network import Network
from nengo_spa.pointer import SemanticPointer
from nengo_spa.vocab import VocabularyOrDimParam


def make_sp_func(fn, vocab):
    def sp_func(t, v):
        return fn(t, SemanticPointer(v), vocab)
    return sp_func


class DecodeFunctionParam(Parameter):
    def coerce(self, obj, value):
        if not callable(value):
            raise ValidationError("Not callable.", attr=self.name, obj=obj)
        _, invoked = checked_call(
            value, 0., SemanticPointer(obj.vocab.dimensions), obj.vocab)
        if not invoked:
            raise ValidationError(
                "Decode function %r is expected to accept exactly 3 "
                "arguments: time as a float, a SemanticPointer, and a "
                "Vocabulary.", attr=self.name, obj=node)
        return super(DecodeFunctionParam, self).coerce(obj, value)


class Decode(Network):
    function = DecodeFunctionParam(
        'function', optional=False, default=None, readonly=True)
    vocab = VocabularyOrDimParam(
        'vocab', optional=False, default=None, readonly=True)
    size_out = IntParam(
        'size_out', low=0, optional=True, default=None, readonly=True)

    def __init__(
            self, function=Default, vocab=Default, size_out=Default, **kwargs):
        super(Decode, self).__init__(**kwargs)

        # Vocab needs to be set before function which accesses vocab for
        # validation.
        self.vocab = vocab
        self.function = function

        with self:
            self.node = nengo.Node(
                make_sp_func(self.function, self.vocab),
                size_in=self.vocab.dimensions, size_out=self.size_out)
            self.input = self.node
            self.output = self.node

        self.inputs = dict(default=(self.input, self.vocab))
        self.outputs = dict()
