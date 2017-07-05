import nengo
from nengo.config import Default
from nengo.exceptions import ValidationError
from nengo.params import IntParam, Parameter
from nengo.utils.compat import is_string
from nengo.utils.stdlib import checked_call
import numpy as np

from nengo_spa.network import Network
from nengo_spa.pointer import SemanticPointer
from nengo_spa.vocab import VocabularyOrDimParam


class SpArrayExtractor(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, value):
        if is_string(value):
            value = self.vocab.parse(value)
        if isinstance(value, SemanticPointer):
            value = value.v
        return value


def make_sp_func(fn, vocab):
    def sp_func(t, v):
        return fn(t, SemanticPointer(v), vocab)
    return sp_func


def make_parse_func(fn, vocab):
    """Create a function that calls func and parses the output in vocab."""

    extractor = SpArrayExtractor(vocab)

    def parse_func(*args):
        return extractor(fn(*args))

    return parse_func


class TranscodeFunctionParam(Parameter):
    def coerce(self, obj, fn):
        fn = super(TranscodeFunctionParam, self).coerce(obj, fn)

        if fn is None:
            return fn
        elif callable(fn):
            return self.coerce_callable(obj, fn)
        elif not obj.input_vocab and is_string(fn):
            return fn
        else:
            raise ValidationError("Invalid output type {!r}".format(
                type(fn)), attr=self.name, obj=obj)

    def coerce_callable(self, obj, fn):
        t = 0.
        if obj.input_vocab is not None:
            args = (t, SemanticPointer(obj.input_vocab.dimensions),
                    obj.input_vocab)
        elif obj.size_in is not None:
            args = (t, np.zeros(obj.size_in))
        else:
            args = (t,)

        _, invoked = checked_call(fn, *args)
        fn(*args)
        if not invoked:
            if obj.input_vocab is not None:
                raise ValidationError(
                    "Transcode function %r is expected to accept exactly 3 "
                    "arguments: time as a float, a SemanticPointer, and a "
                    "Vocabulary.", attr=self.name, obj=obj)
            else:
                raise ValidationError(
                    "Transcode function %r is expected to accept exactly 1 or 2 "
                    "arguments: time as a float, and optionally the input "
                    "data as NumPy array.", attr=self.name, obj=obj)
        return fn

    @classmethod
    def to_node_output(cls, fn, input_vocab=None, output_vocab=None):
        if fn is None:
            return None
        elif callable(fn):
            if input_vocab is not None:
                fn = make_sp_func(fn, input_vocab)
            if output_vocab is not None:
                fn = make_parse_func(fn, output_vocab)
            return fn
        elif is_string(fn):
            return SpArrayExtractor(output_vocab)(fn)
        else:
            raise ValueError("Invalid output type {!r}".format(type(fn)))


class Transcode(Network):
    function = TranscodeFunctionParam(
        'function', optional=True, default=None, readonly=True)
    input_vocab = VocabularyOrDimParam(
        'input_vocab', optional=True, default=None, readonly=True)
    output_vocab = VocabularyOrDimParam(
        'output_vocab', optional=True, default=None, readonly=True)
    size_in = IntParam(
        'size_in', optional=True, default=None, readonly=True)
    size_out = IntParam(
        'size_out', optional=True, default=None, readonly=True)

    def __init__(
            self, function=Default, input_vocab=Default, output_vocab=Default,
            size_in=Default, size_out=Default, **kwargs):
        super(Transcode, self).__init__(**kwargs)

        # Vocabs need to be set before function which accesses vocab for
        # validation.
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.size_in = size_in
        self.size_out = size_out

        if self.input_vocab is None and self.output_vocab is None:
            raise ValidationError(
                "At least one of input_vocab and output_vocab needs to be "
                "set. If neither the input nor the output is a Semantic "
                "Pointer, use a basic nengo.Node instead.", self)
        if self.input_vocab is not None and self.size_in is not None:
            raise ValidationError(
                "The input_vocab and size_in arguments are mutually "
                "exclusive.", 'size_in', self)
        if self.output_vocab is not None and self.size_out is not None:
            raise ValidationError(
                "The output_vocab and size_out arguments are mutually "
                "exclusive.", 'size_in', self)

        self.function = function

        node_size_in = (self.input_vocab.dimensions
                        if self.input_vocab is not None else self.size_in)
        node_size_out = (self.output_vocab.dimensions
                         if self.output_vocab is not None else self.size_out)
        if self.function is None:
            if node_size_in is None:
                node_size_in = self.output_vocab.dimensions
            node_size_out = None

        with self:
            self.node = nengo.Node(
                TranscodeFunctionParam.to_node_output(
                    self.function, self.input_vocab, self.output_vocab),
                size_in=node_size_in, size_out=node_size_out)
            self.input = self.node
            self.output = self.node

        if self.input_vocab is not None:
            self.declare_input(self.input, self.input_vocab)
        if self.output_vocab is not None:
            self.declare_output(self.output, self.output_vocab)
