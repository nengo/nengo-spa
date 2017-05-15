import nengo
from nengo.config import Default
from nengo.exceptions import ValidationError
from nengo.params import Parameter
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
    def __init__(self, name, sp_input=True, sp_output=True, *args, **kwargs):
        super(TranscodeFunctionParam, self).__init__(name, *args, **kwargs)
        self.sp_input = sp_input
        self.sp_output = sp_output

    def coerce(self, obj, fn):
        fn = super(TranscodeFunctionParam, self).coerce(obj, fn)

        if fn is None:
            return fn
        elif callable(fn):
            return self.coerce_callable(obj, fn)
        elif not self.sp_input and is_string(fn):
            return fn
        else:
            raise ValidationError("Invalid output type {!r}".format(
                type(fn)), attr=self.name, obj=obj)

    def coerce_callable(self, obj, fn):
        t = 0.
        if self.sp_input:
            args = (t, SemanticPointer(obj.input_vocab.dimensions),
                    obj.input_vocab)
        elif obj.size_in is not None:
            args = (t, np.zeros(obj.size_in))
        else:
            args = (t,)

        _, invoked = checked_call(fn, *args)
        if not invoked:
            if self.sp_input:
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
        'input_vocab', optional=False, default=None, readonly=True)
    output_vocab = VocabularyOrDimParam(
        'output_vocab', optional=False, default=None, readonly=True)

    def __init__(
            self, function=Default, input_vocab=Default, output_vocab=Default,
            **kwargs):
        super(Transcode, self).__init__(**kwargs)

        # Vocabs need to be set before function which accesses vocab for
        # validation.
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.function = function

        with self:
            self.node = nengo.Node(
                TranscodeFunctionParam.to_node_output(
                    self.function, self.input_vocab, self.output_vocab),
                size_in=self.input_vocab.dimensions,
                size_out=self.output_vocab.dimensions)
            self.input = self.node
            self.output = self.node

        self.inputs = dict(default=(self.input, self.input_vocab))
        self.outputs = dict(default=(self.output, self.output_vocab))
