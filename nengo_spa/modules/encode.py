import nengo
from nengo.config import Default
from nengo.exceptions import ValidationError
from nengo.params import Parameter
from nengo.utils.compat import is_string

from nengo_spa.network import Network
from nengo_spa.pointer import SemanticPointer
from nengo_spa.vocab import VocabularyOrDimParam


class SpArrayExtractor(object):
    # TODO is there a better place for this class?
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, value):
        if is_string(value):
            value = self.vocab.parse(value)
        if isinstance(value, SemanticPointer):
            value = value.v
        return value


def make_parse_func(fn, vocab):
    """Create a function that calls func and parses the output in vocab."""

    extractor = SpArrayExtractor(vocab)

    def parse_func(t):
        return extractor(fn(t))

    return parse_func


class EncodeFunctionParam(Parameter):
    def coerce(self, node, output):
        output = super(EncodeFunctionParam, self).coerce(node, output)

        if output is None:
            return output
        elif isinstance(output, nengo.Process):
            raise NotImplementedError()
        elif callable(output):
            return output
        elif is_string(output):
            return output
        else:
            raise ValidationError("Invalid output type {!r}".format(
                type(output)), attr=self.name, obj=node)

    @classmethod
    def to_vector_output(cls, output, vocab):
        if output is None:
            return None
        elif isinstance(output, nengo.Process):
            raise NotImplementedError()
        elif callable(output):
            return make_parse_func(output, vocab)
        elif is_string(output):
            return SpArrayExtractor(vocab)(output)
        else:
            raise ValueError("Invalid output type {!r}".format(type(output)))


class Encode(Network):
    """A SPA network for providing external inputs to other networks."""

    function = EncodeFunctionParam(
        'function', optional=False, default=None, readonly=True)
    vocab = VocabularyOrDimParam(
        'vocab', optional=False, default=None, readonly=True)

    def __init__(self, sp_output=Default, vocab=Default, **kwargs):
        super(Encode, self).__init__(**kwargs)

        self.function = sp_output
        self.vocab = vocab

        with self:
            self.node = nengo.Node(
                EncodeFunctionParam.to_vector_output(
                    self.function, self.vocab),
                size_out=self.vocab.dimensions)
            self.output = self.node

        self.inputs = dict()
        self.outputs = dict(default=(self.output, self.vocab))
