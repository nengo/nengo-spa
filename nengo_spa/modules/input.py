import nengo
from nengo.config import Default
from nengo.exceptions import ValidationError
from nengo.params import Parameter
from nengo.utils.compat import is_string

from nengo_spa.network import Network
from nengo_spa.pointer import SemanticPointer
from nengo_spa.vocab import VocabularyOrDimParam


def make_parse_func(func, vocab):
    """Create a function that calls func and parses the output in vocab."""

    def parse_func(t):
        value = func(t)
        if is_string(value):
            value = vocab.parse(value)
        if isinstance(value, SemanticPointer):
            value = value.v
        return value

    return parse_func


class SpOutputParam(Parameter):
    def coerce(self, node, output):
        output = super(SpOutputParam, self).coerce(node, output)

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
            return vocab.parse(output).v
        else:
            raise ValueError("Invalid output type {!r}".format(type(output)))


class Input(Network):
    """A SPA network for providing external inputs to other networks."""

    sp_output = SpOutputParam(
        'sp_output', optional=False, default=None, readonly=True)
    vocab = VocabularyOrDimParam(
        'vocab', optional=False, default=None, readonly=True)

    def __init__(self, sp_output=Default, vocab=Default, **kwargs):
        super(Input, self).__init__(**kwargs)

        self.sp_output = sp_output
        self.vocab = vocab

        with self:
            self.node = nengo.Node(
                SpOutputParam.to_vector_output(self.sp_output, self.vocab),
                size_out=self.vocab.dimensions)
            self.output = self.node

        self.inputs = dict()
        self.outputs = dict(default=(self.output, self.vocab))
