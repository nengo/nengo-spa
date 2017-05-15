import nengo
from nengo.config import Default
from nengo.params import Parameter
from nengo.utils.compat import is_string
from nengo.utils.stdlib import checked_call

from nengo_spa.modules.encode import make_parse_func
from nengo_spa.modules.decode import make_sp_func
from nengo_spa.network import Network
from nengo_spa.pointer import SemanticPointer
from nengo_spa.vocab import VocabularyOrDimParam


def make_func(fn, input_vocab, output_vocab):
    return make_parse_func(make_sp_func(fn, input_vocab), output_vocab)


class TranscodeFunctionParam(Parameter):
    def coerce(self, node, output):
        output = super(TranscodeFunctionParam, self).coerce(node, output)

        if output is None:
            return output
        elif isinstance(output, nengo.Process):
            raise NotImplementedError()
        elif callable(output):
            return self.coerce_callable(node, output)
        else:
            raise ValidationError("Invalid output type {!r}".format(
                type(output)), attr=self.name, obj=node)

    def coerce_callable(self, obj, value):
        _, invoked = checked_call(
            value, 0., SemanticPointer(obj.input_vocab.dimensions),
            obj.input_vocab)
        if not invoked:
            raise ValidationError(
                "Transcode function %r is expected to accept exactly 3 "
                "arguments: time as a float, a SemanticPointer, and a "
                "Vocabulary.", attr=self.name, obj=node)
        return value

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
                make_func(self.function, self.input_vocab, self.output_vocab),
                size_in=self.input_vocab.dimensions,
                size_out=self.output_vocab.dimensions)
            self.input = self.node
            self.output = self.node

        self.inputs = dict(default=(self.input, self.input_vocab))
        self.outputs = dict(default=(self.output, self.output_vocab))
