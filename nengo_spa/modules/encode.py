import nengo
from nengo.config import Default
from nengo.params import IntParam

from nengo_spa.modules.transcode import TranscodeFunctionParam
from nengo_spa.network import Network
from nengo_spa.vocab import VocabularyOrDimParam


class Encode(Network):
    """A SPA network for providing external inputs to other networks."""

    function = TranscodeFunctionParam(
        'function', sp_input=False, optional=False, default=None,
        readonly=True)
    vocab = VocabularyOrDimParam(
        'vocab', optional=False, default=None, readonly=True)
    size_in = IntParam(
        'size_in', low=0, optional=True, default=None, readonly=True)

    def __init__(
            self, function=Default, vocab=Default, size_in=Default, **kwargs):
        super(Encode, self).__init__(**kwargs)

        # Vocab and size_in need to be set before function which accesses these
        # for validation.
        self.vocab = vocab
        self.size_in = size_in
        self.function = function

        with self:
            self.node = nengo.Node(
                TranscodeFunctionParam.to_node_output(
                    self.function, output_vocab=self.vocab),
                size_in=self.size_in, size_out=self.vocab.dimensions)
            self.input = self.node
            self.output = self.node

        self.inputs = dict()
        self.outputs = dict(default=(self.output, self.vocab))

    @property
    def output_vocab(self):
        return self.vocab
