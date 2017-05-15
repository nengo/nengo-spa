import nengo
from nengo.config import Default
from nengo.params import IntParam

from nengo_spa.modules.transcode import TranscodeFunctionParam
from nengo_spa.network import Network
from nengo_spa.vocab import VocabularyOrDimParam


class Decode(Network):
    function = TranscodeFunctionParam(
        'function', sp_output=False, optional=False, default=None,
        readonly=True)
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
                TranscodeFunctionParam.to_node_output(
                    self.function, input_vocab=self.vocab),
                size_in=self.vocab.dimensions, size_out=self.size_out)
            self.input = self.node
            self.output = self.node

        self.inputs = dict(default=(self.input, self.vocab))
        self.outputs = dict()

    @property
    def input_vocab(self):
        return self.vocab
