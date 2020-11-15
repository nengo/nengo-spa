import nengo
import numpy as np
from nengo.config import Default
from nengo.exceptions import ValidationError
from nengo.params import IntParam, Parameter
from nengo.utils.stdlib import checked_call

from nengo_spa.ast.symbolic import PointerSymbol
from nengo_spa.network import Network
from nengo_spa.semantic_pointer import SemanticPointer
from nengo_spa.vocabulary import VocabularyOrDimParam


class SpArrayExtractor:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, value):
        if isinstance(value, PointerSymbol):
            value = value.expr
        if isinstance(value, str):
            value = self.vocab.parse(value)
        if isinstance(value, SemanticPointer):
            value = value.v
        return value


def make_sp_func(fn, vocab):
    def sp_func(t, v):
        return fn(t, SemanticPointer(v, vocab=vocab))

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

        pointer_cls = (SemanticPointer, PointerSymbol)

        if fn is None:
            return fn
        elif callable(fn):
            return self.coerce_callable(obj, fn)
        elif not obj.input_vocab and isinstance(fn, (str, pointer_cls)):
            return fn
        else:
            raise ValidationError(
                "Invalid output type {!r}".format(type(fn)), attr=self.name, obj=obj
            )

    def coerce_callable(self, obj, fn):
        t = 0.0
        if obj.input_vocab is not None:
            args = (
                t,
                SemanticPointer(
                    np.zeros(obj.input_vocab.dimensions), vocab=obj.input_vocab
                ),
            )
        elif obj.size_in is not None:
            args = (t, np.zeros(obj.size_in))
        else:
            args = (t,)

        _, invoked = checked_call(fn, *args)
        fn(*args)
        if not invoked:
            if obj.input_vocab is not None:
                raise ValidationError(
                    "Transcode function %r is expected to accept exactly 2 "
                    "arguments: time as a float, and a SemanticPointer",
                    attr=self.name,
                    obj=obj,
                )
            else:
                raise ValidationError(
                    "Transcode function %r is expected to accept exactly 1 or "
                    "2 arguments: time as a float, and optionally the input "
                    "data as NumPy array.",
                    attr=self.name,
                    obj=obj,
                )
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
        elif isinstance(fn, (str, SemanticPointer, PointerSymbol)):
            return SpArrayExtractor(output_vocab)(fn)
        else:
            raise ValueError("Invalid output type {!r}".format(type(fn)))


class Transcode(Network):
    """Transcode from, to, and between Semantic Pointers.

    This can thought of the equivalent of a `nengo.Node` for Semantic Pointers.

    Either the *input_vocab* or the *output_vocab* argument must not be *None*.
    (If you want both arguments to be *None*, use a normal `nengo.Node`.)
    Which one of the parameters in the pairs *input_vocab/size_in* and
    *output_vocab/size_out* is not set to *None*, determines whether a Semantic
    Pointer input/output or a normal vector input/output is expected.

    Parameters
    ----------
    function : func, optional (Default: None)
        Function that transforms the input Semantic Pointer to an output
        Semantic Pointer. The function signature depends on *input_vocab*:

        * If *input_vocab* is *None*, the allowed signatures are the same as
          for a `nengo.Node`. Either ``function(t)`` or ``function(t, x)``
          where *t* (float) is the current simulation time and *x* (NumPy
          array) is the current input to transcode with size *size_in*.
        * If *input_vocab* is not *None*, the signature has to be
          ``function(t, sp)`` where *t* (float) is the current simulation time
          and *sp* (`.SemanticPointer`) is the current Semantic Pointer input.
          The associated vocabulary can be obtained via ``sp.vocab``.

        The allowed function return value depends on *output_vocab*:

        * If *output_vocab* is *None*, the return value must be a NumPy array
          (or equivalent) of size *size_out* or *None* (i.e. no return value)
          if *size_out* is *None*.
        * If *output_vocab* is not *None*, the return value can be either of:
          NumPy array, `.SemanticPointer` instance, or an SemanticPointer
          expression or symbolic expression as string that gets parsed with
          the *output_vocab*.
    input_vocab : Vocabulary, optional (Default: None)
        Input vocabulary. Mutually exclusive with *size_in*.
    output_vocab : Vocabulary, optional (Default: None)
        Output vocabulary. Mutually exclusive with *size_out*.
    size_in : int, optional (Default: None)
        Input size. Mutually exclusive with *input_vocab*.
    size_out : int, optional (Default: None)
        Output size. Mutually exclusive with *output_vocab*.
    **kwargs : dict
        Additional keyword arguments passed to `nengo_spa.Network`.

    Attributes
    ----------
    input : nengo.Node
        Input.
    output : nengo.Node
        Output.
    """

    function = TranscodeFunctionParam(
        "function", optional=True, default=None, readonly=True
    )
    input_vocab = VocabularyOrDimParam(
        "input_vocab", optional=True, default=None, readonly=True
    )
    output_vocab = VocabularyOrDimParam(
        "output_vocab", optional=True, default=None, readonly=True
    )
    size_in = IntParam("size_in", optional=True, default=None, readonly=True)
    size_out = IntParam("size_out", optional=True, default=None, readonly=True)

    def __init__(
        self,
        function=Default,
        input_vocab=Default,
        output_vocab=Default,
        size_in=Default,
        size_out=Default,
        **kwargs
    ):
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
                "Pointer, use a basic nengo.Node instead.",
                self,
            )
        if self.input_vocab is not None and self.size_in is not None:
            raise ValidationError(
                "The input_vocab and size_in arguments are mutually " "exclusive.",
                "size_in",
                self,
            )
        if self.output_vocab is not None and self.size_out is not None:
            raise ValidationError(
                "The output_vocab and size_out arguments are mutually " "exclusive.",
                "size_in",
                self,
            )

        self.function = function

        node_size_in = (
            self.input_vocab.dimensions
            if self.input_vocab is not None
            else self.size_in
        )
        node_size_out = (
            self.output_vocab.dimensions
            if self.output_vocab is not None
            else self.size_out
        )
        if self.function is None:
            if node_size_in is None:
                node_size_in = self.output_vocab.dimensions
            node_size_out = None

        with self:
            self.node = nengo.Node(
                TranscodeFunctionParam.to_node_output(
                    self.function, self.input_vocab, self.output_vocab
                ),
                size_in=node_size_in,
                size_out=node_size_out,
            )
            self.input = self.node
            self.output = self.node

        if self.input_vocab is not None:
            self.declare_input(self.input, self.input_vocab)
        if self.output_vocab is not None:
            self.declare_output(self.output, self.output_vocab)
