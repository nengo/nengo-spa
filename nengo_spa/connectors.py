import weakref

import nengo
import numpy as np

from nengo_spa.ast.base import Fixed, Node, infer_types
from nengo_spa.ast.dynamic import ModuleOutput
from nengo_spa.ast.symbolic import FixedScalar
from nengo_spa.exceptions import SpaTypeError
from nengo_spa.typechecks import is_number
from nengo_spa.types import TScalar, TVocabulary


def as_ast_node(obj):
    if isinstance(obj, Node):
        return obj
    elif is_number(obj):
        return FixedScalar(obj)
    elif isinstance(obj, nengo.Network) and hasattr(obj, "output"):
        output = obj.output
    else:
        output = obj

    try:
        # Trying to create weakref on access of weak dict can raise TypeError
        vocab = output_vocab_registry[output]
    except (KeyError, TypeError):
        if getattr(output, "size_out", 0) == 1:
            return ModuleOutput(output, TScalar)
        err = SpaTypeError("{} was not registered as a SPA output.".format(output))
        err.__suppress_context__ = True
        raise err
    finally:
        err = None  # prevent cyclic reference, traceback might reference this

    if vocab is None:
        return ModuleOutput(output, TScalar)
    else:
        return ModuleOutput(output, TVocabulary(vocab))


def as_sink(obj):
    if isinstance(obj, nengo.Network) and hasattr(obj, "input"):
        input_ = obj.input
    else:
        input_ = obj

    try:
        # Trying to create weakref on access of weak dict can raise TypeError
        vocab = input_vocab_registry[input_]
    except (KeyError, TypeError):
        err = SpaTypeError("{} was not registered as a SPA input.".format(input_))
        err.__suppress_context__ = True
        raise err
    finally:
        err = None  # prevent cyclic reference, traceback might reference this

    if vocab is None:
        return ModuleInput(input_, TScalar)
    else:
        return ModuleInput(input_, TVocabulary(vocab))


class ModuleInput:
    """Represents the input to a module with type information.

    Supports the ``>>`` operator to provide input from an AST node. It will
    create a simple connection by default, but a `.RoutedConnection` instance
    if used within the context of an `.ActionSelection` instance.

    Parameters
    ----------
    input_ : NengoObject
        Nengo object that retrieves the module input.
    type_ : nengo_spa.types.Type
        Type of the input.
    """

    routed_mode = False

    def __init__(self, input_, type_):
        self.input = input_
        self.type = type_

    def __rrshift__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        if self.routed_mode:
            return RoutedConnection(other, self)
        else:
            infer_types(self, other)
            other.connect_to(self.input)


class RoutedConnection:
    """Represents a routed connection from an AST node to a `.ModuleInput`.

    A routed connection is passed through an inhibitable channel to allow the
    routed connection to be disabled.

    Parameters
    ----------
    source : nengo_spa.ast.base.Node
        The AST node providing the source signal.
    sink : ModuleInput
        The module input that the source signal should be routed to.
    """

    free_floating = set()

    def __init__(self, source, sink):
        self.type = infer_types(source, sink)
        self.source = source
        self.sink = sink
        RoutedConnection.free_floating.add(self)

    def connect_to(self, sink):
        return self.source.connect_to(sink)

    def construct(self):
        return self.connect_to(self.sink.input)

    @property
    def fixed(self):
        """Whether the source provides a fixed value."""
        return isinstance(self.source, Fixed)

    def transform(self):
        """For a fixed source, returns the transform to implement the output."""
        assert self.fixed
        if self.type == TScalar:
            return self.source.evaluate()
        else:
            return np.atleast_2d(self.source.evaluate().v).T


class SpaOperatorMixin:
    """Mixin class that implements the SPA operators.

    All operands will be converted to AST node and the implementation of the
    operator itself is delegated to the implementation provided by those nodes.
    """

    @staticmethod
    def __define_unary_op(op):
        def op_impl(self):
            return getattr(as_ast_node(self), op)()

        return op_impl

    @staticmethod
    def __define_binary_op(op):
        def op_impl(self, other):
            return getattr(as_ast_node(self), op)(as_ast_node(other))

        return op_impl

    __invert__ = __define_unary_op.__func__("__invert__")
    linv = __define_unary_op.__func__("linv")
    rinv = __define_unary_op.__func__("rinv")
    __neg__ = __define_unary_op.__func__("__neg__")

    __add__ = __define_binary_op.__func__("__add__")
    __radd__ = __define_binary_op.__func__("__radd__")
    __sub__ = __define_binary_op.__func__("__sub__")
    __rsub__ = __define_binary_op.__func__("__rsub__")
    __mul__ = __define_binary_op.__func__("__mul__")
    __rmul__ = __define_binary_op.__func__("__rmul__")
    __truediv__ = __define_binary_op.__func__("__truediv__")
    __matmul__ = __define_binary_op.__func__("__matmul__")
    __rmatmul__ = __define_binary_op.__func__("__rmatmul__")

    def __rshift__(self, other):
        return as_ast_node(self) >> as_sink(other)

    def __rrshift__(self, other):
        return as_ast_node(other) >> as_sink(self)

    dot = __define_binary_op.__func__("dot")
    rdot = __define_binary_op.__func__("rdot")

    def reinterpret(self, vocab=None):
        return as_ast_node(self).reinterpret(vocab)

    def translate(self, vocab, populate=None, keys=None, solver=None):
        return as_ast_node(self).translate(vocab, populate, keys, solver)


class ConnectorRegistry:
    """Registry associating connectors with vocabularies and enable SPA syntax.

    A connector is either an input or output to a SPA module.

    Declaring a Nengo object as connector dynamically changes the class of the
    instance to inject the `.SpaOperatorMixin` to allow the object to be used
    in SPA rules. These substituted classes will be cached to use the same
    type instance where appropriate (i.e. two Nengo objects of identical type A
    will still be of identical type B after being declared as connectors and
    B inherits from A and *SpaOperatorMixin*).
    """

    _type_cache = {}

    def __init__(self):
        self._registry = weakref.WeakKeyDictionary()

    def __contains__(self, key):
        return key in self._registry

    def __getitem__(self, key):
        return self._registry[key]

    def declare_connector(self, obj, vocab):
        """Declares a connector.

        Parameters
        ----------
        obj : nengo.base.NengoObject
            Nengo object to use as a connector.
        vocab: Vocabulary
            Vocabulary to assign to the connector.
        """
        try:
            extended_type = self._type_cache[obj.__class__]
        except KeyError:
            extended_type = type(
                "Connector<%s>" % (obj.__class__.__name__),
                (obj.__class__, SpaOperatorMixin),
                {},
            )
            self._type_cache[obj.__class__] = extended_type
        obj.__class__ = extended_type
        self._registry[obj] = vocab
        return obj


input_vocab_registry = ConnectorRegistry()
output_vocab_registry = ConnectorRegistry()
