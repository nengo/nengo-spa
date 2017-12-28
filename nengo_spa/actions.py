"""Implementation of action syntax."""

import nengo
import numpy as np

from nengo_spa.ast.base import Fixed, infer_types, Node
from nengo_spa.ast import dynamic
from nengo_spa.exceptions import SpaActionSelectionError
from nengo_spa.types import TScalar


class ModuleInput(object):
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

    def __init__(self, input_, type_):
        self.input = input_
        self.type = type_

    def __rrshift__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        if ActionSelection.active is None:
            infer_types(self, other)
            other.connect_to(self.input)
        else:
            return RoutedConnection(other, self)


class RoutedConnection(object):
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

    _free_floating = set()

    def __init__(self, source, sink):
        self.type = infer_types(source, sink)
        self.source = source
        self.sink = sink
        RoutedConnection._free_floating.add(self)

    def connect_to(self, sink):
        return self.source.connect_to(sink)

    def construct(self):
        return self.connect_to(self.sink.input)

    @property
    def fixed(self):
        """Whether the source provides a fixed value."""
        return isinstance(self.source, Fixed)

    def transform(self):
        """For a fixed source, returns the transform to implement the output.
        """
        assert self.fixed
        if self.type == TScalar:
            return self.source.evaluate()
        else:
            return np.atleast_2d(self.source.evaluate().v).T


class ActionSelection(object):
    """Implements an action selection system with basal ganglia and thalamus.

    The *ActionSelection* instance has to be used as context manager and each
    potential action is defined by an *ifmax* call providing an expression
    for the utility value and any number of _actions (routing of information)
    to activate when this utility value is highest of all.

    Example::

        with ActionSelection():
            ifmax(dot(state, sym.A), B >> state)
            ifmax(dot(state, sym.B), C >> state)
            ifmax(dot(state, sym.C), A >> state)

    Attributes
    ----------
    active : ActionSelection
        Class attribute providing the currently active ActionSelection
        instance (if any).
    built : bool
        Indicates whether the action selection system has been built
        successfully.
    bg : nengo.Network
        Basal ganglia network. Available after the action selection system has
        been built.
    thalamus : nengo.Network
        Thalamus network. Available after the action selection system has
        been built.
    """

    active = None

    def __init__(self):
        self.built = False
        self.bg = None
        self.thalamus = None
        self._utilities = []
        self._actions = []

    def __enter__(self):
        assert not self.built
        if ActionSelection.active is None:
            ActionSelection.active = self
        else:
            raise SpaActionSelectionError(
                "Must not nest action selection contexts.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        ActionSelection.active = None

        if exc_type is not None:
            return

        if len(RoutedConnection._free_floating) > 0:
            raise SpaActionSelectionError(
                "All actions in an action selection context must be part of "
                "an ifmax call.")

        if len(self._utilities) <= 0:
            return

        self.bg = dynamic.BasalGangliaRealization(len(self._utilities))
        self.thalamus = dynamic.ThalamusRealization(len(self._utilities))
        self.thalamus.connect_bg(self.bg)

        for index, utility in enumerate(self._utilities):
            self.bg.connect_input(utility, index=index)

        for index, action in enumerate(self._actions):
            for effect in action:
                if effect.fixed:
                    self.thalamus.connect_fixed(
                        index, effect.sink.input, transform=effect.transform())
                else:
                    self.thalamus.construct_gate(
                        index, net=nengo.Network.context[-1])
                    channel = self.thalamus.construct_channel(
                        effect.sink.input, effect.type)
                    effect.connect_to(channel.input)
                    self.thalamus.connect_gate(index, channel)

        self.built = True

    def add_action(self, *actions):
        assert ActionSelection.active is self
        utility = nengo.Node(size_in=1)
        self._utilities.append(utility)
        self._actions.append(actions)
        RoutedConnection._free_floating.difference_update(actions)
        return utility


def ifmax(condition, *actions):
    """Defines a potential action within an `ActionSelection` context.

    Parameters
    ----------
    condition : nengo_spa.ast.base.Node
        The utility value for the given actions.
    actions : sequence of `RoutedConnection`
        The actions to activate if the given utility is the highest.

    Returns
    -------
    NengoObject
        Nengo object that can be connected to, to provide additional input to
        the utility value.
    """
    utility = ActionSelection.active.add_action(*actions)
    condition.connect_to(utility)
    return utility
