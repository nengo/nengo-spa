from nengo.network import Network as NengoNetwork
import numpy as np

from nengo_spa.ast.base import Fixed, infer_types, Node
from nengo_spa.ast import dynamic
from nengo_spa.types import TScalar


class ModuleInput(object):
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
        return isinstance(self.source, Fixed)

    def transform(self):
        assert self.fixed
        if self.type == TScalar:
            return self.source.evaluate()
        else:
            return np.atleast_2d(self.source.evaluate().v).T


class ActionSelection(object):
    active = None

    def __init__(self):
        self.built = False
        self.bg = None
        self.thalamus = None
        self.utilities = []
        self.actions = []

    def __enter__(self):
        assert not self.built
        if ActionSelection.active is None:
            ActionSelection.active = self
        else:
            raise RuntimeError()  # FIXME better error
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        ActionSelection.active = None

        if exc_type is not None:
            return

        if len(RoutedConnection.free_floating) > 0:
            raise RuntimeError()  # FIXME better error

        if len(self.utilities) <= 0:
            return

        self.bg = dynamic.BasalGangliaRealization(len(self.utilities))
        self.thalamus = dynamic.ThalamusRealization(len(self.utilities))
        self.thalamus.connect_bg(self.bg)

        for index, utility in enumerate(self.utilities):
            self.bg.connect_input(utility.output, index=index)

        for index, action in enumerate(self.actions):
            for effect in action:
                if effect.fixed:
                    self.thalamus.connect_fixed(
                        index, effect.sink.input, transform=effect.transform())
                else:
                    self.thalamus.construct_gate(
                        index, net=NengoNetwork.context[-1])
                    channel = self.thalamus.construct_channel(
                        effect.sink.input, effect.type)
                    effect.connect_to(channel.input)
                    self.thalamus.connect_gate(index, channel)

        self.built = True

    def add_action(self, condition, *actions):
        assert ActionSelection.active is self
        utility = dynamic.ScalarRealization()  # FIXME should be node
        self.utilities.append(utility)
        self.actions.append(actions)
        RoutedConnection.free_floating.difference_update(actions)
        return utility


def ifmax(condition, *actions):
    utility = ActionSelection.active.add_action(condition, *actions)
    condition.connect_to(utility.input)
    return utility
