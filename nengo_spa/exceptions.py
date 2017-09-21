class SpaException(Exception):
    """A exception within the SPA subsystem."""


class SpaConstructionError(SpaException):
    """An error in the construction of SPA action rules."""


class SpaNameError(NameError):
    """An error in finding a network, input, or output.

    Parameters
    ----------
    name : str
        Name that could not be found.
    kind : str
        Type of the name not being found (e.g. 'network', 'network input', ...)

    Attributes
    ----------
    name : str
        Name that could not be found.
    kind : str
        Type of the name not being found (e.g. 'network', 'network input', ...)
    """

    def __init__(self, name, kind):
        super(SpaNameError, self).__init__(
            "Could not find {kind} {name!r}.".format(kind=kind, name=name))
        self.name = name
        self.kind = kind


class SpaParseError(SpaException, ValueError):
    """An error encountered while parsing a SPA expression."""


class SpaTypeError(SpaException, ValueError):
    """The evaluation of types in an SPA expression was invalid."""
