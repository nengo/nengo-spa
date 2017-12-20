class SpaException(Exception):
    """A exception within the SPA subsystem."""


class SpaActionSelectionError(SpaException):
    """An error in the usage of the SPA action selection system."""


class SpaParseError(SpaException, ValueError):
    """An error encountered while parsing a SPA expression."""


class SpaTypeError(SpaException, ValueError):
    """The evaluation of types in an SPA expression was invalid."""
