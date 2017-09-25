class SpaException(Exception):
    """A exception within the SPA subsystem."""


class SpaConstructionError(SpaException):
    """An error in the construction of SPA action rules."""


class SpaParseError(SpaException, ValueError):
    """An error encountered while parsing a SPA expression."""


class SpaTypeError(SpaException, ValueError):
    """The evaluation of types in an SPA expression was invalid."""
