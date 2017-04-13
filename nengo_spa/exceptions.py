class SpaException(Exception):
    """A exception within the SPA subsystem."""


class SpaNetworkError(SpaException, ValueError):
    """An error in how SPA keeps track of networks."""


class SpaParseError(SpaException, ValueError):
    """An error encountered while parsing a SPA expression."""


class SpaTypeError(SpaException, ValueError):
    """The evaluation of types in an SPA expression was invalid."""
