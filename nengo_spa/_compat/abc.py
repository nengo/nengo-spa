try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

assert Mapping
