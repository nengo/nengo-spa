from collections.abc import Iterable

import numpy as np


def is_array(obj):
    # np.generic allows us to return true for scalars as well as true arrays
    return isinstance(obj, (np.ndarray, np.generic))


def is_array_like(obj):
    # While it's possible that there are some iterables other than list/tuple
    # that can be made into arrays, it's very likely that those arrays
    # will have dtype=object, which is likely to cause unexpected issues.
    return is_array(obj) or is_number(obj) or isinstance(obj, (list, tuple))


def is_float(obj):
    return isinstance(obj, (float, np.float))


def is_integer(obj):
    return isinstance(obj, (int, np.integer))


def is_iterable(obj):
    if isinstance(obj, np.ndarray):
        return obj.ndim > 0
    else:
        return isinstance(obj, Iterable)


def is_number(obj):
    return is_integer(obj) or is_float(obj)
