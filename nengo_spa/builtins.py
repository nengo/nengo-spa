def dot(a, b):
    result = NotImplemented
    if hasattr(a, 'dot'):
        result = a.dot(b)
    if result is NotImplemented and hasattr(b, 'dot'):
        result = b.dot(a)
    if result is NotImplemented:
        raise TypeError(
            "'dot' not supported between instances of '{}' and '{}'".format(
                type(a), type(b)))
    return result


def reinterpret(source, vocab=None):
    if hasattr(source, 'reinterpret'):
        return source.reinterpret(vocab)
    else:
        raise TypeError("bad operand type for 'reinterpret'")


def translate(source, vocab, populate=None, keys=None, solver=None):
    if hasattr(source, 'translate'):
        return source.translate(vocab, populate, keys, solver)
    else:
        raise TypeError("bad operand type for 'translate'")
