def dot(a, b):
    try:
        return a.dot(b)
    except (AttributeError, NotImplementedError):
        return b.dot(a)


def translate(source, vocab, populate=None, keys=None, solver=None):
    return source.translate(vocab, populate, keys, solver)
