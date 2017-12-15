from nengo_spa.exceptions import SpaTypeError


def dot(a, b):
    try:
        return a.dot(b)
    except (AttributeError, NotImplementedError):
        try:
            return b.dot(a)
        except (AttributeError, NotImplementedError):
            raise SpaTypeError()


def reinterpret(source, vocab=None):
    return source.reinterpret(vocab)


def translate(source, vocab, populate=None, keys=None, solver=None):
    return source.translate(vocab, populate, keys, solver)
