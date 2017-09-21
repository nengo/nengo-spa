from collections import deque


class Peekable(object):
    """Wrapper for generator to allow peeking at coming elements.

    Parameters
    ----------
    generator : generator
        Generator to wrap.
    maxlen : int, optional
        Maximum number of items that can be looked ahead. If *None*, unlimited
        look-ahead is allowed.

    Attributes
    ----------
    generator : generator
        The wrapped generator.
    """

    def __init__(self, generator, maxlen=None):
        self.generator = generator
        self._buffer = deque(maxlen=maxlen)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._buffer) > 0:
            return self._buffer.popleft()
        else:
            return next(self.generator)

    def next(self):
        return self.__next__()

    def peek(self, n=1):
        """Looks *n* items ahead.

        Returns
        -------
        list
            List of coming items.
        """
        assert self._buffer.maxlen is None or n <= self._buffer.maxlen
        try:
            while len(self._buffer) < n:
                self._buffer.append(next(self.generator))
        except StopIteration:
            pass
        return [self._buffer[i] for i in range(min(n, len(self._buffer)))]
