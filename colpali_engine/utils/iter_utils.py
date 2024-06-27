import sys


def islice(iterable, *args):
    """
    Yield a slice of an iterable.
    >>> islice('ABCDEFG', 2) → A B
    >>> islice('ABCDEFG', 2, 4) → C D
    >>> islice('ABCDEFG', 2, None) → C D E F G
    >>> islice('ABCDEFG', 0, None, 2) → A C E G
    """
    s = slice(*args)
    start, stop, step = s.start or 0, s.stop or sys.maxsize, s.step or 1
    it = iter(range(start, stop, step))
    try:
        nexti = next(it)
    except StopIteration:
        # Consume *iterable* up to the *start* position.
        for i, element in zip(range(start), iterable):
            pass
        return
    try:
        for i, element in enumerate(iterable):
            if i == nexti:
                yield element
                nexti = next(it)
    except StopIteration:
        # Consume to *stop*.
        for i, element in zip(range(i + 1, stop), iterable):
            pass


def batched(iterable, n: int):
    """
    Yield batches of n elements from an iterable.
    >>> batched('ABCDEFG', 3) → ABC DEF G
    """
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch
