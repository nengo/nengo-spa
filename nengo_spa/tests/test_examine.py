import re

import numpy as np
from numpy.testing import assert_almost_equal

from nengo_spa.examine import pairs, similarity, text
from nengo_spa.vocabulary import Vocabulary


def test_similarity(rng):
    v = Vocabulary(64, max_similarity=0.1, pointer_gen=rng)
    v.populate(
        """
        A;
        B;
        C = .7 * A + .3 * B;
        D = C.normalized()
    """
    )

    s = similarity(v["A"].v, v)
    assert_almost_equal(s[0], 1.0)
    assert s[1] < 0.1
    assert_almost_equal(s[2], 0.7 + 0.3 * np.dot(v["A"].v, v["B"].v))

    sn = similarity(v["A"].v, v, normalize=True)
    assert_almost_equal(s[3], sn[2])

    data = np.array([v["A"].v, v["B"].v])
    s = similarity(data, v)
    assert_almost_equal(s[0, 0], 1.0)
    assert_almost_equal(s[1, 1], 1.0)
    assert s[0, 1] < 0.1
    assert s[1, 0] < 0.1

    s = similarity(data, v.parse_n("A", "B"))
    assert_almost_equal(s[0, 0], 1.0)
    assert_almost_equal(s[1, 1], 1.0)
    assert s[0, 1] < 0.1
    assert s[1, 0] < 0.1


def test_pairs():
    v = Vocabulary(64)
    v.populate("A; B; C")
    actual = pairs(v)
    expected = {"A*B", "A*C", "B*C"}
    assert actual == expected


def test_text(rng):
    v = Vocabulary(64, pointer_gen=rng)
    v.populate("A; B; C; D; E; F")
    x = v.parse("A+B+C")
    y = v.parse("-D-E-F")
    ptr = r"-?[01]\.[0-9]{2}[A-F]"
    assert re.match(";".join([ptr] * 3), text(x, v))
    assert re.match(";".join([ptr] * 2), text(x, v, maximum_count=2))
    assert re.match(ptr, text(x, v, maximum_count=1))
    assert len(text(x, v, maximum_count=10).split(";")) <= 10
    assert re.match(";".join([ptr] * 4), text(x, v, minimum_count=4))
    assert re.match(
        ";".join([ptr.replace("F", "C")] * 3),
        text(x, v, minimum_count=4, terms=["A", "B", "C"]),
    )

    assert re.match(ptr, text(y, v, threshold=0.6))
    assert text(y, v, minimum_count=None, threshold=0.6) == ""

    assert text(x, v, join=",") == text(x, v).replace(";", ",")
    assert re.match(";".join([ptr] * 2), text(x, v, normalize=True))

    assert text([0] * 64, v) == "0.00F"
    assert text(v["D"].v, v) == "1.00D"
