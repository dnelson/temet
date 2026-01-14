""" Test numerical recipes and algorithms. """
import numpy as np
import pytest

from temet.util.match import match


def test_match():
    # cross-matching integer arrays
    N = 100

    x = np.arange(N)
    y = np.arange(N)
    rng = np.random.default_rng(424242)
    rng.shuffle(y)

    i1, i2 = match(x, y)

    assert i1.size == i2.size == N
    assert np.array_equal(y[i2],y)

def test_match_int64():
    # int64 dtypes
    N = 100
    x = np.arange(N, dtype=np.int64) + int(1e10)
    y = x[5:10].copy()

    i1, i2 = match(x, y)

    assert np.array_equal(i1, np.arange(5,10))
    assert np.array_equal(i2, np.arange(5))
    