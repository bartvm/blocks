import numpy
from numpy.testing import assert_equal

import theano
from theano import tensor

from blocks.bricks.lookup import LookupTable

floatX = theano.config.floatX


def test_lookup_table():
    lt = LookupTable(5, 3)
    lt.allocate()

    lt.W.set_value(numpy.arange(15).reshape(5, 3).astype(floatX))

    x = tensor.lmatrix("x")
    y = lt.apply(x)
    f = theano.function([x], [y])

    x_val = [[1, 2], [0, 3]]
    desired = numpy.array([[[3, 4, 5], [6, 7, 8]], [[0, 1, 2], [9, 10, 11]]],
                          dtype=floatX)
    assert_equal(f(x_val)[0], desired)
