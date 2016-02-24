import numpy
from numpy.testing import assert_equal, assert_raises

import theano
from theano import tensor

from blocks.bricks.lookup import LookupTable


def test_lookup_table():
    lt = LookupTable(5, 3)
    lt.allocate()

    lt.W.set_value(numpy.arange(15).reshape(5, 3).astype(theano.config.floatX))

    x = tensor.lmatrix("x")
    y = lt.apply(x)
    f = theano.function([x], [y])

    x_val = [[1, 2], [0, 3]]
    desired = numpy.array([[[3, 4, 5], [6, 7, 8]], [[0, 1, 2], [9, 10, 11]]],
                          dtype=theano.config.floatX)
    assert_equal(f(x_val)[0], desired)

    # Test get_dim
    assert_equal(lt.get_dim(lt.apply.inputs[0]), 0)
    assert_equal(lt.get_dim(lt.apply.outputs[0]), lt.dim)
    assert_raises(ValueError, lt.get_dim, 'random_name')

    # Test feedforward interface
    assert lt.input_dim == 0
    assert lt.output_dim == 3
    lt.output_dim = 4
    assert lt.output_dim == 4

    def assign_input_dim():
        lt.input_dim = 11
    assert_raises(ValueError, assign_input_dim)
    lt.input_dim = 0
