import numpy
import theano
from numpy.testing import assert_allclose
from theano import tensor

from blocks.theano_expressions import l2_norm, hessian_times_vector


def test_l2_norm():
    assert_allclose(l2_norm([2]).eval(), 2.0)
    assert_allclose(l2_norm([3, 4]).eval(), 5.0)
    assert_allclose(l2_norm([3, [1, 2]]).eval(), 14.0 ** 0.5)
    assert_allclose(
        l2_norm([3, [1, 2], [[1, 2], [3, 4]]]).eval(), 44.0 ** 0.5)
    assert_allclose(
        l2_norm([3, [1, 2], [[1, 2], [3, 4]]], squared=True).eval(), 44.0)


def test_hessian_times_vector():
    x_y = tensor.vector('x_y')
    x, y = x_y[0], x_y[1]
    # The Hessian of this should be the identity
    c = 0.5 * (x ** 2 + y ** 2)
    g = tensor.grad(c, x_y)

    v = tensor.vector('v')
    Hv = hessian_times_vector(g, x_y, v)
    Hv_rop = hessian_times_vector(g, x_y, v, r_op=True)

    f = theano.function([x_y, v], Hv)
    f_rop = theano.function([x_y, v], Hv_rop)

    x_y_val = numpy.random.rand(2).astype(theano.config.floatX)
    v_val = numpy.random.rand(2).astype(theano.config.floatX)

    assert_allclose(v_val, f(x_y_val, v_val))
    assert_allclose(v_val, f_rop(x_y_val, v_val))
