import numpy
import theano
from numpy.testing import assert_allclose
from six.moves import cPickle
from theano import tensor

from blocks.bricks import Linear, Softmax
from blocks.bricks.wrappers import WithExtraDims, WithAxesSwapped
from blocks.initialization import Constant


def test_with_extra_dims_ndim_gt_2():
    X = tensor.tensor4('X')
    brick = Linear(input_dim=3, output_dim=4, weights_init=Constant(1),
                   biases_init=Constant(0))
    wrapper = WithExtraDims(brick.apply)
    wrapper.initialize()
    f = theano.function([X], wrapper.apply(2, X))
    assert_allclose(
        f(numpy.ones(shape=(2, 2, 2, 3), dtype=theano.config.floatX)),
        3 * numpy.ones(shape=(2, 2, 2, 4), dtype=theano.config.floatX))


def test_with_extra_dims_ndim_leq_2():
    X = tensor.matrix('X')
    brick = Linear(input_dim=3, output_dim=4, weights_init=Constant(1),
                   biases_init=Constant(0))
    wrapper = WithExtraDims(brick.apply)
    wrapper.initialize()
    f = theano.function([X], wrapper.apply(0, X))
    assert_allclose(
        f(numpy.ones(shape=(2, 3), dtype=theano.config.floatX)),
        3 * numpy.ones(shape=(2, 4), dtype=theano.config.floatX))


def test_with_extra_dims_is_serializable():
    brick = Linear(input_dim=3, output_dim=4, weights_init=Constant(1),
                   biases_init=Constant(0))
    wrapper = WithExtraDims(brick.apply)
    wrapper.initialize()
    cPickle.loads(cPickle.dumps(wrapper))


def test_with_extra_dims_cross_entropy_2d():
    x = tensor.matrix('x')
    y = tensor.lvector('y')
    brick = Softmax()
    wrapper = WithExtraDims(brick.categorical_cross_entropy)
    f = theano.function([y, x], [wrapper.apply(0, y, x)])
    assert_allclose(
        f([0, 1, 2, 3],
           [[1, 2, 1, 2], [1, 2, 3, 4],
            [4, 3, 2, 1], [2, 2, 2, 2]])[0],
        numpy.array([ 2.00640,  2.44019 ,  2.44019 ,  1.3863]),
        rtol=1e-5)


def test_with_extra_dims_cross_entropy_3d():
    x = tensor.tensor3('x')
    y = tensor.lmatrix('y')
    brick = Softmax()
    wrapper = WithExtraDims(brick.categorical_cross_entropy)
    f = theano.function([y, x], [wrapper.apply(1, y, x)])
    assert_allclose(
        f([[0, 1], [2, 3]],
          [[[1, 2, 1, 2], [1, 2, 3, 4]],
           [[4, 3, 2, 1], [2, 2, 2, 2]]])[0],
        numpy.array([[2.0064,  2.44019 ],
               [2.44019 ,  1.3863]]),
        rtol=1e-5)


def test_withaxesswapped_dim0_dim1_neq():
    X = tensor.matrix('X')
    brick = Linear(input_dim=2, output_dim=2, weights_init=Constant(1),
                   biases_init=Constant(0))
    wrapper = WithAxesSwapped(brick.apply, 0, 1)
    wrapper.initialize()
    brick.W.set_value(
        numpy.asarray([[1, 2], [1, 1]], dtype=theano.config.floatX))
    f = theano.function([X], wrapper.apply(X))
    assert_allclose(
        f(numpy.arange(4, dtype=theano.config.floatX).reshape((2, 2))),
        numpy.array([[2, 4], [2, 5]]))


def test_withaxesswapped_dim0_dim1_eq():
    X = tensor.matrix('X')
    brick = Linear(input_dim=2, output_dim=2, weights_init=Constant(1),
                   biases_init=Constant(0))
    wrapper = WithAxesSwapped(brick.apply, 0, 0)
    wrapper.initialize()
    brick.W.set_value(
        numpy.asarray([[1, 2], [1, 1]], dtype=theano.config.floatX))
    f = theano.function([X], wrapper.apply(X))
    assert_allclose(
        f(numpy.arange(4, dtype=theano.config.floatX).reshape((2, 2))),
        numpy.array([[1, 1], [5, 7]]))


def test_withaxesswapped_is_serializable():
    brick = Linear(input_dim=2, output_dim=2, weights_init=Constant(1),
                   biases_init=Constant(0))
    wrapper = WithAxesSwapped(brick.apply, 0, 1)
    wrapper.initialize()
    cPickle.loads(cPickle.dumps(wrapper))
