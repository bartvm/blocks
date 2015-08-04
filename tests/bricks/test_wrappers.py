import numpy
import theano
from numpy.testing import assert_allclose
from six.moves import cPickle
from theano import tensor
from blocks.bricks import Linear, Softmax
from blocks.bricks.wrappers import WithExtraDims
from blocks.initialization import Constant


class LinearWithExtraDims(Linear):
    decorators = [WithExtraDims()]


class SoftmaxWithExtraDims(Softmax):
    decorators = [WithExtraDims()]


def test_with_extra_dims_ndim_gt_2():
    X = tensor.tensor4('X')
    brick = LinearWithExtraDims(
        input_dim=3, output_dim=4,
        weights_init=Constant(1), biases_init=Constant(0))
    brick.initialize()
    f = theano.function([X], brick.apply(X, extra_ndim=2))
    assert_allclose(
        f(numpy.ones(shape=(2, 2, 2, 3), dtype=theano.config.floatX)),
        3 * numpy.ones(shape=(2, 2, 2, 4), dtype=theano.config.floatX))


def test_with_extra_dims_ndim_leq_2():
    X = tensor.matrix('X')
    brick = LinearWithExtraDims(
        input_dim=3, output_dim=4,
        weights_init=Constant(1), biases_init=Constant(0))
    brick.initialize()
    f = theano.function([X], brick.apply(X, extra_ndim=0))
    assert_allclose(
        f(numpy.ones(shape=(2, 3), dtype=theano.config.floatX)),
        3 * numpy.ones(shape=(2, 4), dtype=theano.config.floatX))


def test_with_extra_dims_is_serializable():
    brick = LinearWithExtraDims(
        input_dim=3, output_dim=4,
        weights_init=Constant(1), biases_init=Constant(0))
    brick.initialize()
    cPickle.loads(cPickle.dumps(brick))


def test_with_extra_dims_cross_entropy_2d():
    x = tensor.matrix('x')
    y = tensor.lvector('y')
    brick = SoftmaxWithExtraDims()
    f = theano.function(
        [y, x], [brick.categorical_cross_entropy(y, x, extra_ndim=0)])
    assert_allclose(
        f([0, 1, 2, 3],
          [[1, 2, 1, 2], [1, 2, 3, 4],
           [4, 3, 2, 1], [2, 2, 2, 2]])[0],
        numpy.array([2.00640, 2.44019, 2.44019, 1.3863]),
        rtol=1e-5)


def test_with_extra_dims_cross_entropy_3d():
    x = tensor.tensor3('x')
    y = tensor.lmatrix('y')
    brick = SoftmaxWithExtraDims()
    f = theano.function(
        [y, x], [brick.categorical_cross_entropy(y, x, extra_ndim=1)])
    assert_allclose(
        f([[0, 1], [2, 3]],
          [[[1, 2, 1, 2], [1, 2, 3, 4]],
           [[4, 3, 2, 1], [2, 2, 2, 2]]])[0],
        numpy.array([[2.0064, 2.44019],
                     [2.44019, 1.3863]]),
        rtol=1e-5)
