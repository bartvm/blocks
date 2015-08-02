import numbers

import numpy
import theano
from numpy.testing import assert_equal, assert_allclose, assert_raises

from blocks.initialization import Constant, IsotropicGaussian, Sparse
from blocks.initialization import Uniform, Orthogonal


def test_constant():
    def check_constant(const, shape, ground_truth):
        # rng unused, so pass None.
        init = Constant(const).generate(None, ground_truth.shape)
        assert ground_truth.dtype == theano.config.floatX
        assert ground_truth.shape == init.shape
        assert_equal(ground_truth, init)

    # Test scalar init.
    yield (check_constant, 5, (5, 5),
           5 * numpy.ones((5, 5), dtype=theano.config.floatX))
    # Test broadcasting.
    yield (check_constant, [1, 2, 3], (7, 3),
           numpy.array([[1, 2, 3]] * 7, dtype=theano.config.floatX))
    yield (check_constant, numpy.array([[1], [2], [3]]), (3, 2),
           numpy.array([[1, 1], [2, 2], [3, 3]], dtype=theano.config.floatX))


def test_gaussian():
    rng = numpy.random.RandomState(1)

    def check_gaussian(rng, mean, std, shape):
        weights = IsotropicGaussian(std, mean).generate(rng, shape)
        assert weights.shape == shape
        assert weights.dtype == theano.config.floatX
        assert_allclose(weights.mean(), mean, atol=1e-2)
        assert_allclose(weights.std(), std, atol=1e-2)
    yield check_gaussian, rng, 0, 1, (500, 600)
    yield check_gaussian, rng, 5, 3, (600, 500)


def test_uniform():
    rng = numpy.random.RandomState(1)

    def check_uniform(rng, mean, width, std, shape):
        weights = Uniform(mean=mean, width=width,
                          std=std).generate(rng, shape)
        assert weights.shape == shape
        assert weights.dtype == theano.config.floatX
        assert_allclose(weights.mean(), mean, atol=1e-2)
        if width is not None:
            std_ = width / numpy.sqrt(12)
        else:
            std_ = std
        assert_allclose(std_, weights.std(), atol=1e-2)
    yield check_uniform, rng, 0, 0.05, None, (500, 600)
    yield check_uniform, rng, 0, None, 0.001, (600, 500)
    yield check_uniform, rng, 5, None, 0.004, (700, 300)

    assert_raises(ValueError, Uniform, 0, 1, 1)


def test_sparse():
    rng = numpy.random.RandomState(1)

    def check_sparse(rng, num_init, weights_init, sparse_init, shape, total):
        weights = Sparse(num_init=num_init, weights_init=weights_init,
                         sparse_init=sparse_init).generate(rng, shape)
        assert weights.shape == shape
        assert weights.dtype == theano.config.floatX
        if sparse_init is None:
            if isinstance(num_init, numbers.Integral):
                assert (numpy.count_nonzero(weights) <=
                        weights.size - num_init * weights.shape[0])
            else:
                assert (numpy.count_nonzero(weights) <=
                        weights.size - num_init * weights.shape[1])
        if total is not None:
            assert numpy.sum(weights) == total

    yield check_sparse, rng, 5, Constant(1.), None, (10, 10), None
    yield check_sparse, rng, 0.5, Constant(1.), None, (10, 10), None
    yield check_sparse, rng, 0.5, Constant(1.), Constant(1.), (10, 10), None
    yield check_sparse, rng, 3, Constant(1.), None, (10, 10), 30
    yield check_sparse, rng, 3, Constant(0.), Constant(1.), (10, 10), 70
    yield check_sparse, rng, 0.3, Constant(1.), None, (10, 10), 30
    yield check_sparse, rng, 0.3, Constant(0.), Constant(1.), (10, 10), 70


def test_orthogonal():
    rng = numpy.random.RandomState(1)

    def check_orthogonal(rng, shape, scale=1.0):
        W = Orthogonal(scale).generate(rng, shape)

        assert W.shape == shape

        # For square matrices the following to should
        # be diagonal. For non-square matrices, we relax
        # a bit.
        WWT = numpy.dot(W, W.T)
        WTW = numpy.dot(W.T, W)

        atol = 0.2

        # Sanity check, just to be save
        assert WWT.shape == (shape[0], shape[0])
        assert WTW.shape == (shape[1], shape[1])

        # Diagonals ~= 1. ?
        assert_allclose(numpy.diag(WWT), scale ** 2, atol=atol)
        assert_allclose(numpy.diag(WTW), scale ** 2, atol=atol)

        # Non-diagonal ~= 0. ?
        WWT_residum = WWT - numpy.eye(shape[0]) * scale ** 2
        WTW_residum = WTW - numpy.eye(shape[1]) * scale ** 2

        assert_allclose(WWT_residum, 0., atol=atol)
        assert_allclose(WTW_residum, 0., atol=atol)

    yield check_orthogonal, rng, (50, 50)
    yield check_orthogonal, rng, (50, 51)
    yield check_orthogonal, rng, (51, 50)
    yield check_orthogonal, rng, (50, 50), .5
    yield check_orthogonal, rng, (50, 51), .5
    yield check_orthogonal, rng, (51, 50), .5
