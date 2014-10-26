import unittest
import itertools
import numpy
from numpy.testing import assert_allclose
import theano
from theano import tensor
from blocks.bricks import Recurrent, GatedRecurrent, Tanh
from blocks.initialization import Constant, IsotropicGaussian


floatX = theano.config.floatX


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


class TestRecurrent(unittest.TestCase):

    def setUp(self):
        self.simple = Recurrent(dim=3, weights_init=Constant(2),
                                activation=Tanh())
        self.simple.initialize()

    def test_one_step(self):
        h0 = tensor.matrix('h0')
        x = tensor.matrix('x')
        mask = tensor.vector('mask')
        h1 = self.simple.apply(x, h0, mask=mask, one_step=True)
        next_h = theano.function(inputs=[h0, x, mask], outputs=[h1])

        h0_val = 0.1 * numpy.array([[1, 1, 0], [0, 1, 1]],
                                   dtype=floatX)
        x_val = 0.1 * numpy.array([[1, 2, 3], [4, 5, 6]],
                                  dtype=floatX)
        mask_val = numpy.array([1, 0]).astype(floatX)
        h1_val = numpy.tanh(h0_val.dot(2 * numpy.ones((3, 3))) + x_val)
        h1_val = mask_val[:, None] * h1_val + (1 - mask_val[:, None]) * h0_val
        assert_allclose(h1_val, next_h(h0_val, x_val, mask_val)[0])

    def test_many_steps(self):
        x = tensor.tensor3('x')
        mask = tensor.matrix('mask')
        h = self.simple.apply(x, mask=mask)
        calc_h = theano.function(inputs=[x, mask], outputs=[h])

        x_val = 0.1 * numpy.asarray(list(itertools.permutations(range(4))),
                                    dtype=floatX)
        x_val = numpy.ones((24, 4, 3),
                           dtype=floatX) * x_val[..., None]
        mask_val = numpy.ones((24, 4), dtype=floatX)
        mask_val[12:24, 3] = 0
        h_val = numpy.zeros((25, 4, 3), dtype=floatX)
        for i in range(1, 25):
            h_val[i] = numpy.tanh(h_val[i - 1].dot(
                2 * numpy.ones((3, 3))) + x_val[i - 1])
            h_val[i] = (mask_val[i - 1, :, None] * h_val[i] +
                        (1 - mask_val[i - 1, :, None]) * h_val[i - 1])
        h_val = h_val[1:]
        assert_allclose(h_val, calc_h(x_val, mask_val)[0])


class TestGatedRecurrent(unittest.TestCase):

    def setUp(self):
        self.gated = GatedRecurrent(
            dim=3, weights_init=Constant(2), activation=Tanh())
        self.gated.initialize()
        self.reset_only = GatedRecurrent(
            dim=3, weights_init=IsotropicGaussian(), activation=Tanh(),
            use_update_gate=False, rng=numpy.random.RandomState(1))
        self.reset_only.initialize()

    def test_one_step(self):
        h0 = tensor.matrix('h0')
        x = tensor.matrix('x')
        z = tensor.matrix('z')
        r = tensor.matrix('r')
        h1 = self.gated.apply(x, z, r, h0, one_step=True)
        next_h = theano.function(inputs=[h0, x, z, r], outputs=[h1])

        h0_val = 0.1 * numpy.array([[1, 1, 0], [0, 1, 1]],
                                   dtype=floatX)
        x_val = 0.1 * numpy.array([[1, 2, 3], [4, 5, 6]],
                                  dtype=floatX)
        zi_val = (h0_val + x_val) / 2
        ri_val = -x_val
        W_val = 2 * numpy.ones((3, 3), dtype=floatX)

        z_val = sigmoid(h0_val.dot(W_val) + zi_val)
        r_val = sigmoid(h0_val.dot(W_val) + ri_val)
        h1_val = (z_val * numpy.tanh((r_val * h0_val).dot(W_val) + x_val)
                  + (1 - z_val) * h0_val)
        assert_allclose(h1_val, next_h(h0_val, x_val, zi_val, ri_val)[0],
                        rtol=1e-6)

    def test_reset_only_many_steps(self):
        x = tensor.tensor3('x')
        ri = tensor.tensor3('ri')
        mask = tensor.matrix('mask')
        h = self.reset_only.apply(x, reset_inps=ri, mask=mask)
        calc_h = theano.function(inputs=[x, ri, mask], outputs=[h])

        x_val = 0.1 * numpy.asarray(list(itertools.permutations(range(4))),
                                    dtype=floatX)
        x_val = numpy.ones((24, 4, 3), dtype=floatX) * x_val[..., None]
        ri_val = 0.3 - x_val
        mask_val = numpy.ones((24, 4), dtype=floatX)
        mask_val[12:24, 3] = 0
        h_val = numpy.zeros((25, 4, 3), dtype=floatX)
        W = self.reset_only.state2state.get_value()
        U = self.reset_only.state2reset.get_value()

        for i in range(1, 25):
            r_val = sigmoid(h_val[i - 1].dot(U) + ri_val[i - 1])
            h_val[i] = numpy.tanh((r_val * h_val[i - 1]).dot(W)
                                  + x_val[i - 1])
            h_val[i] = (mask_val[i - 1, :, None] * h_val[i] +
                        (1 - mask_val[i - 1, :, None]) * h_val[i - 1])
        h_val = h_val[1:]
        assert_allclose(h_val, calc_h(x_val, ri_val,  mask_val)[0])
