import itertools
import unittest

import numpy
import theano
from numpy.testing import assert_allclose
from theano import tensor
from theano.gof.graph import is_same_graph

from blocks.bricks.base import application
from blocks.bricks import Tanh
from blocks.bricks.recurrent import (
    recurrent, BaseRecurrent, GatedRecurrent,
    SimpleRecurrent, Bidirectional, LSTM)
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal
from blocks.filter import get_application_call, VariableFilter
from blocks.graph import ComputationGraph


floatX = theano.config.floatX


class RecurrentWrapperTestClass(BaseRecurrent):
    def __init__(self, dim, ** kwargs):
        super(RecurrentWrapperTestClass, self).__init__(self, ** kwargs)
        self.dim = dim

    def get_dim(self, name):
        if name in ['inputs', 'states', 'outputs', 'states_2', 'outputs_2']:
            return self.dim
        if name == 'mask':
            return 0
        return super(RecurrentWrapperTestClass, self).get_dim(name)

    @recurrent(sequences=['inputs', 'mask'], states=['states', 'states_2'],
               outputs=['outputs', 'states_2', 'outputs_2', 'states'],
               contexts=[])
    def apply(self, inputs=None, states=None, states_2=None, mask=None):
        next_states = states + inputs
        next_states_2 = states_2 + .5
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        outputs = 10 * next_states
        outputs_2 = 10 * next_states_2
        return outputs, next_states_2, outputs_2, next_states


class TestRecurrentWrapper(unittest.TestCase):
    def setUp(self):
        self.recurrent_example = RecurrentWrapperTestClass(dim=1)

    def test(self):
        X = tensor.tensor3('X')
        out, H2, out_2, H = self.recurrent_example.apply(
            inputs=X, mask=None)

        x_val = numpy.ones((5, 1, 1), dtype=floatX)

        h = H.eval({X: x_val})
        h2 = H2.eval({X: x_val})

        out_eval = out.eval({X: x_val})
        out_2_eval = out_2.eval({X: x_val})

        assert_allclose(h, x_val.cumsum(axis=0))
        assert_allclose(h2, .5 * (numpy.arange(5).reshape((5, 1, 1)) + 1))
        assert_allclose(h * 10, out_eval)
        assert_allclose(h2 * 10, out_2_eval)


class TestRecurrent(unittest.TestCase):
    def setUp(self):
        self.simple = SimpleRecurrent(dim=3, weights_init=Constant(2),
                                      activation=Tanh())
        self.simple.initialize()

    def test_one_step(self):
        h0 = tensor.matrix('h0')
        x = tensor.matrix('x')
        mask = tensor.vector('mask')
        h1 = self.simple.apply(x, h0, mask=mask, iterate=False)
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
        h = self.simple.apply(x, mask=mask, iterate=True)
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
        assert_allclose(h_val, calc_h(x_val, mask_val)[0], rtol=1e-04)


class TestLSTM(unittest.TestCase):
    def setUp(self):
        self.lstm = LSTM(dim=3, weights_init=Constant(2),
                         biases_init=Constant(0))
        self.lstm.initialize()

    def test_one_step(self):
        h0 = tensor.matrix('h0')
        c0 = tensor.matrix('h0')
        x = tensor.matrix('x')
        h1, c1 = self.lstm.apply(x, h0, c0, iterate=False)
        next_h = theano.function(inputs=[x, h0, c0], outputs=[h1])

        h0_val = 0.1 * numpy.array([[1, 1, 0], [0, 1, 1]],
                                   dtype=floatX)
        c0_val = 0.1 * numpy.array([[1, 1, 0], [0, 1, 1]],
                                   dtype=floatX)
        x_val = 0.1 * numpy.array([range(12), range(12, 24)],
                                  dtype=floatX)
        W_state_val = 2 * numpy.ones((3, 12), dtype=floatX)
        W_cell_to_in = 2 * numpy.ones((3,), dtype=floatX)
        W_cell_to_out = 2 * numpy.ones((3,), dtype=floatX)
        W_cell_to_forget = 2 * numpy.ones((3,), dtype=floatX)

        # omitting biases because they are zero
        activation = numpy.dot(h0_val, W_state_val) + x_val

        def sigmoid(x):
            return 1. / (1. + numpy.exp(-x))

        i_t = sigmoid(activation[:, :3] + c0_val * W_cell_to_in)
        f_t = sigmoid(activation[:, 3:6] + c0_val * W_cell_to_forget)
        next_cells = f_t * c0_val + i_t * numpy.tanh(activation[:, 6:9])
        o_t = sigmoid(activation[:, 9:12] +
                      next_cells * W_cell_to_out)
        h1_val = o_t * numpy.tanh(next_cells)
        assert_allclose(h1_val, next_h(x_val, h0_val, c0_val)[0],
                        rtol=1e-6)

    def test_many_steps(self):
        x = tensor.tensor3('x')
        mask = tensor.matrix('mask')
        h, c = self.lstm.apply(x, mask=mask, iterate=True)
        calc_h = theano.function(inputs=[x, mask], outputs=[h])

        x_val = (0.1 * numpy.asarray(
            list(itertools.islice(itertools.permutations(range(12)), 0, 24)),
            dtype=floatX))
        x_val = numpy.ones((24, 4, 12),
                           dtype=floatX) * x_val[:, None, :]
        mask_val = numpy.ones((24, 4), dtype=floatX)
        mask_val[12:24, 3] = 0
        h_val = numpy.zeros((25, 4, 3), dtype=floatX)
        c_val = numpy.zeros((25, 4, 3), dtype=floatX)
        W_state_val = 2 * numpy.ones((3, 12), dtype=floatX)
        W_cell_to_in = 2 * numpy.ones((3,), dtype=floatX)
        W_cell_to_out = 2 * numpy.ones((3,), dtype=floatX)
        W_cell_to_forget = 2 * numpy.ones((3,), dtype=floatX)

        def sigmoid(x):
            return 1. / (1. + numpy.exp(-x))

        for i in range(1, 25):
            activation = numpy.dot(h_val[i-1], W_state_val) + x_val[i-1]
            i_t = sigmoid(activation[:, :3] + c_val[i-1] * W_cell_to_in)
            f_t = sigmoid(activation[:, 3:6] + c_val[i-1] * W_cell_to_forget)
            c_val[i] = f_t * c_val[i-1] + i_t * numpy.tanh(activation[:, 6:9])
            o_t = sigmoid(activation[:, 9:12] +
                          c_val[i] * W_cell_to_out)
            h_val[i] = o_t * numpy.tanh(c_val[i])
            h_val[i] = (mask_val[i - 1, :, None] * h_val[i] +
                        (1 - mask_val[i - 1, :, None]) * h_val[i - 1])
            c_val[i] = (mask_val[i - 1, :, None] * c_val[i] +
                        (1 - mask_val[i - 1, :, None]) * c_val[i - 1])

        h_val = h_val[1:]
        assert_allclose(h_val, calc_h(x_val, mask_val)[0], rtol=1e-04)


class TestGatedRecurrent(unittest.TestCase):
    def setUp(self):
        self.gated = GatedRecurrent(
            dim=3, activation=Tanh(),
            gate_activation=Tanh(), weights_init=Constant(2))
        self.gated.initialize()
        self.reset_only = GatedRecurrent(
            dim=3, activation=Tanh(),
            gate_activation=Tanh(), use_update_gate=False,
            weights_init=IsotropicGaussian(), seed=1)
        self.reset_only.initialize()

    def test_one_step(self):
        h0 = tensor.matrix('h0')
        x = tensor.matrix('x')
        z = tensor.matrix('z')
        r = tensor.matrix('r')
        h1 = self.gated.apply(x, z, r, h0, iterate=False)
        next_h = theano.function(inputs=[h0, x, z, r], outputs=[h1])

        h0_val = 0.1 * numpy.array([[1, 1, 0], [0, 1, 1]],
                                   dtype=floatX)
        x_val = 0.1 * numpy.array([[1, 2, 3], [4, 5, 6]],
                                  dtype=floatX)
        zi_val = (h0_val + x_val) / 2
        ri_val = -x_val
        W_val = 2 * numpy.ones((3, 3), dtype=floatX)

        z_val = numpy.tanh(h0_val.dot(W_val) + zi_val)
        r_val = numpy.tanh(h0_val.dot(W_val) + ri_val)
        h1_val = (z_val * numpy.tanh((r_val * h0_val).dot(W_val) + x_val) +
                  (1 - z_val) * h0_val)
        assert_allclose(h1_val, next_h(h0_val, x_val, zi_val, ri_val)[0],
                        rtol=1e-6)

    def test_reset_only_many_steps(self):
        x = tensor.tensor3('x')
        ri = tensor.tensor3('ri')
        mask = tensor.matrix('mask')
        h = self.reset_only.apply(x, reset_inputs=ri, mask=mask)
        calc_h = theano.function(inputs=[x, ri, mask], outputs=[h])

        x_val = 0.1 * numpy.asarray(list(itertools.permutations(range(4))),
                                    dtype=floatX)
        x_val = numpy.ones((24, 4, 3), dtype=floatX) * x_val[..., None]
        ri_val = 0.3 - x_val
        mask_val = numpy.ones((24, 4), dtype=floatX)
        mask_val[12:24, 3] = 0
        h_val = numpy.zeros((25, 4, 3), dtype=floatX)
        W = self.reset_only.state_to_state.get_value()
        U = self.reset_only.state_to_reset.get_value()

        for i in range(1, 25):
            r_val = numpy.tanh(h_val[i - 1].dot(U) + ri_val[i - 1])
            h_val[i] = numpy.tanh((r_val * h_val[i - 1]).dot(W) +
                                  x_val[i - 1])
            h_val[i] = (mask_val[i - 1, :, None] * h_val[i] +
                        (1 - mask_val[i - 1, :, None]) * h_val[i - 1])
        h_val = h_val[1:]
        # TODO Figure out why this tolerance needs to be so big
        assert_allclose(h_val, calc_h(x_val, ri_val,  mask_val)[0], 1e-03)


class TestBidirectional(unittest.TestCase):
    def setUp(self):
        self.bidir = Bidirectional(weights_init=Orthogonal(),
                                   prototype=SimpleRecurrent(
                                       dim=3, activation=Tanh()))
        self.simple = SimpleRecurrent(dim=3, weights_init=Orthogonal(),
                                      activation=Tanh(), seed=1)
        self.bidir.allocate()
        self.simple.initialize()
        self.bidir.children[0].params[0].set_value(
            self.simple.params[0].get_value())
        self.bidir.children[1].params[0].set_value(
            self.simple.params[0].get_value())
        self.x_val = 0.1 * numpy.asarray(
            list(itertools.permutations(range(4))),
            dtype=floatX)
        self.x_val = (numpy.ones((24, 4, 3), dtype=floatX) *
                      self.x_val[..., None])
        self.mask_val = numpy.ones((24, 4), dtype=floatX)
        self.mask_val[12:24, 3] = 0

    def test(self):
        x = tensor.tensor3('x')
        mask = tensor.matrix('mask')
        calc_bidir = theano.function([x, mask],
                                     [self.bidir.apply(x, mask=mask)])
        calc_simple = theano.function([x, mask],
                                      [self.simple.apply(x, mask=mask)])
        h_bidir = calc_bidir(self.x_val, self.mask_val)[0]
        h_simple = calc_simple(self.x_val, self.mask_val)[0]
        h_simple_rev = calc_simple(self.x_val[::-1], self.mask_val[::-1])[0]

        output_names = self.bidir.apply.outputs

        assert output_names == ['states']
        assert_allclose(h_simple, h_bidir[..., :3], rtol=1e-04)
        assert_allclose(h_simple_rev, h_bidir[::-1, ...,  3:], rtol=1e-04)


def test_saved_inner_graph():
    """Make sure that the original inner graph is saved."""
    x = tensor.tensor3()
    recurrent = SimpleRecurrent(dim=3, activation=Tanh())
    y = recurrent.apply(x)

    application_call = get_application_call(y)
    assert application_call.inner_inputs
    assert application_call.inner_outputs

    cg = ComputationGraph(application_call.inner_outputs)
    # Check that the inner scan graph is annotated
    # with `recurrent.apply`
    assert len(VariableFilter(applications=[recurrent.apply])(cg)) == 3
    # Check that the inner graph is equivalent to the one
    # produced by a stand-alone of `recurrent.apply`
    assert is_same_graph(application_call.inner_outputs[0],
                         recurrent.apply(*application_call.inner_inputs,
                                         iterate=False))


def test_super_in_recurrent_overrider():
    # A regression test for the issue #475
    class SimpleRecurrentWithContext(SimpleRecurrent):
        @application(contexts=['context'])
        def apply(self, context, *args, **kwargs):
            kwargs['inputs'] += context
            return super(SimpleRecurrentWithContext, self).apply(*args,
                                                                 **kwargs)

        @apply.delegate
        def apply_delegate(self):
            return super(SimpleRecurrentWithContext, self).apply

    brick = SimpleRecurrentWithContext(100, Tanh())
    inputs = tensor.tensor3('inputs')
    context = tensor.matrix('context').dimshuffle('x', 0, 1)
    brick.apply(context, inputs=inputs)
