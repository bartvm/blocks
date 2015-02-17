from collections import OrderedDict

import numpy
import theano
from numpy.testing import assert_allclose, assert_raises
from theano import tensor

from blocks.algorithms import (GradientDescent, StepClipping, CompositeRule,
                               Scale, StepRule, BasicMomentum, Momentum,
                               AdaDelta, BasicRMSProp, RMSProp)
from blocks.utils import shared_floatx


def test_gradient_descent():
    W = shared_floatx(numpy.array([[1, 2], [3, 4]]))
    W_start_value = W.get_value()
    cost = tensor.sum(W ** 2)

    algorithm = GradientDescent(cost=cost, params=[W])
    algorithm.step_rule.learning_rate.set_value(0.75)
    algorithm.initialize()
    algorithm.process_batch(dict())
    assert_allclose(W.get_value(), -0.5 * W_start_value)


def test_gradient_descent_with_gradients():
    W = shared_floatx(numpy.array([[1, 2], [3, 4]]))
    W_start_value = W.get_value()
    cost = tensor.sum(W ** 2)
    gradients = OrderedDict()
    gradients[W] = tensor.grad(cost, W)

    algorithm = GradientDescent(cost=cost, gradients=gradients)
    algorithm.step_rule.learning_rate.set_value(0.75)
    algorithm.initialize()
    algorithm.process_batch(dict())
    assert_allclose(W.get_value(), -0.5 * W_start_value)


def test_basic_momentum():
    a = shared_floatx([3, 4])
    cost = (a ** 2).sum()
    steps, updates = BasicMomentum(0.5).compute_steps(
        OrderedDict([(a, tensor.grad(cost, a))]))
    f = theano.function([], [steps[a]], updates=updates)
    assert_allclose(f()[0], [6., 8.])
    assert_allclose(f()[0], [9., 12.])
    assert_allclose(f()[0], [10.5, 14.])


def test_momentum():
    a = shared_floatx([3, 4])
    cost = (a ** 2).sum()
    steps, updates = Momentum(0.1, 0.5).compute_steps(
        OrderedDict([(a, tensor.grad(cost, a))]))
    f = theano.function([], [steps[a]], updates=updates)
    assert_allclose(f()[0], [0.6, 0.8])
    assert_allclose(f()[0], [0.9, 1.2])
    assert_allclose(f()[0], [1.05, 1.4])


def test_adadelta():
    a = shared_floatx([3, 4])
    cost = (a ** 2).sum()
    steps, updates = AdaDelta(decay_rate=0.5, epsilon=1e-7).compute_steps(
        OrderedDict([(a, tensor.grad(cost, a))]))
    f = theano.function([], [steps[a]], updates=updates)
    assert_allclose(f()[0], [0.00044721, 0.00044721], rtol=1e-5)
    assert_allclose(f()[0], [0.0005164, 0.0005164], rtol=1e-5)
    assert_allclose(f()[0], [0.00056904, 0.00056904], rtol=1e-5)


def test_adadelta_decay_rate_sanity_check():
    assert_raises(ValueError, AdaDelta, -1.0)
    assert_raises(ValueError, AdaDelta, 2.0)


def test_basicrmsprop():
    a = shared_floatx([3, 4])
    cost = (a ** 2).sum()
    step_rule = BasicRMSProp(decay_rate=0.5, max_scaling=1e5)
    steps, updates = step_rule.compute_steps(
        OrderedDict([(a, tensor.grad(cost, a))]))
    f = theano.function([], [steps[a]], updates=updates)
    assert_allclose(f()[0], [1.41421356, 1.41421356])
    a.set_value([2, 3])
    assert_allclose(f()[0], [0.9701425, 1.02899151])
    a.set_value([1, 1.5])
    assert_allclose(f()[0], [0.6172134, 0.64699664])


def test_basicrmsprop_max_scaling():
    a = shared_floatx([1e-6, 1e-6])
    cost = (a ** 2).sum()
    step_rule = BasicRMSProp(decay_rate=0.5, max_scaling=1e5)
    steps, updates = step_rule.compute_steps(
        OrderedDict([(a, tensor.grad(cost, a))]))
    f = theano.function([], [steps[a]], updates=updates)
    assert_allclose(f()[0], [0.2, 0.2])


def test_basicrmsprop_decay_rate_sanity_check():
    assert_raises(ValueError, BasicRMSProp, -1.0)
    assert_raises(ValueError, BasicRMSProp, 2.0)


def test_basicrmsprop_max_scaling_sanity_check():
    assert_raises(ValueError, BasicRMSProp, 0.5, -1.0)


def test_rmsprop():
    a = shared_floatx([3, 4])
    cost = (a ** 2).sum()
    step_rule = RMSProp(learning_rate=0.1, decay_rate=0.5, max_scaling=1e5)
    steps, updates = step_rule.compute_steps(
        OrderedDict([(a, tensor.grad(cost, a))]))
    f = theano.function([], [steps[a]], updates=updates)
    assert_allclose(f()[0], [0.141421356, 0.141421356])
    a.set_value([2, 3])
    assert_allclose(f()[0], [0.09701425, 0.102899151])
    a.set_value([1, 1.5])
    assert_allclose(f()[0], [0.06172134, 0.064699664])


def test_step_clipping():
    rule1 = StepClipping(4)
    rule2 = StepClipping(5)

    gradients = {0: shared_floatx(3.0), 1: shared_floatx(4.0)}
    clipped1, _ = rule1.compute_steps(gradients)
    assert_allclose(clipped1[0].eval(), 12 / 5.0)
    assert_allclose(clipped1[1].eval(), 16 / 5.0)
    clipped2, _ = rule2.compute_steps(gradients)
    assert_allclose(clipped2[0].eval(), 3.0)
    assert_allclose(clipped2[1].eval(), 4.0)


def test_composite_rule():
    rule = CompositeRule([StepClipping(4), Scale(0.1)])
    gradients = {0: shared_floatx(3.0), 1: shared_floatx(4.0)}
    result, _ = rule.compute_steps(gradients)
    assert_allclose(result[0].eval(), 12 / 50.0)
    assert_allclose(result[1].eval(), 16 / 50.0)

    class RuleWithUpdates(StepRule):
        def __init__(self, updates):
            self.updates = updates

        def compute_steps(self, previous_steps):
            return previous_steps, self.updates

    rule = CompositeRule([RuleWithUpdates([(1, 2)]),
                          RuleWithUpdates([(3, 4)])])
    assert rule.compute_steps(None)[1] == [(1, 2), (3, 4)]
