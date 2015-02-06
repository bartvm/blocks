from collections import OrderedDict

import numpy
import theano
from numpy.testing import assert_allclose
from theano import tensor

from blocks.algorithms import (GradientDescent, GradientClipping,
                               CompositeRule, SteepestDescent,
                               StepRule, Momentum)
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


def test_momentum():
    a = shared_floatx([3, 4])
    cost = (a ** 2).sum()
    steps, updates = Momentum(0.5).compute_steps(
        OrderedDict([(a, tensor.grad(cost, a))]))
    f = theano.function([], [steps[a]], updates=updates)
    assert_allclose(f()[0], [6., 8.])
    assert_allclose(f()[0], [9., 12.])
    assert_allclose(f()[0], [10.5, 14.])


def test_gradient_clipping():
    rule1 = GradientClipping(4)
    rule2 = GradientClipping(5)

    gradients = {0: shared_floatx(3.0), 1: shared_floatx(4.0)}
    clipped1, _ = rule1.compute_steps(gradients)
    assert_allclose(clipped1[0].eval(), 12 / 5.0)
    assert_allclose(clipped1[1].eval(), 16 / 5.0)
    clipped2, _ = rule2.compute_steps(gradients)
    assert_allclose(clipped2[0].eval(), 3.0)
    assert_allclose(clipped2[1].eval(), 4.0)


def test_composite_rule():
    rule = CompositeRule([GradientClipping(4), SteepestDescent(0.1)])
    gradients = {0: shared_floatx(3.0), 1: shared_floatx(4.0)}
    result, _ = rule.compute_steps(gradients)
    assert_allclose(result[0].eval(), -12 / 50.0)
    assert_allclose(result[1].eval(), -16 / 50.0)

    class RuleWithUpdates(StepRule):
        def __init__(self, updates):
            self.updates = updates

        def compute_steps(self, gradients):
            return gradients, self.updates

    rule = CompositeRule([RuleWithUpdates([(1, 2)]),
                          RuleWithUpdates([(3, 4)])])
    assert rule.compute_steps(None)[1] == [(1, 2), (3, 4)]
