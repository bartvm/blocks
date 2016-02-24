from collections import OrderedDict

import numpy
import theano
from numpy.testing import assert_allclose, assert_raises
from theano import tensor
from theano.compile.profiling import ProfileStats

from blocks.algorithms import (GradientDescent, StepClipping, VariableClipping,
                               CompositeRule, Scale, StepRule, BasicMomentum,
                               Momentum, AdaDelta, BasicRMSProp, RMSProp, Adam,
                               AdaGrad, RemoveNotFinite, Restrict)
from blocks.utils import shared_floatx, shared_floatx_zeros


def verify_broadcastable_handling(step_rule):
    def check(param):
        grad = tensor.grad(param.sum(), wrt=param)
        step, _ = step_rule.compute_steps(OrderedDict([(param, grad)]))
        assert step[param].broadcastable == grad.broadcastable

    check(shared_floatx_zeros((5, 6, 1, 5),
                              broadcastable=(False, False, True, False)))
    check(shared_floatx_zeros((2, 1, 3),
                              broadcastable=(False, True, False)))
    check(shared_floatx_zeros((3, 4, 1),
                              broadcastable=(False, False, True)))
    check(shared_floatx_zeros((1, 9, 6),
                              broadcastable=(True, False, False)))
    check(shared_floatx_zeros((1, 1, 1),
                              broadcastable=(True, True, True)))
    check(shared_floatx_zeros((1, 5, 1),
                              broadcastable=(True, False, True)))


def test_gradient_descent():
    W = shared_floatx(numpy.array([[1, 2], [3, 4]]))
    W_start_value = W.get_value()
    cost = tensor.sum(W ** 2)

    algorithm = GradientDescent(cost=cost, parameters=[W])
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


def test_gradient_descent_spurious_sources():
    W = shared_floatx(numpy.array([[1, 2], [3, 4]]))
    W_start_value = W.get_value()
    cost = tensor.sum(W ** 2)

    algorithm = GradientDescent(cost=cost, parameters=[W])
    algorithm.step_rule.learning_rate.set_value(0.75)
    algorithm.initialize()
    assert_raises(lambda:
                  algorithm.process_batch(dict(example_id='test')))

    algorithm = GradientDescent(cost=cost, parameters=[W],
                                on_unused_sources='ignore')
    algorithm.step_rule.learning_rate.set_value(0.75)
    algorithm.initialize()
    algorithm.process_batch(dict(example_id='test'))
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


def test_basic_momentum_broadcastable():
    verify_broadcastable_handling(BasicMomentum(0.5))


def test_momentum():
    a = shared_floatx([3, 4])
    cost = (a ** 2).sum()
    steps, updates = Momentum(0.1, 0.5).compute_steps(
        OrderedDict([(a, tensor.grad(cost, a))]))
    f = theano.function([], [steps[a]], updates=updates)
    assert_allclose(f()[0], [0.6, 0.8])
    assert_allclose(f()[0], [0.9, 1.2])
    assert_allclose(f()[0], [1.05, 1.4])


def test_momentum_broadcastable():
    verify_broadcastable_handling(Momentum(0.5))


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


def test_basicrmsprop_broadcastable():
    verify_broadcastable_handling(BasicRMSProp(0.5, 1e5))


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


def test_rmsprop_broadcastable():
    verify_broadcastable_handling(RMSProp(0.1, 0.5, 1e5))


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


def test_step_clipping_broadcastable():
    verify_broadcastable_handling(StepClipping(0.4))


def test_variable_clipping():
    # Test simple variable clipping with no axis.
    rule1 = VariableClipping(5)

    gradients = OrderedDict([(shared_floatx([1, 1]), shared_floatx([3, 2])),
                             (shared_floatx([-1, -1, -1]),
                              shared_floatx([[3, 9, 2]])),
                             (shared_floatx([[[1], [-1], [1], [-1]]]),
                              shared_floatx([[[1], [2], [3], [2]]]))])
    steps, _ = rule1.compute_steps(gradients)
    border, clipped, notclipped = steps.items()
    assert_allclose(border[1].eval(), [3, 2])
    assert_allclose(clipped[1].eval(),
                    numpy.array([[0.78885438, 3.47213595, 0.34164079]]),
                    rtol=1e-5)
    assert_allclose(notclipped[1].eval(), [[[1], [2], [3], [2]]])

    # Test variable clipping on one axis.
    rule2 = VariableClipping(10, axis=1)
    gradients = {shared_floatx([[1, -1, 1, -1], [-1, 1, -1, 1]]):
                 shared_floatx([[1, 2, 3, 4], [5, 6, 7, 8]])}
    steps, _ = rule2.compute_steps(gradients)
    clipped, = steps.items()
    assert_allclose(clipped[1].eval(),
                    [[1, 2, 3, 4],
                     [3.54858826, 4.79049022, 5.06478435, 6.30668631]],
                    rtol=1e-5)

    # Test variable clipping on two axis.
    rule3 = VariableClipping(10, axis=(1, 2))
    gradients = {shared_floatx([[[[1], [-1]],
                                 [[-1], [1]]],
                                [[[-1], [1]],
                                 [[2], [-1]]]]):
                 shared_floatx([[[[1], [2]],
                                 [[3], [4]]],
                                [[[5], [6]],
                                 [[7], [8]]]])}
    steps, _ = rule3.compute_steps(gradients)
    clipped, = steps.items()
    assert_allclose(clipped[1].eval(),
                    [[[[1], [2]],
                      [[3], [4]]],
                     [[[3.6429394], [4.86911616]],
                      [[5.86911616], [5.96440909]]]],
                    rtol=1e-5)

    # Test exceptions.
    assert_raises(ValueError, rule3.compute_steps, {0: shared_floatx([1.0])})
    assert_raises(ValueError, VariableClipping, 50, axis=(0, 0))


def test_variable_clipping_broadcastable():
    verify_broadcastable_handling(VariableClipping(1))


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


def test_adam():
    a = shared_floatx([3, 4])
    cost = (a ** 2).sum()
    steps, updates = Adam().compute_steps(
        OrderedDict([(a, tensor.grad(cost, a))]))
    f = theano.function([], [steps[a]], updates=updates)

    rtol = 1e-4
    assert_allclose(f()[0], [0.002, 0.002], rtol=rtol)
    a.set_value([2, 3])
    assert_allclose(f()[0], [0.0019407, 0.00196515], rtol=rtol)
    a.set_value([1, 1.5])
    assert_allclose(f()[0], [0.00178724, 0.0018223], rtol=rtol)


def test_adam_broadcastable():
    verify_broadcastable_handling(Adam())


def test_adagrad():
    a = shared_floatx([3, 4])
    cost = (a ** 2).sum()
    steps, updates = AdaGrad().compute_steps(
        OrderedDict([(a, tensor.grad(cost, a))]))
    f = theano.function([], [steps[a]], updates=updates)

    rtol = 1e-4
    assert_allclose(f()[0], [0.002,  0.002], rtol=rtol)
    a.set_value([2, 3])
    assert_allclose(f()[0], [0.0011094,  0.0012], rtol=rtol)
    a.set_value([1, 1.5])
    assert_allclose(f()[0], [0.00053452,  0.0005747], rtol=rtol)


def test_adagrad_broadcastable():
    verify_broadcastable_handling(AdaGrad())


def test_remove_not_finite():
    rule1 = RemoveNotFinite(0.1)
    rule2 = RemoveNotFinite()

    gradients = {1: shared_floatx(numpy.nan), 2: shared_floatx(numpy.inf),
                 3: 0.123}
    rval1, _ = rule1.compute_steps(gradients)
    assert_allclose(rval1[1].eval(), 0.9)
    assert_allclose(rval1[2].eval(), 1.8)
    assert_allclose(rval1[3].eval(), 0.123)
    rval2, _ = rule2.compute_steps(gradients)
    assert_allclose(rval2[1].eval(), 0.0)
    assert_allclose(rval2[2].eval(), 0.0)


def test_remove_not_finite_broadcastable():
    verify_broadcastable_handling(RemoveNotFinite())
    verify_broadcastable_handling(RemoveNotFinite(0.1))


class DummyUpdatesStepRule(StepRule):
    def compute_step(self, parameter, previous_step):
        return previous_step + 2, [(parameter * 10, parameter * 100)]


def test_restrict():
    rule1 = Scale(0.1)
    rule2 = Restrict(rule1, (1, 4))
    rval, _ = rule2.compute_steps(OrderedDict((i, shared_floatx(i * i))
                                              for i in range(6)))
    assert_allclose(rval[0].eval(), 0.0)
    assert_allclose(rval[1].eval(), 0.1)
    assert_allclose(rval[2].eval(), 4.0)
    assert_allclose(rval[3].eval(), 9.0)
    assert_allclose(rval[4].eval(), 1.6)
    assert_allclose(rval[5].eval(), 25.0)

    steps, updates = Restrict(DummyUpdatesStepRule(), (1, 4)).compute_steps(
        OrderedDict((i, shared_floatx(i * i)) for i in range(6)))

    assert_allclose(steps[0].eval(), 0.0)
    assert_allclose(steps[1].eval(), 3.0)
    assert_allclose(steps[2].eval(), 4.0)
    assert_allclose(steps[3].eval(), 9.0)
    assert_allclose(steps[4].eval(), 18.0)
    assert_allclose(steps[5].eval(), 25.0)

    assert updates == [(10, 100), (40, 400)]


def test_theano_profile_for_sgd_function():
    W = shared_floatx(numpy.array([[1, 2], [3, 4]]))
    W_start_value = W.get_value()
    cost = tensor.sum(W ** 2)

    algorithm = GradientDescent(
        cost=cost, parameters=[W], theano_func_kwargs={'profile': True})
    algorithm.step_rule.learning_rate.set_value(0.75)
    algorithm.initialize()
    algorithm.process_batch(dict())
    assert_allclose(W.get_value(), -0.5 * W_start_value)
    assert isinstance(algorithm._function.profile, ProfileStats)
