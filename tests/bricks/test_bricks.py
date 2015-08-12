import numpy
import six
import theano
from numpy.testing import assert_allclose, assert_raises
from theano import tensor

from blocks.bricks import (Identity, Linear, Maxout, LinearMaxout, MLP, Tanh,
                           Sequence, Random, Logistic, Softplus, Softmax)
from blocks.bricks.base import application, Brick, lazy, NoneAllocation
from blocks.bricks.parallel import Parallel, Fork
from blocks.filter import get_application_call, get_brick
from blocks.initialization import Constant
from blocks.utils import shared_floatx


class TestBrick(Brick):
    @lazy(allocation=['config'])
    def __init__(self, config, **kwargs):
        super(TestBrick, self).__init__(**kwargs)
        self.config = config

    @application
    def apply(self, x, y=1, **kwargs):
        if isinstance(x, list):
            x = x[0]
        return [x, y] + list(kwargs.values())

    @application(inputs=['x'], outputs=['y'])
    def second_apply(self, x):
        return x + 1

    @second_apply.property('all')
    def second_apply_all(self):
        return self.second_apply.inputs + self.second_apply.outputs

    @application
    def delegated_apply(self, x, w):
        pass

    @delegated_apply.delegate
    def delegate(self):
        return self.second_apply

    @application
    def access_application_call(self, x, application_call):
        application_call.add_auxiliary_variable(shared_floatx(numpy.ones((1,)),
                                                              name='test_val'))
        return x


class ParentBrick(Brick):
    def __init__(self, child=None, **kwargs):
        super(ParentBrick, self).__init__(**kwargs)
        self.child = child
        if child is None:
            child = TestBrick(0)
        self.children = [child]

    @application
    def apply(self, *args, **kwargs):
        return self.child.apply(*args, **kwargs)

    @application
    def second_apply(self, x):
        return x - 1

    @second_apply.property('inputs')
    def second_apply_inputs(self):
        return self.child.second_apply.all

    @second_apply.delegate
    def second_apply_delegate(self):
        return self.child.delegated_apply


class BrokenAllocateBrick(Brick):
    def _push_allocation_config(self):
        raise AttributeError

    def _allocate(self):
        raise AttributeError


class BrokenInitializeBrick(Brick):
    def _initialize(self):
        raise AttributeError


class ParameterBrick(Brick):
    def _allocate(self):
        self.parameters.append(
            theano.shared(numpy.zeros((10, 10), dtype=theano.config.floatX)))


def test_super():
    brick = TestBrick()
    assert isinstance(brick.name, six.string_types)
    assert brick.children == []
    assert not any([brick.allocated, brick.allocation_config_pushed,
                    brick.initialized, brick.initialization_config_pushed])

    parent_brick = ParentBrick()
    assert len(parent_brick.children) == 1

    brick = TestBrick(name='test_name')
    assert brick.name == 'test_name'


def test_repr():
    brick = TestBrick()
    assert 'name=testbrick' in repr(brick)
    assert hex(id(brick)) in repr(brick)
    assert str(brick) == repr(brick)


def test_lazy():
    linear = Linear()
    assert linear.allocation_args == ['input_dim', 'output_dim']
    brick = TestBrick()
    assert brick.config is NoneAllocation
    brick = TestBrick(config='config')
    assert brick.config == 'config'
    assert_raises(ValueError, TestBrick, 'config', config='config')


def test_allocate():
    brick = TestBrick(0)
    brick.allocate()
    assert brick.allocated
    assert brick.allocation_config_pushed

    parent_brick = ParentBrick()
    parent_brick.allocate()
    assert parent_brick.children[0].allocated
    assert parent_brick.children[0].allocation_config_pushed

    parameter_brick = ParameterBrick()
    assert not hasattr(parameter_brick, 'parameters')
    parameter_brick.allocate()
    assert len(parameter_brick.parameters) == 1
    parameter_brick.parameters[0].set_value(
        numpy.ones((10, 10), dtype=theano.config.floatX))
    parameter_brick.allocate()
    assert numpy.all(parameter_brick.parameters[0].get_value() == 0)

    broken_parent_brick = ParentBrick(BrokenAllocateBrick())
    assert_raises(AttributeError, broken_parent_brick.allocate)
    assert not broken_parent_brick.allocation_config_pushed
    assert not broken_parent_brick.allocated

    broken_parent_brick = ParentBrick(BrokenAllocateBrick())
    assert_raises(AttributeError, broken_parent_brick.allocate)
    assert not broken_parent_brick.allocation_config_pushed
    assert not broken_parent_brick.allocated


def test_initialize():
    brick = TestBrick(0)
    brick.initialize()

    parent_brick = ParentBrick()
    parent_brick.initialize()

    broken_parent_brick = ParentBrick(BrokenInitializeBrick())
    assert_raises(AttributeError, broken_parent_brick.initialize)

    broken_parent_brick = ParentBrick(BrokenInitializeBrick())
    assert_raises(AttributeError, broken_parent_brick.initialize)


def test_tagging():
    brick = TestBrick(0)
    x = tensor.vector('x')
    y = tensor.vector('y')
    z = tensor.vector('z')

    def check_output_variable(o):
        assert get_application_call(o).application.brick is brick
        assert (get_application_call(o.owner.inputs[0]).application.brick
                is brick)

    # Case 1: both positional arguments are provided.
    u, v = brick.apply(x, y)
    for o in [u, v]:
        check_output_variable(o)

    # Case 2: `b` is given as a keyword argument.
    u, v = brick.apply(x, y=y)
    for o in [u, v]:
        check_output_variable(o)

    # Case 3: two positional and one keyword argument.
    u, v, w = brick.apply(x, y, z=z)
    for o in [u, v, w]:
        check_output_variable(o)

    # Case 4: one positional argument.
    u, v = brick.apply(x)
    check_output_variable(u)
    assert v == 1

    # Case 5: variable was wrapped in a list. We can not handle that.
    u, v = brick.apply([x])
    assert_raises(AttributeError, check_output_variable, u)


def test_apply_not_child():
    child = TestBrick()
    parent = ParentBrick(child)
    parent.children = []
    assert_raises(ValueError, parent.apply, tensor.matrix())


def test_request_unknown_dimension():
    brick = TestBrick()
    assert_raises(ValueError, brick.get_dim, 'unknown')


def test_application():
    brick = TestBrick()
    assert brick.second_apply.inputs == ['x']
    assert brick.second_apply.outputs == ['y']

    assert brick.delegated_apply.inputs == ['x']
    assert brick.delegated_apply.outputs == ['y']

    assert brick.second_apply.all == ['x', 'y']

    brick.second_apply.inputs = ['x', 'z']
    assert brick.second_apply.inputs == ['x', 'z']
    assert brick.second_apply.all == ['x', 'z', 'y']

    brick.delegated_apply.outputs = ['z']
    assert brick.delegated_apply.outputs == ['z']
    assert brick.delegated_apply.inputs == ['x', 'z']

    parent_brick = ParentBrick(brick)
    parent_brick.second_apply.inputs = ['x', 'z', 'y']
    parent_brick.second_apply.inputs == ['x', 'z']

    assert_raises(AttributeError, setattr, TestBrick.second_apply, 'all', 'w')

    TestBrick.delegated_apply.inputs = ['w']
    assert TestBrick.delegated_apply.inputs == ['w']
    test_brick = TestBrick()
    assert test_brick.delegated_apply.inputs == ['w']
    test_brick.delegated_apply.inputs = ['x']
    assert test_brick.delegated_apply.inputs == ['x']
    assert TestBrick.delegated_apply.inputs == ['w']


def test_apply():
    brick = TestBrick(0)
    assert TestBrick.apply(brick, [0]) == [0, 1]
    if six.PY2:
        assert_raises(TypeError, TestBrick.apply, [0])


def test_rng():
    linear = Linear()
    assert isinstance(linear.rng, numpy.random.RandomState)
    linear = Linear(seed=1)
    assert linear.rng.rand() == numpy.random.RandomState(1).rand()
    linear = Linear()
    linear2 = Linear()
    assert linear.seed != linear2.seed


def test_random_brick():
    random = Random()
    # This makes sure that a Random brick doesn't instantiate more than one
    # Theano RNG during its lifetime (see PR #485 on Github)
    assert random.theano_rng is random.theano_rng


def test_linear():
    x = tensor.matrix()

    linear = Linear(input_dim=16, output_dim=8, weights_init=Constant(2),
                    biases_init=Constant(1))
    y = linear.apply(x)
    linear.initialize()
    x_val = numpy.ones((4, 16), dtype=theano.config.floatX)
    assert_allclose(
        y.eval({x: x_val}),
        x_val.dot(2 * numpy.ones((16, 8))) + numpy.ones((4, 8)))

    linear = Linear(input_dim=16, output_dim=8, weights_init=Constant(2),
                    use_bias=False)
    y = linear.apply(x)
    linear.initialize()
    x_val = numpy.ones((4, 16), dtype=theano.config.floatX)
    assert_allclose(y.eval({x: x_val}), x_val.dot(2 * numpy.ones((16, 8))))


def test_linear_maxout():
    x = tensor.matrix()

    linear_maxout = LinearMaxout(input_dim=16, output_dim=8, num_pieces=3,
                                 weights_init=Constant(2),
                                 biases_init=Constant(1))
    y = linear_maxout.apply(x)
    linear_maxout.initialize()
    x_val = numpy.ones((4, 16), dtype=theano.config.floatX)
    assert_allclose(
        y.eval({x: x_val}),
        (x_val.dot(2 * numpy.ones((16, 24))) +
            numpy.ones((4, 24))).reshape(4, 8, 3).max(2))


def test_maxout():
    x = tensor.tensor3()
    maxout = Maxout(num_pieces=3)
    y = maxout.apply(x)
    x_val = numpy.asarray(numpy.random.normal(0, 1, (4, 5, 24)),
                          dtype=theano.config.floatX)
    assert_allclose(
        y.eval({x: x_val}),
        x_val.reshape(4, 5, 8, 3).max(3))
    assert y.eval({x: x_val}).shape == (4, 5, 8)


def test_activations():
    x = tensor.vector()
    x_val = numpy.random.rand(8).astype(theano.config.floatX)
    exp_x_val = numpy.exp(x_val)

    assert_allclose(x_val, Identity().apply(x).eval({x: x_val}))
    assert_allclose(numpy.tanh(x_val), Tanh().apply(x).eval({x: x_val}),
                    rtol=1e-06)
    assert_allclose(numpy.log(1 + exp_x_val),
                    Softplus(x).apply(x).eval({x: x_val}), rtol=1e-6)
    assert_allclose(exp_x_val / numpy.sum(exp_x_val),
                    Softmax(x).apply(x).eval({x: x_val}).flatten(), rtol=1e-6)
    assert_allclose(1.0 / (1.0 + numpy.exp(-x_val)),
                    Logistic(x).apply(x).eval({x: x_val}), rtol=1e-6)


def test_mlp():
    x = tensor.matrix()
    x_val = numpy.random.rand(2, 16).astype(theano.config.floatX)
    mlp = MLP(activations=[Tanh(), None], dims=[16, 8, 4],
              weights_init=Constant(1), biases_init=Constant(1))
    y = mlp.apply(x)
    mlp.initialize()
    assert_allclose(
        numpy.tanh(x_val.dot(numpy.ones((16, 8))) + numpy.ones((2, 8))).dot(
            numpy.ones((8, 4))) + numpy.ones((2, 4)),
        y.eval({x: x_val}), rtol=1e-06)

    mlp = MLP(activations=[None], weights_init=Constant(1), use_bias=False)
    mlp.dims = [16, 8]
    y = mlp.apply(x)
    mlp.initialize()
    assert_allclose(x_val.dot(numpy.ones((16, 8))),
                    y.eval({x: x_val}), rtol=1e-06)
    assert mlp.rng == mlp.linear_transformations[0].rng


def test_mlp_apply():
    x = tensor.matrix()
    x_val = numpy.random.rand(2, 16).astype(theano.config.floatX)
    mlp = MLP(activations=[Tanh().apply, None], dims=[16, 8, 4],
              weights_init=Constant(1), biases_init=Constant(1))
    y = mlp.apply(x)
    mlp.initialize()
    assert_allclose(
        numpy.tanh(x_val.dot(numpy.ones((16, 8))) + numpy.ones((2, 8))).dot(
            numpy.ones((8, 4))) + numpy.ones((2, 4)),
        y.eval({x: x_val}), rtol=1e-06)

    mlp = MLP(activations=[None], weights_init=Constant(1), use_bias=False)
    mlp.dims = [16, 8]
    y = mlp.apply(x)
    mlp.initialize()
    assert_allclose(x_val.dot(numpy.ones((16, 8))),
                    y.eval({x: x_val}), rtol=1e-06)
    assert mlp.rng == mlp.linear_transformations[0].rng


def test_sequence():
    x = tensor.matrix()

    linear_1 = Linear(input_dim=16, output_dim=8, weights_init=Constant(2),
                      biases_init=Constant(1))

    linear_2 = Linear(input_dim=8, output_dim=4, weights_init=Constant(3),
                      biases_init=Constant(4))
    sequence = Sequence([linear_1.apply, linear_2.apply])
    sequence.initialize()
    y = sequence.apply(x)
    x_val = numpy.ones((4, 16), dtype=theano.config.floatX)
    assert_allclose(
        y.eval({x: x_val}),
        (x_val.dot(2 * numpy.ones((16, 8))) + numpy.ones((4, 8))).dot(
            3 * numpy.ones((8, 4))) + 4 * numpy.ones((4, 4)))


def test_sequence_variable_outputs():
    x = tensor.matrix()

    linear_1 = Linear(input_dim=16, output_dim=8, weights_init=Constant(2),
                      biases_init=Constant(1))

    fork = Fork(input_dim=8, output_names=['linear_2_1', 'linear_2_2'],
                output_dims=[4, 5], prototype=Linear(),
                weights_init=Constant(3), biases_init=Constant(4))
    sequence = Sequence([linear_1.apply, fork.apply])
    sequence.initialize()
    y_1, y_2 = sequence.apply(x)
    x_val = numpy.ones((4, 16), dtype=theano.config.floatX)
    assert_allclose(
        y_1.eval({x: x_val}),
        (x_val.dot(2 * numpy.ones((16, 8))) + numpy.ones((4, 8))).dot(
            3 * numpy.ones((8, 4))) + 4 * numpy.ones((4, 4)))
    assert_allclose(
        y_2.eval({x: x_val}),
        (x_val.dot(2 * numpy.ones((16, 8))) + numpy.ones((4, 8))).dot(
            3 * numpy.ones((8, 5))) + 4 * numpy.ones((4, 5)))


def test_sequence_variable_inputs():
    x, y = tensor.matrix(), tensor.matrix()

    parallel_1 = Parallel(input_names=['input_1', 'input_2'],
                          input_dims=[4, 5], output_dims=[3, 2],
                          prototype=Linear(), weights_init=Constant(2),
                          biases_init=Constant(1))
    parallel_2 = Parallel(input_names=['input_1', 'input_2'],
                          input_dims=[3, 2], output_dims=[5, 4],
                          prototype=Linear(), weights_init=Constant(2),
                          biases_init=Constant(1))
    sequence = Sequence([parallel_1.apply, parallel_2.apply])
    sequence.initialize()
    new_x, new_y = sequence.apply(x, y)
    x_val = numpy.ones((4, 4), dtype=theano.config.floatX)
    y_val = numpy.ones((4, 5), dtype=theano.config.floatX)
    assert_allclose(
        new_x.eval({x: x_val}),
        (x_val.dot(2 * numpy.ones((4, 3))) + numpy.ones((4, 3))).dot(
            2 * numpy.ones((3, 5))) + numpy.ones((4, 5)))
    assert_allclose(
        new_y.eval({y: y_val}),
        (y_val.dot(2 * numpy.ones((5, 2))) + numpy.ones((4, 2))).dot(
            2 * numpy.ones((2, 4))) + numpy.ones((4, 4)))


def test_application_call():
    X = tensor.matrix('X')
    brick = TestBrick(0)
    Y = brick.access_application_call(X)
    (auxiliary_variable,) = get_application_call(Y).auxiliary_variables
    assert auxiliary_variable.name == 'test_val'
    assert get_brick(auxiliary_variable) == brick
    assert get_application_call(Y).auxiliary_variables[0].name == 'test_val'


def test_linear_nan_allocation():
    x = tensor.matrix()

    linear = Linear(input_dim=16, output_dim=8, weights_init=Constant(2),
                    biases_init=Constant(1))
    linear.apply(x)
    w1 = numpy.nan * numpy.zeros((16, 8))
    w2 = linear.parameters[0].get_value()
    b1 = numpy.nan * numpy.zeros(8)
    b2 = linear.parameters[1].get_value()
    numpy.testing.assert_equal(w1, w2)
    numpy.testing.assert_equal(b1, b2)
