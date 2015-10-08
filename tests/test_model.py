import numpy
import theano
from theano import tensor
from numpy.testing import assert_allclose, assert_raises

from blocks.bricks import MLP, Tanh
from blocks.model import Model
from blocks.graph import add_role, PARAMETER
from blocks.utils import shared_floatx


def test_model():
    x = tensor.matrix('x')
    mlp1 = MLP([Tanh(), Tanh()], [10, 20, 30], name="mlp1")
    mlp2 = MLP([Tanh()], [30, 40], name="mlp2")
    h1 = mlp1.apply(x)
    h2 = mlp2.apply(h1)

    model = Model(h2)
    assert model.get_top_bricks() == [mlp1, mlp2]
    # The order of parameters returned is deterministic but
    # not sensible.
    assert list(model.get_parameter_dict().items()) == [
        ('/mlp2/linear_0.b', mlp2.linear_transformations[0].b),
        ('/mlp1/linear_1.b', mlp1.linear_transformations[1].b),
        ('/mlp1/linear_0.b', mlp1.linear_transformations[0].b),
        ('/mlp1/linear_0.W', mlp1.linear_transformations[0].W),
        ('/mlp1/linear_1.W', mlp1.linear_transformations[1].W),
        ('/mlp2/linear_0.W', mlp2.linear_transformations[0].W)]

    # Test getting and setting parameter values
    mlp3 = MLP([Tanh()], [10, 10])
    mlp3.allocate()
    model3 = Model(mlp3.apply(x))
    parameter_values = {
        '/mlp/linear_0.W': 2 * numpy.ones((10, 10),
                                          dtype=theano.config.floatX),
        '/mlp/linear_0.b': 3 * numpy.ones(10, dtype=theano.config.floatX)}
    model3.set_parameter_values(parameter_values)
    assert numpy.all(
        mlp3.linear_transformations[0].parameters[0].get_value() == 2)
    assert numpy.all(
        mlp3.linear_transformations[0].parameters[1].get_value() == 3)
    got_parameter_values = model3.get_parameter_values()
    assert len(got_parameter_values) == len(parameter_values)
    for name, value in parameter_values.items():
        assert_allclose(value, got_parameter_values[name])

    # Test exception is raised if parameter shapes don't match
    def helper():
        parameter_values = {
            '/mlp/linear_0.W': 2 * numpy.ones((11, 11),
                                              dtype=theano.config.floatX),
            '/mlp/linear_0.b': 3 * numpy.ones(11, dtype=theano.config.floatX)}
        model3.set_parameter_values(parameter_values)
    assert_raises(ValueError, helper)

    # Test name conflict handling
    mlp4 = MLP([Tanh()], [10, 10])

    def helper():
        Model(mlp4.apply(mlp3.apply(x)))
    assert_raises(ValueError, helper)


def test_model_handles_brickless_parameteres():
    x = tensor.matrix('x')
    v = shared_floatx(numpy.zeros((10, 10)), name='V')
    add_role(v, PARAMETER)
    y = x.dot(v)
    model = Model(y)
    assert list(model.get_parameter_dict().items()) == [('V', v)]
