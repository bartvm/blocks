import numpy
import theano
from theano import tensor
from numpy.testing import assert_allclose, assert_raises

from blocks.bricks import MLP, Tanh
from blocks.model import Model

floatX = theano.config.floatX


def test_model():
    x = tensor.matrix('x')
    mlp1 = MLP([Tanh(), Tanh()], [10, 20, 30], name="mlp1")
    mlp2 = MLP([Tanh()], [30, 40], name="mlp2")
    h1 = mlp1.apply(x)
    h2 = mlp2.apply(h1)

    model = Model(h2.sum())
    assert model.get_top_bricks() == [mlp1, mlp2]
    # The order of parameters returned is deterministic but
    # not sensible.
    assert list(model.get_parameters(hierarchical_names=True).items()) == [
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
    param_values = {
        '/mlp/linear_0.W': 2 * numpy.ones((10, 10), dtype=floatX),
        '/mlp/linear_0.b': 3 * numpy.ones(10, dtype=floatX)}
    model3.set_param_values(param_values)
    assert numpy.all(mlp3.linear_transformations[0].params[0].get_value() == 2)
    assert numpy.all(mlp3.linear_transformations[0].params[1].get_value() == 3)
    got_param_values = model3.get_param_values()
    assert len(got_param_values) == len(param_values)
    for name, value in param_values.items():
        assert_allclose(value, got_param_values[name])

    # Test name conflict handling
    mlp4 = MLP([Tanh()], [10, 10])

    def helper():
        Model(mlp4.apply(mlp3.apply(x)))
    assert_raises(ValueError, helper)
